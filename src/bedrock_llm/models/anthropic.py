import json

from typing import Any, AsyncGenerator, Optional, Tuple, List, Dict

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock, TextBlock, ToolUseBlock
from src.bedrock_llm.types.enums import StopReason


class ClaudeImplementation(BaseModelImplementation):
    
    async def prepare_request(
        self, 
        prompt: str | List[Dict], 
        config: ModelConfig,
        system: Optional[str | Dict] = None,
        tools: Optional[List[Dict] | Dict] = None,
        tool_choice: Optional[Dict] = None, 
    ) -> Dict[str, Any]:
        
        if isinstance(prompt, str):
            prompt = [
                MessageBlock(
                    role="user", 
                    content=prompt
                ).model_dump()
            ]
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": config.max_tokens,
            "messages": prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "stop_sequences": config.stop_sequences
        }
        
        # Conditionally add system prompt if it is not None
        if system is not None:
            request_body["system"] = system
        
        # Conditionally add tools and tool_choice if they are not None
        if tools is not None:
            if isinstance(tools, dict):
                tools = [tools]
            request_body["tools"] = tools
        
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice

        return request_body

    async def parse_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[Tuple[str | None, StopReason | None, MessageBlock | None], None]:
        full_response = ""
        tool_input = ""
        message = MessageBlock(role="assistant", content=[])
        
        # Handle the synchronous EventStream
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])

            if chunk['type'] == 'content_block_delta':
                if chunk['delta']['type'] == 'text_delta':
                    text_chunk = chunk['delta']['text']
                    yield text_chunk, None, None
                    full_response += text_chunk     
                elif chunk['delta']['type'] == 'input_json_delta':
                    text_chunk = chunk['delta']['partial_json']
                    tool_input += text_chunk
            
            elif chunk['type'] == 'content_block_start':
                id = chunk['content_block'].get('id')
                name = chunk['content_block'].get('name')
            
            elif chunk['type'] == 'content_block_stop':
                if full_response != "":
                    message.content.append(
                        TextBlock(
                            type="text",
                            text=full_response
                        )
                    )
                    full_response = ""
                else:
                    try:
                        input_data = json.loads(tool_input)
                    except json.JSONDecodeError:
                        input_data = {}
                    tool = ToolUseBlock(
                        type="tool_use",
                        id=id, 
                        name=name, 
                        input=input_data
                    )
                    message.content.append(tool)
                    tool_input = ""
            
            elif chunk['type'] == 'message_delta':
                stop_reason = chunk['delta']['stop_reason']
                if stop_reason:
                    if stop_reason == "end_turn":
                        yield None, StopReason.END_TURN, message
                    elif stop_reason == "stop_sequence":
                        yield None, StopReason.STOP_SEQUENCE, message
                    elif stop_reason == "max_tokens":
                        yield None, StopReason.MAX_TOKENS, message
                    elif stop_reason == "tool_use":
                        yield None, StopReason.TOOL_USE, message
                    else:
                        yield None, StopReason.ERROR, message
                    return