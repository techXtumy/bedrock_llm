import json

from typing import Any, AsyncGenerator, Optional, Tuple, List, Dict, Union

from ..models.base import BaseModelImplementation, ModelConfig
from ..schema.message import MessageBlock, TextBlock, ToolUseBlock, SystemBlock
from ..schema.tools import ToolMetadata
from ..types.enums import StopReason


class ClaudeImplementation(BaseModelImplementation):
    
    def prepare_request(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        tool_choice: Optional[Dict] = None, 
    ) -> Dict[str, Any]:
        
        if isinstance(prompt, str):
            prompt = [MessageBlock(role="user", content=prompt).model_dump()]
        elif isinstance(prompt, MessageBlock):
            prompt = [prompt.model_dump()]
        elif isinstance(prompt, list):
            prompt = [msg.model_dump() if isinstance(msg, MessageBlock) else msg for msg in prompt]
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": config.max_tokens,
            "messages": prompt,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "stop_sequences": config.stop_sequences
        }
        
        if system is not None:
            request_body["system"] = system.text.strip() if isinstance(system, SystemBlock) else system.strip()
        
        if tools is not None:
            if isinstance(tools, dict):
                tools = [tools]
            request_body["tools"] = tools
        
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice

        return request_body

    
    async def prepare_request_async(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        tool_choice: Optional[Dict] = None, 
    ) -> Dict[str, Any]:
               
        if isinstance(prompt, str):
            prompt = [
                MessageBlock(
                    role="user", 
                    content=prompt
                ).model_dump()
            ]
        
        if isinstance(prompt, MessageBlock):
            prompt = [prompt.model_dump()]
        
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
    
    def parse_response(
        self, 
        response: Any
    ) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response.read())
        message = MessageBlock(
            role=chunk["role"],
            content=chunk["content"]
        )
        stop_reason = chunk["stop_reason"]
        if stop_reason == "end_turn":
            stop_reason = StopReason.END_TURN
        elif stop_reason == "stop_sequence":
            stop_reason = StopReason.STOP_SEQUENCE
        elif stop_reason == "max_token":
            stop_reason = StopReason.MAX_TOKENS
        elif stop_reason == "tool_use":
            stop_reason = StopReason.TOOL_USE
        else:
            stop_reason = StopReason.ERROR
        return message, stop_reason

    async def parse_stream_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None]:
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