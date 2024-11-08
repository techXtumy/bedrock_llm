import json

from typing import Any, AsyncGenerator, Optional, List, Dict, Tuple

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.types.enums import ToolChoiceEnum, StopReason


class MistralChatImplementation(BaseModelImplementation):
    """
    Read more: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-large-2407.html
    """
    
    async def prepare_request(
        self, 
        prompt: str | List[Dict], 
        config: ModelConfig,
        system: Optional[str] = None,
        tools: Optional[List[Dict] | Dict] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if isinstance(prompt, str):
            messages = [
                MessageBlock(
                    role="user",
                    content=prompt
                ).model_dump()
            ]
            
        if system is not None:
            system = MessageBlock(
                role="system",
                content=system
            ).model_dump()
            messages.insert(0, system)
        
        request_body = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }
        
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
        full_response = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            chunk = chunk["choices"][0]
            if chunk["stop_reason"]:    
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_response)
                )
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "tool_calls":
                    yield None, StopReason.TOOL_USE, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
            else:
                yield chunk["message"]["content"], None, None
                full_response.append(chunk["message"]["content"])
            

class MistralInstructImplementation(BaseModelImplementation):
    """
    Read more: https://docs.mistral.ai/guides/prompting_capabilities/
    """
    
    async def prepare_request(
        self, 
        prompt: str | List[Dict], 
        config: ModelConfig,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k
        }
    
    async def parse_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[Tuple[str | None, StopReason | None, MessageBlock | None], None]:
        full_response = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            chunk = chunk["outputs"][0]
            if chunk["stop_reason"]:
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_response)
                )
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "tool_calls":
                    yield None, StopReason.TOOL_USE, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
            else:
                yield chunk["text"], None, None
                full_response.append(chunk["text"])