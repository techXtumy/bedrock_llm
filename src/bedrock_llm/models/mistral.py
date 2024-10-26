import json

from typing import Any, AsyncGenerator, Optional, List, Dict

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.types.aliases import ToolChoiceEnum


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
    ) -> AsyncGenerator[str, None]:
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            chunk = chunk["choices"][0]
            yield chunk["message"]["content"], chunk["stop_reason"]
            

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
    ) -> AsyncGenerator[str, None]:
        full_response = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            txt_chunk = chunk["outputs"][0]
            yield txt_chunk["text"], None
            full_response.append(txt_chunk["text"])
            if txt_chunk["stop_reason"]:
                yield "".join(full_response), txt_chunk["stop_reason"]
        return