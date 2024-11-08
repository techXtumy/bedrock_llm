import json

from typing import Any, AsyncGenerator, Optional, Tuple, List, Dict

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.models.base import StopReason, MessageBlock


class LlamaImplementation(BaseModelImplementation):
    
    def prepare_request(
        self, 
        prompt: str | List[Dict], 
        config: ModelConfig,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "prompt": prompt,
            "max_gen_len": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        } 
    
    async def prepare_request_async(
        self, 
        prompt: str | List[Dict], 
        config: ModelConfig,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "prompt": prompt,
            "max_gen_len": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        } 
        
    def parse_response(
        self, 
        response: Any
    ) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response.read())
        message = MessageBlock(
            role="assistant",
            content=chunk["generation"]
        )
        if chunk["stop_reason"] == "stop":
            return message, StopReason.END_TURN
        elif chunk["stop_reason"] == "length":
            return message, StopReason.MAX_TOKENS
        else:
            return message, StopReason.ERROR
    
    async def parse_stream_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[Tuple[str | None, StopReason | None, MessageBlock | None], None]:
        full_answer = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["generation"], None, None
            full_answer.append(chunk["generation"])
            if chunk.get("stop_reason"):
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_answer)
                )
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return