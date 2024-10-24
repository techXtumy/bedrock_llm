import json

from typing import Any, AsyncGenerator, Optional, Tuple, List, Dict

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig


class LlamaImplementation(BaseModelImplementation):
    
    async def prepare_request(
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
    
    
    async def parse_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[str, None]:
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["generation"], chunk.get("stop_reason")