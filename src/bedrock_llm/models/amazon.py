import json

from typing import Any, AsyncGenerator, Optional, Tuple, List, Dict

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig


class TitanImplementation(BaseModelImplementation):
    
    async def prepare_request(
        self, 
        prompt: str | List[Dict],
        config: ModelConfig,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": config.max_tokens,
                "temperature": config.temperature,
                "topP": config.top_p,
                "stopSequences": config.stop_sequences
            }
        }
    
    
    async def parse_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[str, None]:
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            text = chunk["outputText"]
            yield text