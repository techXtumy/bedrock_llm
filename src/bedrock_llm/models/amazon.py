import json

from typing import Any, AsyncGenerator, Tuple, List, Dict, Literal

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.types.enums import StopReason


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
    ) -> AsyncGenerator[Tuple[str | None, StopReason | None, MessageBlock | None], None]:
        full_response = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["outputText"], None, None
            full_response.append(chunk["outputText"])
            if chunk["completionReason"]:
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_response)
                )
                if chunk["completionReason"] == "FINISH":
                    yield None, StopReason.END_TURN, message
                elif chunk["completionReason"] == "LENGTH":
                    yield None, StopReason.MAX_TOKENS, message
                elif chunk["completionReason"] == "STOP":
                    yield None, StopReason.STOP_SEQUENCE, message
                else:
                    yield None, StopReason.ERROR, message
                return