import json

from typing import Any, AsyncGenerator, Tuple, List, Dict, Literal

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
    ) -> AsyncGenerator[Tuple[str, Literal["FINISH", "LENGTH", "STOP", "ERROR"] | None], None]:
        """
        Parse the response from the model into a stream of tokens.

        Args:
            stream (Any): The response from the model.

        Yields:
            Tuple[str, Literal["FINISH", "LENGTH", "STOP", "ERROR"] | None]: 
            A tuple containing:
            - str: The token from the model.
            - Literal["FINISH", "LENGTH", "STOP", "ERROR"] | None: The completion reason, if any.
        """
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["outputText"], chunk.get("completionReason")