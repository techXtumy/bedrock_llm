from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, List

from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.config.model import ModelConfig


class BaseModelImplementation(ABC):
    @abstractmethod
    async def prepare_request(
        self, 
        prompt: str | List[Dict],
        config: ModelConfig, 
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def parse_response(
        self, 
        response: Any
    ) -> AsyncGenerator[Tuple[str | MessageBlock, Optional[str]], None]:
        pass
