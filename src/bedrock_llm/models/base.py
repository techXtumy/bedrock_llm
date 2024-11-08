from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, List

from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.types.enums import StopReason


class BaseModelImplementation(ABC):
    @abstractmethod
    def prepare_request(
        self, 
        prompt: str | List[Dict],
        config: ModelConfig, 
        **kwargs
    ) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def prepare_request_async(
        self, 
        prompt: str | List[Dict],
        config: ModelConfig, 
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def parse_response(
        self, 
        response: Any
    ) -> Tuple[MessageBlock, StopReason]:
        pass

    @abstractmethod
    async def parse_stream_response(
        self, 
        response: Any
    ) -> AsyncGenerator[Tuple[str | None, StopReason | None, MessageBlock | None], None]:
        pass
