"""Base model implementation."""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from ..config.model import ModelConfig
from ..schema.message import MessageBlock, SystemBlock
from ..schema.tools import ToolMetadata
from ..types.enums import StopReason


class BaseModelImplementation(ABC):
    @abstractmethod
    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def prepare_request_async(
        self,
        config: ModelConfig,
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def parse_response(self, response: Any) -> Tuple[MessageBlock, StopReason]:
        pass

    @abstractmethod
    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        pass
