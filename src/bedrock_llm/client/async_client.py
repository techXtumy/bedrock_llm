"""Async client implementation."""

from typing import (Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple,
                    Union, cast)

from ..config.base import RetryConfig
from ..config.model import ModelConfig
from ..schema.message import MessageBlock
from ..schema.tools import ToolMetadata
from ..types.enums import ModelName, StopReason
from .base import BaseClient


class AsyncClient(BaseClient):
    """Async client for Bedrock LLM implementations."""

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        memory: Optional[List[MessageBlock]] = None,
        retry_config: Optional[RetryConfig] = None,
        max_iterations: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize async client."""
        super().__init__(
            region_name,
            model_name,
            memory,
            retry_config,
            max_iterations,
            **kwargs
        )

    async def generate_async(
        self,
        prompt: Union[str, MessageBlock, Sequence[MessageBlock]],
        system: Optional[str] = None,
        tools: Optional[Union[List[Dict[str, Any]], List[ToolMetadata]]] = None,
        config: Optional[ModelConfig] = None,
        auto_update_memory: bool = True,
        **kwargs: Any,
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        """Generate a response from the model asynchronously with streaming."""
        config_internal = config or ModelConfig()
        invoke_messages = self._process_prompt(prompt, auto_update_memory)

        async def _generate_stream():
            request_body = await self.model_implementation.prepare_request_async(
                config=config_internal,
                prompt=cast(
                    Union[str, List[Dict[Any, Any]]],
                    invoke_messages,
                ),
                system=system,
                tools=tools,
                **kwargs,
            )

            response = await self._invoke_model_async(request_body)

            async for (
                token,
                stop_reason,
                response_msg,
            ) in self.model_implementation.parse_stream_response(response["body"]):
                if (
                    self.memory is not None
                    and auto_update_memory
                    and response_msg is not None
                ):
                    self.memory.append(response_msg.model_dump())
                yield token, stop_reason, response_msg

        try:
            async for result in _generate_stream():
                yield result
        except Exception:
            async for result in self._handle_retry_logic_stream(_generate_stream):
                yield result
