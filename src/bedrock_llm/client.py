"""LLM client implementation."""
import asyncio
import json
import logging
from functools import lru_cache
from typing import (Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple,
                    Union, cast)

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError

from .config.base import RetryConfig
from .config.model import ModelConfig
from .models.ai21 import JambaImplementation
from .models.amazon import TitanImplementation
from .models.anthropic import ClaudeImplementation
from .models.base import BaseModelImplementation
from .models.meta import LlamaImplementation
from .models.mistral import (MistralChatImplementation,
                             MistralInstructImplementation)
from .schema.message import MessageBlock
from .schema.tools import ToolMetadata
from .types.enums import ModelName, StopReason

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    _model_implementations: Dict[ModelName, BaseModelImplementation] = {}
    _bedrock_clients: Dict[str, Any] = {}

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        memory: Optional[List[MessageBlock]] = None,
        retry_config: Optional[RetryConfig] = None,
        max_iterations: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.region_name = region_name
        self.model_name = model_name
        self.retry_config = retry_config or RetryConfig()
        self.bedrock_client = self._get_or_create_bedrock_client(region_name, **kwargs)
        self.model_implementation = self._get_or_create_model_implementation(model_name)
        self.memory = memory
        self.max_iterations = max_iterations

    @classmethod
    def _get_or_create_bedrock_client(cls, region_name: str, **kwargs) -> Any:
        """Get or create a cached Bedrock client for the region."""
        cache_key = f"{region_name}_{hash(frozenset(kwargs.items()))}"
        if cache_key not in cls._bedrock_clients:
            config = Config(
                retries={"max_attempts": 3, "mode": "standard"},
                max_pool_connections=50,
                tcp_keepalive=True,
            )
            profile_name = kwargs.pop("profile_name", None)
            session = (
                boto3.Session(profile_name=profile_name)
                if profile_name
                else boto3.Session()
            )
            cls._bedrock_clients[cache_key] = session.client(
                "bedrock-runtime", region_name=region_name, config=config, **kwargs
            )
        return cls._bedrock_clients[cache_key]

    @classmethod
    @lru_cache(maxsize=32)
    def _get_or_create_model_implementation(
        cls, model_name: ModelName
    ) -> BaseModelImplementation:
        """Get or create a cached model implementation."""
        if model_name not in cls._model_implementations:
            implementations = {
                ModelName.CLAUDE_3_HAIKU: ClaudeImplementation(),
                ModelName.CLAUDE_3_5_HAIKU: ClaudeImplementation(),
                ModelName.CLAUDE_3_5_SONNET: ClaudeImplementation(),
                ModelName.CLAUDE_3_5_OPUS: ClaudeImplementation(),
                ModelName.LLAMA_3_2_1B: LlamaImplementation(),
                ModelName.LLAMA_3_2_3B: LlamaImplementation(),
                ModelName.LLAMA_3_2_11B: LlamaImplementation(),
                ModelName.LLAMA_3_2_90B: LlamaImplementation(),
                ModelName.TITAN_LITE: TitanImplementation(),
                ModelName.TITAN_EXPRESS: TitanImplementation(),
                ModelName.TITAN_PREMIER: TitanImplementation(),
                ModelName.JAMBA_1_5_LARGE: JambaImplementation(),
                ModelName.JAMBA_1_5_MINI: JambaImplementation(),
                ModelName.MISTRAL_7B: MistralInstructImplementation(),
                ModelName.MISTRAL_LARGE_2: MistralChatImplementation(),
            }
            cls._model_implementations[model_name] = implementations[model_name]
        return cls._model_implementations[model_name]

    def _process_prompt(
        self,
        prompt: Union[str, MessageBlock, List[MessageBlock]],
        auto_update_memory: bool,
    ) -> Union[str, MessageBlock, List[Dict[Any, Any]]]:
        """Process and validate the prompt.

        Args:
            prompt: Input prompt as string, MessageBlock, or list of MessageBlocks
            auto_update_memory: Whether to update memory automatically

        Returns:
            Processed prompt in appropriate format for model

        Raises:
            ValueError: If memory is set and prompt is a string
        """
        if self.memory is not None and auto_update_memory:
            if isinstance(prompt, str):
                raise ValueError(
                    "Prompt must be MessageBlock or list when memory is enabled"
                )
            if isinstance(prompt, MessageBlock):
                self.memory.append(prompt.model_dump())
                return self.memory
            if isinstance(prompt, list):
                self.memory.extend(
                    [
                        x.model_dump() if isinstance(x, MessageBlock) else x
                        for x in prompt
                    ]
                )
                return self.memory
        return prompt

    async def _handle_retry_logic_stream(self, operation, *args, **kwargs):
        """Handle retry logic for streaming async operations."""
        for attempt in range(self.retry_config.max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    async for result in await operation(*args, **kwargs):
                        yield result
                else:
                    async for result in operation(*args, **kwargs):
                        yield result
                return
            except (ReadTimeoutError, ClientError) as e:
                if attempt < self.retry_config.max_retries - 1:
                    delay = self.retry_config.retry_delay * (
                        2**attempt if self.retry_config.exponential_backoff else 1
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    raise
        raise Exception("Max retries reached. Unable to invoke model.")

    async def _handle_retry_logic_sync(self, operation, *args, **kwargs):
        """Handle retry logic for non-streaming async operations."""
        for attempt in range(self.retry_config.max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                return result
            except (ReadTimeoutError, ClientError) as e:
                if attempt < self.retry_config.max_retries - 1:
                    delay = self.retry_config.retry_delay * (
                        2**attempt if self.retry_config.exponential_backoff else 1
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed. Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    raise
        raise Exception("Max retries reached. Unable to invoke model.")

    async def _invoke_model(self, request_body: Dict[str, Any]) -> Any:
        """Async wrapper for model invocation."""
        return await asyncio.to_thread(
            self.bedrock_client.invoke_model,
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(request_body),
        )

    async def _invoke_model_stream(self, request_body: Dict[str, Any]) -> Any:
        """Async wrapper for model stream invocation."""
        return await asyncio.to_thread(
            self.bedrock_client.invoke_model_with_response_stream,
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(request_body),
        )

    def generate(
            self,
            prompt: Union[str, MessageBlock, List[MessageBlock]],
            system: Optional[str] = None,
            tools: Optional[Union[List[Dict[str, Any]], List[ToolMetadata]]] = None,
            config: Optional[ModelConfig] = None,
            auto_update_memory: bool = True,
            **kwargs: Any,
    ) -> Tuple[MessageBlock, StopReason]:
        """Generate a response from the model synchronously."""
        async def _run_generate():
            return await self._handle_retry_logic_sync(
                self._generate,
                prompt=prompt,
                system=system,
                tools=tools,
                config=config,
                auto_update_memory=auto_update_memory,
                **kwargs
            )
        return asyncio.run(_run_generate())

    async def _generate(
        self,
        prompt: Union[str, MessageBlock, List[MessageBlock]],
        system: Optional[str] = None,
        tools: Optional[Union[List[Dict[str, Any]], List[ToolMetadata]]] = None,
        config: Optional[ModelConfig] = None,
        auto_update_memory: bool = True,
        **kwargs: Any,
    ):
        """Internal method to generate response."""
        config_internal = config or ModelConfig()
        invoke_message = self._process_prompt(prompt, auto_update_memory)

        request_body = self.model_implementation.prepare_request(
            config=config_internal,
            prompt=cast(
                Union[str, MessageBlock, List[Dict[Any, Any]]],
                invoke_message,
            ),
            system=system,
            tools=tools,
            **kwargs,
        )

        response = await self._invoke_model(request_body)
        response_msg, stop_reason = self.model_implementation.parse_response(
            response["body"]
        )

        if (
            self.memory is not None
            and auto_update_memory
            and response_msg is not None
        ):
            self.memory.append(response_msg.model_dump())

        return response_msg, stop_reason

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
        invoke_message = self._process_prompt(prompt, auto_update_memory)

        async def _generate_stream():
            request_body = await self.model_implementation.prepare_request_async(
                config=config_internal,
                prompt=cast(
                    Union[str, MessageBlock, List[Dict[Any, Any]]],
                    invoke_message,
                ),
                system=system,
                tools=tools,
                **kwargs,
            )

            response = await self._invoke_model_stream(request_body)

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

        # Use async for directly with the generator function
        try:
            async for result in _generate_stream():
                yield result
        except Exception:
            # Wrap the generator in retry logic only if there's an error
            async for result in self._handle_retry_logic_stream(_generate_stream):
                yield result
