"""LLM client implementation."""
import asyncio
import json
import logging
import time
from functools import lru_cache
from typing import (Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple,
                    Union, cast)

from botocore.exceptions import ClientError, ReadTimeoutError

from .aws_clients import AWSClientManager
from .config.base import RetryConfig
from .config.model import ModelConfig
from .models import (BaseEmbeddingsImplementation, BaseModelImplementation,
                     ClaudeImplementation, JambaImplementation,
                     LlamaImplementation, MistralChatImplementation,
                     MistralInstructImplementation,
                     TitanEmbeddingsV1Implementation,
                     TitanEmbeddingsV2Implementation, TitanImplementation)
from .schema.message import MessageBlock
from .schema.tools import ToolMetadata
from .types.enums import ModelName, StopReason

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    _model_implementations: Dict[ModelName, BaseModelImplementation] = {}
    _aws_client_manager = None

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
        self.model_implementation = self._get_or_create_model_implementation(
            model_name
        )
        self.memory = memory
        self.max_iterations = max_iterations
        self._sync_client = self._get_or_create_sync_bedrock_client(
            region_name,
            **kwargs
        )
        self._async_client = None

    @classmethod
    def _get_or_create_sync_bedrock_client(cls, region_name: str, **kwargs) -> Any:
        """Get a synchronous Bedrock client for the region."""
        profile_name = kwargs.pop("profile_name", None)
        return AWSClientManager.get_sync_client(region_name, profile_name, **kwargs)

    async def _get_async_client(self):
        if not self._async_client:
            self._async_client = await AWSClientManager.get_async_client(
                self.region_name
            )
        return self._async_client

    @classmethod
    @lru_cache(maxsize=32)
    def _get_or_create_model_implementation(
        cls, model_name: ModelName
    ) -> Union[BaseModelImplementation, BaseEmbeddingsImplementation]:
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
                ModelName.TITAN_EMBED_V1: TitanEmbeddingsV1Implementation(),
                ModelName.TITAN_EMBED_V2: TitanEmbeddingsV2Implementation(),
            }
            cls._model_implementations[model_name] = implementations[model_name]
        return cls._model_implementations[model_name]

    def _process_prompt(
        self,
        prompt: Union[str, MessageBlock, List[MessageBlock]],
        auto_update_memory: bool,
    ) -> Union[str, MessageBlock, List[Dict[Any, Any]]]:
        """Process and validate the prompt. Update memory if set"""
        if self.memory is not None and auto_update_memory:
            if isinstance(prompt, str):
                raise ValueError(
                    """Prompt must be MessageBlock or list when
                    memory is enabled"""
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

    def _handle_retry_logic_sync(self, operation, *args, **kwargs):
        """Handle retry logic for non-streaming sync operations."""
        for attempt in range(self.retry_config.max_retries):
            try:
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
                    time.sleep(delay)
                else:
                    logger.error(f"Max retries reached. Error: {str(e)}")
                    raise
        raise Exception("Max retries reached. Unable to invoke model.")

    def _invoke_model(self, request_body: Dict[str, Any]) -> Any:
        """Sync model invocation using boto3 client."""
        return self._sync_client.invoke_model(
            modelId=self.model_name,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(request_body),
        )

    async def _invoke_model_stream(self, request_body: Dict[str, Any]) -> Any:
        """Async model stream invocation using aiobotocore."""
        async_client = await self._get_async_client()
        return await async_client.invoke_model_with_response_stream(
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

        response = self._handle_retry_logic_sync(
            self._invoke_model,
            request_body
        )
        response_msg, stop_reason = self.model_implementation.parse_response(
            response["body"].read()
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

    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for the given texts."""
        return asyncio.run(self._embed(texts, **kwargs))

    async def _embed(
        self,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Internal method to generate embeddings asynchronously."""
        model_impl = self._get_or_create_model_implementation(
            self.model_config.model_name
        )
        if not isinstance(model_impl, BaseEmbeddingsImplementation):
            raise ValueError(
                f"""Model {self.model_config.model_id}
                does not support embeddings"""
            )

        request_body = model_impl.prepare_embedding_request(
            config=self.model_config,
            texts=texts,
            **kwargs
        )

        response = await self._handle_retry_logic_sync(
            self._invoke_model,
            request_body
        )

        return model_impl.parse_embedding_response(response)

    async def close(self):
        """Close the async client session."""
        if self._async_client:
            await self._async_client.__aexit__(None, None, None)
            self._async_client = None
        await AWSClientManager.close_async_clients()
