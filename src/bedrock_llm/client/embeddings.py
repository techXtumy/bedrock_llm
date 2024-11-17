"""Embeddings client implementation."""

from typing import List, Optional, Tuple, Union

from ..aws_clients import AWSClientManager
from ..config.base import RetryConfig
from ..models.embeddings import (BaseEmbeddingsImplementation,
                                 EmbeddingInputType, EmbeddingVector, Metadata)
from ..types.enums import ModelName
from .base import BaseClient


class EmbedClient(BaseClient):
    """Client for Bedrock embeddings models."""

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        retry_config: Optional[RetryConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize embeddings client."""
        super().__init__(
            region_name,
            model_name,
            retry_config=retry_config,
            **kwargs
        )
        if not isinstance(
            self.model_implementation,
            BaseEmbeddingsImplementation
        ):
            raise ValueError(
                f"Model {model_name} does not support embeddings"
            )
        self.profile_name = kwargs.pop("profile_name", None)
        self._sync_client = AWSClientManager.get_sync_client(
            self.region_name,
            self.profile_name
        )

    def embed(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = None,
        **kwargs
    ) -> Tuple[EmbeddingVector, Optional[Metadata]]:
        """Generate embeddings for given texts synchronously."""
        request_body = self.model_implementation.prepare_embedding_request(
            texts=texts,
            input_type=input_type,
            embedding_type=embedding_type,
            **kwargs
        )
        response = self._handle_retry_logic_sync(
            self._invoke_model_sync,
            client=self._sync_client,
            request_body=request_body,
        )
        return self.model_implementation.parse_embedding_response(response)

    async def embed_async(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = None,
        **kwargs
    ) -> Tuple[EmbeddingVector, Optional[Metadata]]:
        """Generate embeddings for given texts asynchronously."""
        request_body = await self.model_implementation.prepare_embedding_request_async(
            texts=texts,
            input_type=input_type,
            embedding_type=embedding_type,
            **kwargs
        )
        response = self._invoke_model_sync(self._sync_client, request_body)

        return await self.model_implementation.parse_embedding_response_async(
            response
        )
