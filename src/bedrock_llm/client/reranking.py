"""Reranking client implementation."""

from typing import Optional

from bedrock_llm.client.base import BaseClient
from bedrock_llm.schema import RetryConfig
from bedrock_llm.types import ModelName


class RerankingClient(BaseClient):
    """Client for Bedrock reranking models."""

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        retry_config: Optional[RetryConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize reranking client."""
        super().__init__(
            region_name,
            model_name,
            retry_config=retry_config,
            **kwargs
        )
        self._sync_client = self._get_or_create_sync_bedrock_client(
            region_name,
            **kwargs
        )
        self._async_client = None

    # TODO: Implement reranking-specific methods once AWS adds reranking models
    # to Bedrock. For now this is a placeholder for future implementation.
