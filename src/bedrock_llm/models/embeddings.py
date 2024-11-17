"""Bedrock embeddings model implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union


class EmbeddingInputType(TypedDict):
    input_type: Literal["search_document",
                        "search_query",
                        "classification",
                        "clustering"]


class EmbeddingVector(TypedDict):
    embedding_vetor: Union[List[Any], List[List[Any]]]


class Metadata(TypedDict):
    metadata: Dict[str, Any]


class BaseEmbeddingsImplementation(ABC):
    """Base class for embeddings model implementations."""

    @abstractmethod
    def prepare_embedding_request(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare the request body for embedding generation.

        Args:
            texts: Single text or list of texts to embed
            input_type: Prepends special tokens to differentiate each
                type from one another.
                Read more: https://docs.aws.amazon.com/bedrock/latest/
                userguide/model-parameters-embed.html
            embedding_type: Specifies the types of embeddings
                you want to have returned.
                Optional and default is None,
                which returns the Embed Floats response type
            **kwargs: Additional arguments

        Returns:
            Request body dictionary
        """
        pass

    @abstractmethod
    async def prepare_embedding_request_async(
        self,
        texts: Union[str, List[str]],
        input_type: EmbeddingInputType,
        embedding_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare the request body for embedding generation.

        Args:
            texts: Single text or list of texts to embed
            input_type: Prepends special tokens to differentiate each
                type from one another.
                Read more: https://docs.aws.amazon.com/bedrock/latest/
                userguide/model-parameters-embed.html
            embedding_type: Specifies the types of embeddings
                you want to have returned.
                Optional and default is None,
                which returns the Embed Floats response type
            **kwargs: Additional arguments

        Returns:
            Request body dictionary
        """
        pass

    @abstractmethod
    def parse_embedding_response(
        self,
        response: Any
    ) -> Tuple[EmbeddingVector, Optional[Metadata]]:
        """Parse the embedding response from the model.

        Args:
            response: Raw response from the model

        Returns:
            List of embeddings vectors
        """
        pass

    @abstractmethod
    async def parse_embedding_response_async(
        self,
        response: Any
    ) -> Tuple[EmbeddingVector, Optional[Metadata]]:
        """Parse the embedding response from the model.

        Args:
            response: Raw response from the model

        Returns:
            List of embeddings vectors
        """
        pass
