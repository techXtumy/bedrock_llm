"""Bedrock embeddings model implementations."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from ..config.model import ModelConfig


class EmbeddingInputType(TypedDict):
    input_type: Literal["search_document",
                        "search_query",
                        "classification",
                        "clustering",
    ]


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


class TitanEmbeddingsImplementation(BaseEmbeddingsImplementation):
    """Implementation for Amazon Titan embeddings model."""

    def prepare_embedding_request(
        self,
        config: ModelConfig,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]

        return {
            "inputText": texts[0] if len(texts) == 1 else texts,
        }

    async def prepare_embedding_request_async(
        self,
        config: ModelConfig,
        texts: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        if isinstance(texts, str):
            texts = [texts]

        return {
            "inputText": texts[0] if len(texts) == 1 else texts,
        }

    def parse_embedding_response(
        self,
        response: Any
    ) -> Union[List[float], List[List[float]]]:
        body = response.get("body").read()
        response_json = json.loads(body)

        if "embedding" in response_json:
            return response_json["embedding"]
        elif "embeddings" in response_json:
            return response_json["embeddings"]
        else:
            raise ValueError("No embeddings found in response")

    async def parse_embedding_response_async(
        self,
        response: Any
    ) -> Union[List[float], List[List[float]]]:
        body = response.get("body").read()
        response_json = json.loads(body)

        if "embedding" in response_json:
            return response_json["embedding"]
        elif "embeddings" in response_json:
            return response_json["embeddings"]
        else:
            raise ValueError("No embeddings found in response")
