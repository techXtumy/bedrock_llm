import json
import boto3
import asyncio

from src.bedrock_llm.types.enums import ModelName
from src.bedrock_llm.config.base import RetryConfig
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.models.base import BaseModelImplementation
from src.bedrock_llm.models.anthropic import ClaudeImplementation
from src.bedrock_llm.models.meta import LlamaImplementation
from src.bedrock_llm.models.amazon import TitanImplementation
from botocore.config import Config
from botocore.exceptions import ClientError

from typing import Dict, Any, AsyncGenerator, Tuple, Optional, List


class LLMClient:
    
    def __init__(
        self,
        region_name: str,
        model_name: Optional[ModelName],
        retry_config: Optional[RetryConfig] = None,
    ):
        self.region_name = region_name
        self.model_name = model_name
        self.retry_config = retry_config or RetryConfig()

        # Initialize Bedrock client
        self.bedrock_client = self._initialize_bedrock_client()
        
        # Initialize model implementation
        self.model_implementation = self._get_model_implementation()
        
    
    def _initialize_bedrock_client(self):
        """
        Initialize the Bedrock client based on the region name.
        """
        config = Config(
            retries={
                'max_attempts': self.retry_config.max_retries,
                'mode': 'standard'
            }
        )
        return boto3.client(
            'bedrock-runtime',
            region_name=self.region_name,
            config=config
        )
        
    
    def _get_model_implementation(self) -> BaseModelImplementation:
        """
        Get the appropriate model implementation based on the model name.
        """
        implementations = {
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
        }
        return implementations[self.model_name]
    
    
    async def generate(
        self,
        prompt: str | List[MessageBlock] | Dict[str, Any],
        config: Optional[ModelConfig] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Tuple[str | MessageBlock, Optional[str]], None]:
        """
        Generate a response from the model.
        
        Args:
            prompt: Either a string prompt or message block a dictionary containing the full request structure.
            config: Optional configuration for the request.
            kwargs: Additional optional arguments required by certain models.
            
        Yields:
            Tuple containing either (text_chunk, None) or (message_block, stop_reason).
        """
        config = config or ModelConfig()
        
        for attempt in range(self.retry_config.max_retries):
            try:
                # Prepare the request using the model implementation
                request_body = await self.model_implementation.prepare_request(prompt, config, **kwargs)
                
                # Invoke the model
                response = await asyncio.to_thread(
                    self.bedrock_client.invoke_model_with_response_stream,
                    modelId=self.model_name,
                    accept="application/json",
                    contentType="application/json",
                    body=json.dumps(request_body)
                )
                
                # Parse and yield the response
                async for chunk, stop_reason in self.model_implementation.parse_response(response['body']):
                    yield chunk, stop_reason
                    
                break  # Success, exit retry loop
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'serviceUnavailableException':
                    if attempt < self.retry_config.max_retries - 1:
                        delay = self.retry_config.retry_delay * (2 ** attempt if self.retry_config.exponential_backoff else 1)
                        await asyncio.sleep(delay)
                        continue
                raise

        if attempt >= self.retry_config.max_retries - 1:
            raise Exception("Max retries reached. Unable to invoke model.")