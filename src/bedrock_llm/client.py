import json
import boto3
import asyncio
import time

from .types.enums import ModelName, StopReason
from .config.base import RetryConfig
from .config.model import ModelConfig
from .schema.message import MessageBlock
from .models.base import BaseModelImplementation
from .models.anthropic import ClaudeImplementation
from .models.meta import LlamaImplementation
from .models.amazon import TitanImplementation
from .models.ai21 import JambaImplementation
from .models.mistral import MistralInstructImplementation, MistralChatImplementation
from .schema.tools import ToolMetadata
from botocore.config import Config
from botocore.exceptions import ClientError, ReadTimeoutError

from typing import Dict, Any, AsyncGenerator, Tuple, Optional, List, Union


class LLMClient:
    
    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        memory: Optional[List[MessageBlock]] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.region_name = region_name
        self.model_name = model_name
        self.retry_config = retry_config or RetryConfig()

        # Initialize Bedrock client
        self.bedrock_client = self._initialize_bedrock_client()
        
        # Initialize model implementation
        self.model_implementation = self._get_model_implementation()
        
        # Initiatlize model memory
        self.memory = memory
        
    
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
            ModelName.MISTRAL_LARGE_2: MistralChatImplementation()
        }
        return implementations[self.model_name]
    
    
    def generate(
        self,
        prompt: Union[str, MessageBlock, List[MessageBlock]],
        system: Optional[str] = None,
        tools: Optional[Union[List[Dict[str, Any]], List[ToolMetadata]]] = None,
        config: Optional[ModelConfig] = None,
        auto_update_memory: bool = True,
        **kwargs: Any
    ) -> Tuple[MessageBlock, StopReason]:
        """
        Synchronously generates a single response from an Amazon Bedrock model.

        Makes a blocking call to the model and handles the complete request-response cycle,
        including automatic retries on failures and conversation memory management.
        
        Args:
            prompt (Union[str, MessageBlock, List[MessageBlock]]): The input for the model:
                - str: Direct text prompt (only if memory disabled)
                - MessageBlock: Single message with role and content
                - List[MessageBlock]: Conversation history as sequence of messages
            system (Optional[str]): System instructions to control model behavior.
                Applied at the beginning of the conversation. Defaults to None.
            documents (Optional[str]): Additional context for the model's response,
                typically used for RAG applications. Defaults to None.
            tools (Optional[List[Dict[str, Any]], List[ToolMetadata]]): Function calling definitions that
                the model can use. Each tool must include name, description, and 
                parameters schema. Defaults to None.
            config (Optional[ModelConfig]): Controls model behavior with parameters like temperature,
                max_tokens, top_p, etc. If None, uses default configuration. Defaults to None
            auto_update_memory (bool): When True and memory is enabled, automatically
                adds prompts and responses to conversation history. Defaults to True.
            **kwargs (Any): Additional model-specific parameters passed directly to
                the underlying implementation.

        Raises:
            ValueError: When memory is enabled but prompt is a string instead of
                MessageBlock(s).
            ReadTimeoutError: When model request times out. Will retry according to
                retry_config settings.
            ClientError: On AWS Bedrock API errors. Will retry according to
                retry_config settings.
            Exception: When all retry attempts are exhausted without success.

        Returns:
            Tuple[MessageBlock, StopReason]: Contains:
                - MessageBlock: Model's complete response with role and content
                - StopReason: Enumerated reason for generation completion (e.g.,
                StopReason.END_TURN, StopReason.MAX_TOKEN, StopReason.STOP_SEQUENCE, ...)

        Examples:
            Basic usage with string prompt (memory disabled):
            >>> config = ModelConfig(temperature=0.7, max_tokens=100)
            >>> response, stop_reason = client.generate(
            ...     config=config,
            ...     prompt="Explain quantum computing",
            ...     auto_update_memory=False
            ... )
            >>> print(response.content)

            Using conversation memory:
            >>> message = MessageBlock(role="user", 
            ...                       content="What are the benefits of Python?")
            >>> response, _ = client.generate(
            ...     config=config,
            ...     prompt=message,
            ...     system="You are a programming expert."
            ... )

            With function calling:
            >>> weather_tool = {
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get current weather",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }
            >>> response, _ = client.generate(
            ...     config=config,
            ...     prompt="What's the weather in Paris?",
            ...     tools=[weather_tool]
            ... )
        """
        config = config or ModelConfig()
        
        # Check if have memory, if set, put prompt into history
        if self.memory is not None and auto_update_memory:
            if isinstance(prompt, str):
                raise ValueError("If memory is set, prompt must be a MessageBlock or list of MessageBlock")
            elif isinstance(prompt, MessageBlock):
                self.memory.append(prompt.model_dump())
            elif isinstance(prompt[0], MessageBlock):
                self.memory.extend([x.model_dump() for x in prompt])
            elif isinstance(prompt, list):
                self.memory.extend(prompt)
            invoke_message = self.memory
        else:
            invoke_message = prompt
        
        for attempt in range(self.retry_config.max_retries):
            try:
                # Prepare the request using the model implementation
                request_body = self.model_implementation.prepare_request(
                    config=config,
                    prompt=invoke_message,
                    system=system,
                    tools=tools,
                    **kwargs
                )
                
                # Invoke the model
                response = self.bedrock_client.invoke_model(
                    modelId=self.model_name,
                    accept="application/json",
                    contentType="application/json",
                    body=json.dumps(request_body)
                )
                
                # Parse the response
                response, stop_reason = self.model_implementation.parse_response(response['body'])
            
                if self.memory is not None and auto_update_memory and response:
                    self.memory.append(response.model_dump())
                    
                return response, stop_reason
                
            except (ReadTimeoutError, ClientError) as e:
                if attempt < self.retry_config.max_retries - 1:
                    delay = self.retry_config.retry_delay * (2 ** attempt if self.retry_config.exponential_backoff else 1)
                    print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries reached. Error: {str(e)}")
                    raise

        raise Exception("Max retries reached. Unable to invoke model.")
    
    
    async def generate_async(
        self,
        prompt: Union[str, MessageBlock, List[MessageBlock]],
        system: Optional[str] = None,
        tools: Optional[Union[List[Dict[str, Any]], List[ToolMetadata]]] = None,
        config: Optional[ModelConfig] = None,
        auto_update_memory: bool = True,
        **kwargs: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None]:
        """
        Asynchronously generates responses from the model with streaming capability.

        This function handles the complete flow of preparing the request, invoking the model,
        and streaming the response. It includes automatic retry logic for handling transient failures
        and memory management for conversation history.

        Args:
            prompt (Union[str, MessageBlock, List[MessageBlock]]): The input prompt for the model.
                Can be a string, single MessageBlock, or list of MessageBlocks.
            system (Optional[str]): System message to guide model behavior. Defaults to None.
            documents (Optional[str]): Reference documents for context. Defaults to None.
            tools (Optional[List[Dict[str, Any]], List[ToolMetadata]]): List of tools available to the model. Defaults to None.
            config (Optional[ModelConfig]): Controls model behavior with parameters like temperature,
                max_tokens, top_p, etc. If None, uses default configuration. Defaults to None
            auto_update_memory (bool): Whether to automatically update conversation memory
                with prompts and responses. Defaults to True.
            **kwargs (Any): Additional keyword arguments passed to the model implementation.

        Yields:
            Tuple[str | None, StopReason | None, MessageBlock | None]: A tuple containing:
                - Generated text token (or None)
                - Stop reason indicating why generation stopped (or None)
                - Complete message block for the response (or None)

        Raises:
            ValueError: If memory is enabled and prompt is provided as a string instead of MessageBlock(s).
            ReadTimeoutError: If the model request times out after all retry attempts.
            ClientError: If there's an error communicating with the model service after all retry attempts.
            Exception: If maximum retries are reached without successful model invocation.

        Notes:
            - The function implements exponential backoff retry logic for handling transient failures.
            - When memory is enabled, prompts and responses are automatically added to conversation history.
            - The response is streamed token by token for real-time processing.

        Examples:
            Basic usage with a string prompt (memory disabled):
            >>> config = ModelConfig(temperature=0.7, max_tokens=100)
            >>> async for token, stop_reason, response in client.generate_async(
            ...     config=config,
            ...     prompt="Tell me a joke",
            ...     auto_update_memory=False
            ... ):
            ...     if token:
            ...         print(token, end="")

            Using MessageBlock with memory enabled:
            >>> message = MessageBlock(role="user", content="What is Python?")
            >>> async for token, stop_reason, response in client.generate_async(
            ...     config=config,
            ...     prompt=message,
            ...     system="You are a helpful programming assistant."
            ... ):
            ...     if token:
            ...         print(token, end="")

            Using conversation history with multiple messages:
            >>> messages = [
            ...     MessageBlock(role="user", content="What is a database?"),
            ...     MessageBlock(role="assistant", content="A database is..."),
            ...     MessageBlock(role="user", content="What about SQL?")
            ... ]
            >>> async for token, stop_reason, response in client.generate_async(
            ...     config=config,
            ...     prompt=messages
            ... ):
            ...     if token:
            ...         print(token, end="")

            Using tools:
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get current weather",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> async for token, stop_reason, response in client.generate_async(
            ...     config=config,
            ...     prompt="What's the weather in Seattle?",
            ...     tools=tools
            ... ):
            ...     if token:
            ...         print(token, end="")
        """
        
        config = config or ModelConfig()
        
        # Check if have memory, if set, put prompt into history
        if self.memory is not None and auto_update_memory:
            if isinstance(prompt, str):
                raise ValueError("If memory is set, prompt must be a MessageBlock or list of MessageBlock")
            elif isinstance(prompt, MessageBlock):
                self.memory.append(prompt.model_dump())
            elif isinstance(prompt, list):
                self.memory.extend([x.model_dump() for x in prompt])
            elif isinstance(prompt, list):
                self.memory.extend(prompt)
            invoke_message = self.memory
        else:
            invoke_message = prompt
        
        for attempt in range(self.retry_config.max_retries):
            try:
                # Prepare the request using the model implementation
                request_body = await self.model_implementation.prepare_request_async(
                    config=config, 
                    prompt=invoke_message,
                    system=system,
                    tools=tools,
                    **kwargs
                )
                
                # Invoke the model
                response = await asyncio.to_thread(
                    self.bedrock_client.invoke_model_with_response_stream,
                    modelId=self.model_name,
                    accept="application/json",
                    contentType="application/json",
                    body=json.dumps(request_body),
                    # trace="ENABLED"
                )
                
                # Parse and yield the response
                async for token, stop_reason, response in self.model_implementation.parse_stream_response(response['body']):
                    yield token, stop_reason, response
                    
                if self.memory is not None and auto_update_memory and response:
                    self.memory.append(response.model_dump())
                    
                break  # Success, exit retry loop
                
            except (ReadTimeoutError, ClientError) as e:
                if attempt < self.retry_config.max_retries - 1:
                    delay = self.retry_config.retry_delay * (2 ** attempt if self.retry_config.exponential_backoff else 1)
                    print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Max retries reached. Error: {str(e)}")
                    raise

        if attempt >= self.retry_config.max_retries - 1:
            raise Exception("Max retries reached. Unable to invoke model.")
