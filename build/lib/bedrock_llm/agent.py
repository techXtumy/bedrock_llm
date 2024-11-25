"""Agent implementation."""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from functools import lru_cache, wraps
from typing import (Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple,
                    TypedDict, Union, cast)

from pydantic import ValidationError

from .client import AsyncClient
from .config.base import RetryConfig
from .config.model import ModelConfig
from .schema.message import (MessageBlock, ToolCallBlock, ToolResultBlock,
                             ToolUseBlock)
from .schema.tools import InputSchema, ToolMetadata
from .types.enums import ModelName, StopReason


class ToolState(Enum):
    """Enum to represent different LLM tool-calling conventions."""
    CLAUDE = 1  # Claude way of tool calling
    MISTRAL_JAMBA_LLAMA = 2  # Mistral AI, Jamba, Llama way of tool calling


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""
    def __init__(
        self,
        tool_name: str,
        message: str,
        original_error: Optional[Exception] = None
    ) -> None:
        self.tool_name = tool_name
        self.message = message
        self.original_error = original_error
        super().__init__(f"Error in tool '{tool_name}': {message}")


class AgentResponse(TypedDict):
    token: Optional[str]
    stop_reason: Optional[StopReason]
    message: Optional[MessageBlock]
    tool_results: Optional[
        Union[
            List[ToolResultBlock],
            List[str],
            List[Dict[str, Any]]
        ]
    ]


class Agent(AsyncClient):
    """
    Agent class that extends AsyncClient to provide tool execution capabilities.

    The Agent class manages tool registration, execution, and memory management for
    conversations with Large Language Models (LLMs). It supports different LLM
    tool-calling conventions and provides robust error handling.

    Attributes:
        tool_functions (Dict[str, Dict[str, Any]]): Registry of available tools
        _tool_cache (Dict[str, Any]): Cache for tool function instances
        _executor (ThreadPoolExecutor): Executor for running sync functions
        _memory_limit (int): Maximum number of messages to keep in memory
        _logger (logging.Logger): Logger instance for the Agent class
    """

    tool_functions: Dict[str, Dict[str, Any]] = {}
    _tool_cache: Dict[str, Any] = {}
    _executor = ThreadPoolExecutor(max_workers=10)
    _logger = logging.getLogger(__name__)

    @classmethod
    def tool(cls, metadata: ToolMetadata):
        """
        A decorator to register a function as a tool for the Agent.

        Args:
            metadata (ToolMetadata): Metadata describing the tool's properties
                                   and input schema.

        Returns:
            Callable: Decorated function that can be used as a tool.

        Raises:
            ValueError: If the tool metadata is invalid.
        """
        def decorator(func):
            cache_key = metadata.name
            if cache_key in cls._tool_cache:
                return cls._tool_cache[cache_key]

            # Validate tool metadata
            try:
                metadata_dict = metadata.model_dump()
            except ValidationError as e:
                cls._logger.error(f"Tool metadata validation failed: {str(e)}")
                raise ValueError(f"Invalid tool metadata for {metadata.name}: {str(e)}")

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            is_async = asyncio.iscoroutinefunction(func)
            wrapper = async_wrapper if is_async else sync_wrapper

            tool_info = {
                "function": wrapper,
                "metadata": metadata_dict,
                "is_async": is_async,
                "created_at": datetime.now().isoformat(),
            }
            cls.tool_functions[metadata.name] = tool_info
            cls._tool_cache[cache_key] = wrapper

            # cls._logger.info(f"Registered tool: {metadata.name}")
            return wrapper

        return decorator

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        auto_update_memory: bool = True,
        max_iterations: Optional[int] = 5,
        retry_config: Optional[RetryConfig] = None,
        memory_limit: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Agent.

        Args:
            region_name (str): AWS region name
            model_name (ModelName): Name of the LLM model to use
            max_iterations (Optional[int]): Maximum number of tool execution iterations
            retry_config (Optional[RetryConfig]): Configuration for retry behavior
            memory_limit (Optional[int]): Maximum number of messages to keep in memory
            **kwargs: Additional arguments passed to LLMClient
        """
        super().__init__(region_name, model_name, [], retry_config, **kwargs)
        self.max_iterations = max_iterations
        self._memory_limit = memory_limit or 100
        self._conversation_history: List[MessageBlock] = []
        self.auto_update_memory = auto_update_memory

    def _manage_memory(self) -> None:
        """
        Manage conversation history by keeping only recent messages.

        This method ensures that the conversation history doesn't grow beyond
        the specified memory limit by removing older messages when necessary.
        The most recent messages are always preserved.
        """
        if len(self._conversation_history) > self._memory_limit:
            self._logger.info(
                f"Pruning convo history to {self._memory_limit} messages"
            )
            self._conversation_history = self._conversation_history[
                -self._memory_limit:
            ]

    async def __execute_tool(
        self, tool_data: Dict[str, Any], params: Dict[str, Any]
    ) -> Tuple[Any, bool]:
        """
        Execute a single tool with comprehensive error handling.

        Args:
            tool_data (Dict[str, Any]): Tool metadata and function
            params (Dict[str, Any]): Parameters to pass to the tool

        Returns:
            Tuple[Any, bool]: Tuple of (result, is_error)

        Raises:
            ToolExecutionError: If tool execution fails
        """
        tool_name = tool_data.get("metadata", {}).get("name", "unknown_tool")

        try:
            # Validate input parameters against schema
            if "input_schema" in tool_data["metadata"]:
                try:
                    InputSchema(
                        **tool_data["metadata"]["input_schema"]
                    ).validate(params)
                except ValidationError as e:
                    raise ToolExecutionError(
                        tool_name, f"Invalid parameters: {str(e)}"
                    )

            # Execute the tool
            result = (
                await tool_data["function"](**params)
                if tool_data["is_async"]
                else await asyncio.get_event_loop().run_in_executor(
                    self._executor, lambda: tool_data["function"](**params)
                )
            )

            # Handle different return types
            if isinstance(result, (dict, list)):
                return json.dumps(result), False
            return str(result), False

        except ToolExecutionError as e:
            self._logger.error(f"Tool execution error: {str(e)}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            raise ToolExecutionError(tool_name, error_msg, e)

    async def __process_tools(
        self, tools_list: Union[List[ToolUseBlock], List[ToolCallBlock]]
    ) -> Union[MessageBlock, List[MessageBlock]]:
        """
        Process tool use requests and return results.

        This method handles different LLM tool-calling conventions and executes
        tools concurrently when possible.

        Args:
            tools_list: List of tool use or call blocks

        Returns:
            Union[MessageBlock, List[MessageBlock]]: Tool execution results

        Raises:
            ToolExecutionError: If any tool execution fails
        """
        # Determine the tool calling convention
        tool_state = (
            ToolState.CLAUDE if isinstance(tools_list[-1], ToolUseBlock)
            else ToolState.MISTRAL_JAMBA_LLAMA
        )

        if tool_state == ToolState.CLAUDE:
            message = MessageBlock(role="user", content=[])
        else:
            message: List[MessageBlock] = []

        # Process tools concurrently when possible
        tasks = []
        for tool in tools_list:
            if not isinstance(tool, (ToolUseBlock, ToolCallBlock)):
                continue

            if tool_state == ToolState.CLAUDE:
                tool_name = tool.name
                tool_data = self.tool_functions.get(tool_name)
                if tool_data:
                    tasks.append((tool, tool_data, tool.input))
            else:
                tool_name = tool.function
                tool_params = json.loads(tool_name["arguments"])
                tool_data = self.tool_functions.get(tool_name["name"])
                if tool_data:
                    tasks.append((tool, tool_data, tool_params))

        # Execute tools concurrently
        if tasks:
            try:
                results = await asyncio.gather(
                    *[self.__execute_tool(t_data, params) for _,
                      t_data, params in tasks],
                    return_exceptions=True
                )

                # Process results
                for (tool, tool_data, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        is_error = True
                    else:
                        result, is_error = result

                    if tool_state == ToolState.CLAUDE:
                        if isinstance(message.content, list):
                            message.content.append(
                                ToolResultBlock(
                                    type="tool_result",
                                    tool_use_id=tool.id,
                                    is_error=is_error,
                                    content=str(result),
                                )
                            )
                    else:
                        message.append(
                            MessageBlock(
                                role="tool",
                                name=tool.function["name"],
                                content=str(result),
                                tool_call_id=tool.id,
                            )
                        )

            except Exception as e:
                self._logger.error(
                    f"Error processing tools: {str(e)}", exc_info=True
                )
                raise

        return message

    async def generate_and_action_async(
        self,
        prompt: Union[str, MessageBlock, Sequence[MessageBlock]],
        tools: List[str],
        system: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Tuple[AgentResponse], None]:
        """Generate responses and perform actions based on prompt and tools."""
        if not isinstance(self.memory, list):
            raise ValueError("Memory must be a list")

        if self.auto_update_memory:
            self._update_memory(prompt)
        tool_metadata = None

        if tools:
            tool_metadata = [
                self.tool_functions[name]["metadata"]
                for name in tools
                if name in self.tool_functions
            ]

        if self.max_iterations is None:
            raise ValueError("max_iterations must not be None")

        for _ in range(self.max_iterations):
            async for token, stop_reason, response in super().generate_async(
                prompt=self.memory if self.auto_update_memory else prompt,
                system=system,
                tools=tool_metadata,
                config=config,
                auto_update_memory=False,   # Agent has seprate memory update
                **kwargs,
            ):
                if response and self.auto_update_memory:
                    self.memory.append(response.model_dump())

                if not stop_reason:
                    yield token, None, None, None
                elif stop_reason == StopReason.TOOL_USE:
                    yield None, stop_reason, response, None
                    if not response:
                        raise Exception(
                            "No tool call request from the model. "
                            "Error from API bedrock when "
                            "the model is not return a valid "
                            "tool response, but still return "
                            "StopReason as TOOLUSE request."
                        )

                    tool_content = (
                        response.content
                        if not response.tool_calls
                        else response.tool_calls
                    )
                    result = await self.__process_tools(
                        cast(Union[
                            List[ToolCallBlock],
                            List[ToolUseBlock]],
                            tool_content
                        )
                    )

                    if isinstance(result, list):
                        yield None, None, None, result
                        if self.auto_update_memory:
                            self.memory.extend(result)
                    else:
                        yield None, None, None, result.content
                        if self.auto_update_memory:
                            self.memory.append(result.model_dump())
                    break
                else:
                    yield None, stop_reason, response, None
                    return

    @lru_cache(maxsize=32)  # Cache memory updates for identical prompts
    def _get_memory_update(self, prompt_str: str) -> Dict[str, Any]:
        return MessageBlock(role="user", content=prompt_str).model_dump()

    def _update_memory(
        self, prompt: Union[str, MessageBlock, Sequence[MessageBlock]]
    ) -> None:
        """Update the memory with the given prompt."""
        if not isinstance(self.memory, list):
            raise ValueError("Memory must be a list")

        if isinstance(prompt, str):
            self.memory.append(self._get_memory_update(prompt))
        elif isinstance(prompt, MessageBlock):
            self.memory.append(prompt.model_dump())
        elif isinstance(prompt, (list, Sequence)):
            if all(isinstance(x, MessageBlock) for x in prompt):
                self.memory.extend(msg.model_dump() for msg in prompt)
            else:
                self.memory.extend(prompt)
        else:
            raise ValueError("Invalid prompt format")

        self._manage_memory()
