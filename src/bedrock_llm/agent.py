import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import (Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple,
                    Union, cast)

from .client import LLMClient
from .config.base import RetryConfig
from .config.model import ModelConfig
from .schema.message import (MessageBlock, ToolCallBlock, ToolResultBlock,
                             ToolUseBlock)
from .schema.tools import ToolMetadata
from .types.enums import ModelName, StopReason


class Agent(LLMClient):
    tool_functions: Dict[str, Dict[str, Any]] = {}
    _tool_cache: Dict[str, Any] = {}
    _executor = ThreadPoolExecutor(max_workers=10)

    @classmethod
    def tool(cls, metadata: ToolMetadata):
        """
        A decorator to register a function as a tool for the Agent.
        """

        def decorator(func):
            cache_key = metadata.name
            if cache_key in cls._tool_cache:
                return cls._tool_cache[cache_key]

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
                "metadata": metadata.model_dump(),
                "is_async": is_async,
            }
            cls.tool_functions[metadata.name] = tool_info
            cls._tool_cache[cache_key] = wrapper

            return wrapper

        return decorator

    def __init__(
        self,
        region_name: str,
        model_name: ModelName,
        max_iterations: Optional[int] = 5,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> None:
        super(LLMClient, self).__init__(region_name, model_name, [], retry_config, **kwargs)
        self.max_iterations = max_iterations

    async def __execute_tool(
        self, tool_data: Dict[str, Any], params: Dict[str, Any]
    ) -> Tuple[Any, bool]:
        """Execute a single tool with error handling"""
        try:
            result = (
                await tool_data["function"](**params)
                if tool_data["is_async"]
                else await asyncio.get_event_loop().run_in_executor(
                    self._executor, lambda: tool_data["function"](**params)
                )
            )
            return result, False
        except Exception as e:
            return str(e), True

    async def __process_tools(
        self, tools_list: Union[List[ToolUseBlock], List[ToolCallBlock]]
    ) -> Union[MessageBlock, List[MessageBlock]]:
        """Process tool use requests and return results."""
        if isinstance(tools_list[-1], ToolUseBlock):
            message = MessageBlock(role="user", content=[])
            state = 1
        else:
            message: List[MessageBlock] = []
            state = 0

        # Process tools concurrently when possible
        tasks = []
        for tool in tools_list:
            if not isinstance(tool, (ToolUseBlock, ToolCallBlock)):
                continue

            if state:  # Claude, Llama Way
                tool_name = tool.name
                tool_data = self.tool_functions.get(tool_name)
                if tool_data:
                    tasks.append((tool, tool_data, tool.input))
            else:  # Mistral AI, Jamaba Way
                tool_name = tool.function
                tool_params = json.loads(tool_name["arguments"])
                tool_data = self.tool_functions.get(tool_name["name"])
                if tool_data:
                    tasks.append((tool, tool_data, tool_params))

        # Execute tools concurrently
        if tasks:
            results = await asyncio.gather(
                *[self.__execute_tool(t_data, params) for _, t_data, params in tasks]
            )

            # Process results
            for (tool, tool_data, _), (result, is_error) in zip(tasks, results):
                if state:
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

        return message

    async def generate_and_action_async(
        self,
        prompt: Union[str, MessageBlock, Sequence[MessageBlock]],
        tools: List[str],
        system: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[
        Tuple[
            Optional[str],
            Optional[StopReason],
            Optional[MessageBlock],
            Optional[Union[List[ToolResultBlock], List[str], List[Dict[str, Any]]]],
        ],
        None,
    ]:
        """Generate responses and perform actions based on prompt and tools."""
        if not isinstance(self.memory, list):
            raise ValueError("Memory must be a list")

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
                prompt=self.memory,
                system=system,
                tools=tool_metadata,
                config=config,
                auto_update_memory=False,
                **kwargs,
            ):
                if response:
                    self.memory.append(response.model_dump())

                if not stop_reason:
                    yield token, None, None, None
                elif stop_reason == StopReason.TOOL_USE:
                    yield None, stop_reason, response, None
                    if not response:
                        raise Exception(
                            "No tool call request from the model. "
                            "Error from API bedrock when the model is not return a valid "
                            "tool response, but still return StopReason as TOOLUSE request."
                        )

                    tool_content = (
                        response.content
                        if not response.tool_calls
                        else response.tool_calls
                    )
                    result = await self.__process_tools(
                        cast(List[ToolUseBlock], tool_content)
                    )

                    if isinstance(result, list):
                        yield None, None, None, result
                        self.memory.extend(result)
                    else:
                        yield None, None, None, result.content
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
