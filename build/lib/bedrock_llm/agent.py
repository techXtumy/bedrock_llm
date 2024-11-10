import asyncio
from functools import wraps

from .client import LLMClient
from .types.enums import ModelName, StopReason
from .config.base import RetryConfig
from .config.model import ModelConfig
from .schema.message import MessageBlock, ToolUseBlock, ToolResultBlock
from .schema.tools import ToolMetadata

from typing import Dict, Any, AsyncGenerator, Tuple, Optional, List, Union


class Agent(LLMClient):
    tool_functions = {}
    
    @classmethod
    def tool(cls, metadata: ToolMetadata):
        """
        A decorator to register a function as a tool for the Agent.

        This decorator registers the function as a tool that can be used by the Agent during its
        execution. It handles both synchronous and asynchronous functions.

        Args:
            metadata (ToolMetadata): Metadata describing the tool, including its name and usage.

        Returns:
            Callable: The decorator function.

        Example:
        >>> @Agent.tool(ToolMetadata(name="example_tool", description="An example tool"))
        ... def example_tool(arg1: str, arg2: int) -> str:
        ...     return f"Processed {arg1} with {arg2}"
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            is_async = asyncio.iscoroutinefunction(func)
            wrapper = async_wrapper if is_async else sync_wrapper
            
            cls.tool_functions[metadata.name] = {
                "function": wrapper,
                "metadata": metadata.model_dump(),
                "is_async": is_async
            }
            return func
        return decorator
    
    def __init__(        
        self,
        region_name: str,
        model_name: ModelName,
        max_iterations: Optional[int] = 5,
        retry_config: Optional[RetryConfig] = None
    ):
        super().__init__(region_name, model_name, [], retry_config)
        self.max_iterations = max_iterations
    
    
    async def __process_tools(
        self,
        tools_list: List[ToolUseBlock]
    ) -> MessageBlock:
        """
        Process a list of tool use requests and return the results.

        This method iterates through the list of tool use requests, executes each tool,
        and compiles the results into a MessageBlock.

        Args:
            tools_list (List[ToolUseBlock]): A list of tool use requests.

        Returns:
            MessageBlock: A message containing the results of all tool executions.

        Note:
            If a tool is not found or an error occurs during execution, an error message
            is included in the result.
        """
        message = MessageBlock(role="user", content=[])

        for tool in tools_list:
            if not isinstance(tool, ToolUseBlock):
                continue
            
            tool_name = tool.name
            tool_data = self.tool_functions.get(tool_name)
            
            if tool_data:
                try:
                    result = await tool_data["function"](**tool.input) if tool_data["is_async"] else tool_data["function"](**tool.input)
                    is_error = False
                except Exception as e:
                    result = str(e)
                    is_error = True
            else:
                result = f"Tool {tool_name} not found"
                is_error = True

            message.content.append(
                ToolResultBlock(
                    type="tool_result",
                    tool_use_id=tool.id,
                    is_error=is_error,
                    content=str(result)
                )
            )

        return message   
    

    async def generate_and_action_async(
        self,
        prompt: Union[str, MessageBlock, List[MessageBlock]],
        tools: List[str],
        system: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock], Optional[Union[List[ToolResultBlock], List[str], List[Dict]]]], None]:
        """
        Asynchronously generate responses and perform actions based on the given prompt and tools.

        This method generates responses using the language model, processes any tool use requests,
        and yields the results. It continues this process until a stopping condition is met or
        the maximum number of iterations is reached.

        Args:
            prompt (Union[str, MessageBlock, List[MessageBlock]]): The input prompt or message(s).
            tools (List[str]): List of tool names to be used.
            system (Optional[str]): System message to be used in the conversation.
            config (Optional[ModelConfig]): Configuration for the model.
            **kwargs: Additional keyword arguments to be passed to the generate_async method.

        Yields:
            Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock], Optional[Union[List[ToolResultBlock], List[str], List[Dict]]]]:
                - Token: The generated token (if any).
                - StopReason: The reason for stopping (if any).
                - MessageBlock: The response message (if any).
                - Tool results: The results of tool use (if any).

        Raises:
            ValueError: If memory is set and prompt is not in the correct format.
        """
        self._update_memory(prompt)
        
        tool_metadata = [self.tool_functions[name]["metadata"] for name in tools if name in self.tool_functions] if tools else None
        
        for _ in range(self.max_iterations):
            async for token, stop_reason, response in super().generate_async(
                prompt=self.memory, 
                system=system,
                tools=tool_metadata, 
                config=config, 
                auto_update_memory=False,
                **kwargs
            ):
                if response:
                    self.memory.append(response.model_dump())
                
                if not stop_reason:
                    yield token, None, None, None
                elif stop_reason == StopReason.TOOL_USE:
                    yield None, stop_reason, response, None
                    result = await self.__process_tools(response.content)
                    yield None, None, None, result.content
                    self.memory.append(result.model_dump())
                    break
                else:
                    yield None, stop_reason, response, None
                    return


    def _update_memory(self, prompt: Union[str, MessageBlock, List[MessageBlock]]) -> None:
        """Update the memory with the given prompt."""
        if isinstance(prompt, str):
            self.memory.append(MessageBlock(role="user", content=prompt).model_dump())
        elif isinstance(prompt, MessageBlock):
            self.memory.append(prompt.model_dump())
        elif isinstance(prompt, list) and all(isinstance(x, MessageBlock) for x in prompt):
            self.memory.extend([x.model_dump() for x in prompt])
        elif isinstance(prompt, list):
            self.memory.extend(prompt)
        else:
            raise ValueError("Invalid prompt format")


        