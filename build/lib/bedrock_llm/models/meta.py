"""Meta model implementation."""

import json
import logging
import os
import uuid
from typing import (Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple,
                    Union)

from jinja2 import Environment, FileSystemLoader

from ..models.base import (BaseModelImplementation, MessageBlock, ModelConfig,
                           StopReason, SystemBlock)
from ..schema.tools import ToolMetadata


class LlamaImplementation(BaseModelImplementation):
    # Determine the absolute path to the templates directory
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")

    def load_template(
        self,
        prompt: Sequence[Dict[str, Any]],
        system: Optional[str],
        tools: Optional[Sequence[ToolMetadata]] = None,
    ) -> str:
        env = Environment(
            loader=FileSystemLoader(self.TEMPLATE_DIR),
        )
        template = env.get_template("llama32_template.j2")
        rendered = template.render(
            {"SYSTEM": system, "REQUEST": prompt, "TOOLS": tools}
        )
        return rendered

    def prepare_request(
        self,
        config: ModelConfig,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Sequence[ToolMetadata]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(system, SystemBlock):
            system = system.text

        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, tools)

        return {
            "prompt": prompt,
            "max_gen_len": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

    async def prepare_request_async(
        self,
        config: ModelConfig,
        prompt: Union[str, Sequence[Dict[str, Any]]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Sequence[ToolMetadata]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(system, SystemBlock):
            system = system.text

        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, tools)

        return {
            "prompt": prompt,
            "max_gen_len": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

    def parse_response(self, response: Any) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response)
        response_text = chunk["generation"].strip()

        if response_text[0] == "[" and response_text[-1] == "]":
            message = MessageBlock(role="tool", content=response_text)
            return message, StopReason.TOOL_USE

        message = MessageBlock(role="assistant", content=response_text)
        if chunk["stop_reason"] == "stop":
            return message, StopReason.END_TURN
        elif chunk["stop_reason"] == "length":
            return message, StopReason.MAX_TOKENS
        return message, StopReason.ERROR

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse Llama's string format tool calls into structured format.

        Args:
            response (str): String containing tool calls like
                "[get_weather(location='New York')]"

        Returns:
            List[Dict[str, Any]]: List of parsed tool calls in standard format
        """
        content = response[1:-1].strip()
        if not content:
            return []

        # Split by commas not inside parentheses
        depth = 0
        current = []
        calls = []

        for char in content:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                calls.append(''.join(current).strip())
                current = []
                continue
            current.append(char)

        if current:
            calls.append(''.join(current).strip())

        tool_calls = []
        for call in calls:
            # Extract function name and arguments
            func_name = call[:call.index('(')]
            args_str = call[call.index('(')+1:call.rindex(')')]

            # Parse arguments
            args = {}
            if args_str:
                for arg in args_str.split(','):
                    key, value = arg.split('=')
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    args[key] = value

            tool_calls.append({
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": func_name.strip(),
                    "arguments": json.dumps(args)
                }
            })

        return tool_calls

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        full_answer: List[str] = []

        async for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["generation"], None, None
            full_answer.append(chunk["generation"])

            if chunk.get("stop_reason"):
                response = "".join(full_answer).strip()

                # Handle empty response
                if not response:
                    message = MessageBlock(role="assistant", content="")
                    yield None, StopReason.ERROR, message
                    return

                # Check if response is a tool call
                if response.startswith("[") and response.endswith("]"):
                    try:
                        tool_calls = self._parse_tool_calls(response)
                        if tool_calls:
                            message = MessageBlock(
                                role="assistant",
                                content="<|python_tag|>"+response,
                                tool_calls=tool_calls
                            )
                            yield None, StopReason.TOOL_USE, message
                        else:
                            message = MessageBlock(role="assistant", content=response)
                            yield None, StopReason.ERROR, message
                    except Exception as e:
                        logging.error(f"Failed to parse tool calls: {e}")
                        message = MessageBlock(role="assistant", content=response)
                        yield None, StopReason.ERROR, message
                        return

                message = MessageBlock(role="assistant", content=response)
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
