import json
import os
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
        prompt: Union[MessageBlock, Sequence[Dict[str, Any]]],
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
        prompt: Union[str, MessageBlock, Sequence[Dict[str, Any]]],
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
        prompt: Union[str, MessageBlock, Sequence[Dict[str, Any]]],
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
        chunk = json.loads(response.read())
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

    async def parse_stream_response(
        self, stream: Any
    ) -> AsyncGenerator[
        Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None
    ]:
        full_answer: List[str] = []

        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["generation"], None, None
            full_answer.append(chunk["generation"])

            if chunk.get("stop_reason"):
                response = "".join(full_answer).strip()

                if response[0] == "[" and response[-1] == "]":
                    message = MessageBlock(
                        role="assistant",
                        content="<|python_tag|>" + response,
                    )
                    yield None, StopReason.TOOL_USE, message
                    return

                message = MessageBlock(role="assistant", content=response)
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
