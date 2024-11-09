import json
import os

from jinja2 import Environment, FileSystemLoader
from typing import Any, AsyncGenerator, Optional, Tuple, List, Dict, Union

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.models.base import StopReason, MessageBlock
from src.bedrock_llm.schema.tools import ToolMetadata


class LlamaImplementation(BaseModelImplementation):
    
    # Determine the absolute path to the templates directory
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")
    
    def load_template(
        self, 
        prompt: Union[MessageBlock, List[Dict]], 
        system: Optional[str], 
        document: Optional[str],
        tools: Optional[List[ToolMetadata]] = None
    ) -> str:
        env = Environment(loader=FileSystemLoader(self.TEMPLATE_DIR))
        template = env.get_template("llama32_template.j2")
        prompt = template.render({"SYSTEM": system, "REQUEST": prompt})
        return prompt
    
    def prepare_request(
        self, 
        prompt: Union[str, MessageBlock, List[Dict]],
        config: ModelConfig,
        system: Optional[str] = None, 
        document: Optional[str] = None,
        tools:  Optional[List[ToolMetadata]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, document, tools)
            
        print(prompt)
        
        return {
            "prompt": prompt,
            "max_gen_len": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        } 
    
    async def prepare_request_async(
        self, 
        prompt: Union[str, MessageBlock, List[Dict]],
        config: ModelConfig,
        system: Optional[str] = None, 
        document: Optional[str] = None,
        tools:  Optional[List[ToolMetadata]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, document, tools)
            
        print(prompt)
        
        return {
            "prompt": prompt,
            "max_gen_len": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        } 
        
    def parse_response(
        self, 
        response: Any
    ) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response.read())
        message = MessageBlock(
            role="assistant",
            content=chunk["generation"]
        )
        if chunk["stop_reason"] == "stop":
            return message, StopReason.END_TURN
        elif chunk["stop_reason"] == "length":
            return message, StopReason.MAX_TOKENS
        else:
            return message, StopReason.ERROR
    
    async def parse_stream_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None]:
        full_answer = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["generation"], None, None
            full_answer.append(chunk["generation"])
            if chunk.get("stop_reason"):
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_answer)
                )
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return