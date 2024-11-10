import json
import os

from jinja2 import Environment, FileSystemLoader
from typing import Any, AsyncGenerator, Tuple, List, Dict, Union, Optional

from ..models.base import BaseModelImplementation, ModelConfig
from ..schema.message import MessageBlock, SystemBlock
from ..schema.tools import ToolMetadata
from ..types.enums import StopReason


class TitanImplementation(BaseModelImplementation):
    """
    Read more: https://d2eo22ngex1n9g.cloudfront.net/Documentation/User+Guides/Titan/Amazon+Titan+Text+Prompt+Engineering+Guidelines.pdf
    """
    
    # Determine the absolute path to the templates directory
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")
    
    def load_template(
        self, 
        prompt: Union[MessageBlock, List[Dict]], 
        system: Optional[str]
    ) -> str:
        env = Environment(loader=FileSystemLoader(self.TEMPLATE_DIR))
        template = env.get_template("amazon_template.j2")
        return template.render({"SYSTEM": system, "REQUEST": prompt}).strip() + " "
    
    def prepare_request(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if tools:
            raise ValueError("Amazon Titan models do not support function calling and tools. Please use another model.")
              
        if isinstance(system, SystemBlock):
            system = system.text
        
        formatted_prompt = self.load_template(prompt, system) if not isinstance(prompt, str) else prompt
        
        return {
            "inputText": formatted_prompt,
            "textGenerationConfig": {
                "maxTokenCount": config.max_tokens,
                "temperature": config.temperature,
                "topP": config.top_p,
                "stopSequences": config.stop_sequences
            }
        }
    
    async def prepare_request_async(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if tools:
            raise ValueError("Titan models are not support function callings and tools. Please use another models")
        
        if isinstance(system, SystemBlock):
            system = system.text
        
        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, tools)
        
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": config.max_tokens,
                "temperature": config.temperature,
                "topP": config.top_p,
                "stopSequences": config.stop_sequences
            }
        }
        
    def parse_response(
        self, 
        response: Any
    ) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response.read())
        message = MessageBlock(
            role="assistant",
            content=chunk["results"][0]["outputText"]
        )
        if chunk["results"][0]["completionReason"] == "FINISH":
            return message, StopReason.END_TURN
        elif chunk["results"][0]["completionReason"] == "LENGTH":
            return message, StopReason.MAX_TOKENS
        elif chunk["results"][0]["completionReason"] == "STOP":
            return message, StopReason.STOP_SEQUENCE
        else:
            return message, StopReason.ERROR
    
    async def parse_stream_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None]:
        full_response = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            yield chunk["outputText"], None, None
            full_response.append(chunk["outputText"])
            if chunk["completionReason"]:
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_response)
                )
                if chunk["completionReason"] == "FINISH":
                    yield None, StopReason.END_TURN, message
                elif chunk["completionReason"] == "LENGTH":
                    yield None, StopReason.MAX_TOKENS, message
                elif chunk["completionReason"] == "STOP":
                    yield None, StopReason.STOP_SEQUENCE, message
                else:
                    yield None, StopReason.ERROR, message
                return