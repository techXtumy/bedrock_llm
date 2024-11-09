import json
import os

from jinja2 import Environment, FileSystemLoader
from typing import Any, AsyncGenerator, Optional, List, Dict, Tuple, Union

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.schema.tools import ToolMetadata
from src.bedrock_llm.types.enums import ToolChoiceEnum, StopReason


class MistralChatImplementation(BaseModelImplementation):
    """
    Read more: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-large-2407.html
    """
    
    def prepare_request(
        self, 
        prompt: Union[str, MessageBlock, List[Dict]], 
        config: ModelConfig,
        system: Optional[str] = None,
        tools: Optional[Union[List[Dict], Dict]] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        messages = []
        if isinstance(prompt, str):
            messages.append(
                MessageBlock(
                    role="user", 
                    content=prompt
                ).model_dump()
            )
        elif isinstance(prompt, MessageBlock):
            messages.append(prompt.model_dump())
        else:
            messages.extend(prompt)
            
        if system is not None:
            system = MessageBlock(
                role="system",
                content=system
            ).model_dump()
            messages.insert(0, system)
        
        request_body = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }
        
        # Conditionally add tools and tool_choice if they are not None
        if tools is not None:
            if isinstance(tools, dict):
                tools = [tools]
            request_body["tools"] = tools
        
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        
        return request_body
    
    async def prepare_request_async(
        self, 
        prompt: Union[str, MessageBlock, List[Dict]], 
        config: ModelConfig,
        system: Optional[str] = None,
        tools: Optional[Union[List[Dict], Dict]] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        messages = []
        if isinstance(prompt, str):
            messages.append(
                MessageBlock(
                    role="user", 
                    content=prompt
                ).model_dump()
            )
        elif isinstance(prompt, MessageBlock):
            messages.append(prompt.model_dump())
        else:
            messages.extend(prompt)
            
        if system is not None:
            system = MessageBlock(
                role="system",
                content=system
            ).model_dump()
            messages.insert(0, system)
        
        request_body = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }
        
        # Conditionally add tools and tool_choice if they are not None
        if tools is not None:
            if isinstance(tools, dict):
                tools = [tools]
            request_body["tools"] = tools
        
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        
        return request_body
    
    def parse_response(
        self, 
        response: Any
    ) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response.read())
        chunk = chunk["choices"][0]
        message = MessageBlock(
            role=chunk["message"]["role"],
            content=chunk["message"]["content"],
            tool_calls=chunk["message"]["tool_calls"] if "tool_calls" in chunk["message"] else None,
            tool_call_id=chunk["message"]["tool_call_id"] if "tool_call_id" in chunk["message"] else None
        )
        if chunk["finish_reason"] == "stop":
            return message, StopReason.END_TURN
        elif chunk["finish_reason"] == "tool_calls":
            return message, StopReason.TOOL_USE
        elif chunk["finish_reason"] == "length":
            return message, StopReason.MAX_TOKENS
        else:
            return message, StopReason.ERROR
    
    async def parse_stream_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None]:
        full_response = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            chunk = chunk["choices"][0]
            if chunk["stop_reason"]:    
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_response)
                )
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "tool_calls":
                    yield None, StopReason.TOOL_USE, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
            else:
                yield chunk["message"]["content"], None, None
                full_response.append(chunk["message"]["content"])
            

class MistralInstructImplementation(BaseModelImplementation):
    """
    Read more: https://docs.mistral.ai/guides/prompting_capabilities/
    """
    
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
        template = env.get_template("mistral7_template.j2")
        prompt = template.render({"SYSTEM": system, "REQUEST": prompt})
        return prompt
    
    
    def prepare_request(
        self, 
        prompt: Union[str, MessageBlock, List[Dict]], 
        config: ModelConfig,
        system: Optional[str] = None, 
        document: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, document)
        
        return {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k
        }
    
    async def prepare_request_async(
        self, 
        prompt: Union[str, MessageBlock, List[Dict]], 
        config: ModelConfig,
        system: Optional[str] = None, 
        document: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, document)
        print(prompt)
        
        return {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k
        }
        
    def parse_response(
        self, 
        response: Any
    ) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response.read())
        chunk = chunk["outputs"][0]
        message = MessageBlock(
            role="assistant",
            content=chunk["text"]
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
        full_response = []
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            chunk = chunk["outputs"][0]
            if chunk["stop_reason"]:
                message = MessageBlock(
                    role="assistant",
                    content="".join(full_response)
                )
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
            else:
                yield chunk["text"], None, None
                full_response.append(chunk["text"])