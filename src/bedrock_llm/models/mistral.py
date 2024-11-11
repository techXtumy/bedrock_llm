import json
import os

from jinja2 import Environment, FileSystemLoader
from typing import Any, AsyncGenerator, Optional, List, Dict, Tuple, Union

from ..models.base import BaseModelImplementation, ModelConfig
from ..schema.message import MessageBlock, SystemBlock, ToolCallBlock, TextBlock
from ..schema.tools import ToolMetadata
from ..types.enums import ToolChoiceEnum, StopReason


class MistralChatImplementation(BaseModelImplementation):
    """
    Read more: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-large-2407.html
    """
    
    def _parse_tool_metadata(self, tool: Union[ToolMetadata, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse a ToolMetadata object or a dictionary into the format required by the Mistral model.
        """

        if isinstance(tool, dict):
            # Handle all dictionary inputs consistently
            if "type" in tool and tool["type"] == "function":
                function_data = tool.get("function", {})
            else:
                function_data = tool

            return {
                "type": "function",
                "function": {
                    "name": function_data.get("name", "unnamed_function"),
                    "description": function_data.get("description", "No description provided"),
                    "parameters": function_data.get("input_schema", {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }

        if isinstance(tool, ToolMetadata):
            mistral_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            if tool.input_schema:
                for prop_name, prop_attr in tool.input_schema.properties.items():
                    mistral_tool["function"]["parameters"]["properties"][prop_name] = {
                        "type": prop_attr.type,
                        "description": prop_attr.description
                    }
                
                if tool.input_schema.required:
                    mistral_tool["function"]["parameters"]["required"] = tool.input_schema.required
            
            return mistral_tool

        raise ValueError(f"Unsupported tool type: {type(tool)}. Expected Dict or ToolMetadata.")
    
    def prepare_request(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict], ToolMetadata, Dict]] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs
    ) -> Dict[str, Any]:

        if tools and not isinstance(tools, (list, dict, ToolMetadata)):
            raise ValueError("Tools must be a list, dictionary, or ToolMetadata object.")
        
        messages = []
        if isinstance(prompt, str):
            messages.append(MessageBlock(role="user", content=prompt).model_dump())
        elif isinstance(prompt, MessageBlock):
            messages.append(prompt.model_dump())
        elif isinstance(prompt, list):
            messages.extend([msg.model_dump() if isinstance(msg, MessageBlock) else msg for msg in prompt])
        
        if system is not None:
            system_content = system.text if isinstance(system, SystemBlock) else system
            system_message = MessageBlock(role="system", content=system_content).model_dump()
            messages.insert(0, system_message)
        
        request_body = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p
        }
        
        if tools is not None:
            if isinstance(tools, (dict, ToolMetadata)):
                parsed_tools = [self._parse_tool_metadata(tools)]
            elif isinstance(tools, list):
                parsed_tools = []
                for tool in tools:
                    if isinstance(tool, (dict, ToolMetadata)):
                        parsed_tools.append(self._parse_tool_metadata(tool))
                    else:
                        raise ValueError(f"Unsupported tool type in list: {type(tool)}. Expected Dict or ToolMetadata.")
            else:
                raise ValueError(f"Unsupported tools type: {type(tools)}. Expected List, Dict, or ToolMetadata.")
            request_body["tools"] = parsed_tools
        
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice
        
        return request_body
    
    async def prepare_request_async(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict], ToolMetadata, Dict]] = None,
        tool_choice: Optional[ToolChoiceEnum] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if tools and not isinstance(tools, (list, dict, ToolMetadata)):
            raise ValueError("Tools must be a list, dictionary, or ToolMetadata object.")
        
        messages = []
        if isinstance(prompt, str):
            messages.append(MessageBlock(role="user", content=prompt).model_dump())
        elif isinstance(prompt, MessageBlock):
            messages.append(prompt.model_dump())
        elif isinstance(prompt, list):
            messages.extend([msg.model_dump() if isinstance(msg, MessageBlock) else msg for msg in prompt])
        
        if system is not None:
            if isinstance(system, SystemBlock):
                system = system.text
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
            if isinstance(tools, (dict, ToolMetadata)):
                parsed_tools = [self._parse_tool_metadata(tools)]
            elif isinstance(tools, list):
                parsed_tools = []
                for tool in tools:
                    if isinstance(tool, (dict, ToolMetadata)):
                        parsed_tools.append(self._parse_tool_metadata(tool))
                    else:
                        raise ValueError(f"Unsupported tool type in list: {type(tool)}. Expected Dict or ToolMetadata.")
            else:
                raise ValueError(f"Unsupported tools type: {type(tools)}. Expected List, Dict, or ToolMetadata.")
            request_body["tools"] = parsed_tools
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
                content = "".join(full_response) if full_response else ""
                message = MessageBlock(
                    role="assistant",
                    content=[TextBlock(type="text", text=content)] if content else None
                )
                if chunk["stop_reason"] == "stop":
                    yield None, StopReason.END_TURN, message
                elif chunk["stop_reason"] == "tool_calls":
                    if "tool_calls" in chunk["message"]:
                        tool_calls = [
                            ToolCallBlock(
                                id=tool_call["id"],
                                type=tool_call["type"],
                                function=tool_call["function"]
                            ) for tool_call in chunk["message"]["tool_calls"]
                        ]
                        message.tool_calls = tool_calls
                    yield None, StopReason.TOOL_USE, message
                elif chunk["stop_reason"] == "length":
                    yield None, StopReason.MAX_TOKENS, message
                else:
                    yield None, StopReason.ERROR, message
                return
            else:
                if "content" in chunk["message"] and chunk["message"]["content"]:
                    yield chunk["message"]["content"], None, None
                    full_response.append(chunk["message"]["content"])
                elif "tool_calls" in chunk["message"]:
                    # Handle streaming tool calls
                    tool_calls = [
                        ToolCallBlock(
                            id=tool_call["id"],
                            type=tool_call["type"],
                            function=tool_call["function"]
                        ) for tool_call in chunk["message"]["tool_calls"]
                    ]
                    message = MessageBlock(role="assistant", content=[TextBlock(type="text", text="")], tool_calls=tool_calls)
                    yield None, None, message
            

class MistralInstructImplementation(BaseModelImplementation):
    """
    Read more: https://docs.mistral.ai/guides/prompting_capabilities/
    """
    
    # Determine the absolute path to the templates directory
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")
    
    def load_template(
        self, 
        prompt: Union[MessageBlock, List[Dict]], 
        system: Optional[str]
    ) -> str:
        env = Environment(loader=FileSystemLoader(self.TEMPLATE_DIR))
        template = env.get_template("mistral7_template.j2")
        return template.render({"SYSTEM": system, "REQUEST": prompt})
    
    
    def prepare_request(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if tools:
            raise ValueError("Mistral 7B Instruct does not support tools. Please use another model.")
                
        system_content = system.text if isinstance(system, SystemBlock) else system
        
        formatted_prompt = self.load_template(prompt, system_content) if not isinstance(prompt, str) else prompt
        
        return {
            "prompt": formatted_prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k
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
            raise ValueError("Mistral 7B Instruct does not support tools. Please use another model.")
        
        system = system.text if isinstance(system, SystemBlock) else system
        
        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system)
        
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