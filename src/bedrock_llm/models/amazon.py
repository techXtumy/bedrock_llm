import json
import os

from jinja2 import Environment, FileSystemLoader
from typing import Any, AsyncGenerator, Tuple, List, Dict, Union, Optional

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from src.bedrock_llm.types.enums import StopReason


class TitanImplementation(BaseModelImplementation):
    """
    Read more: https://d2eo22ngex1n9g.cloudfront.net/Documentation/User+Guides/Titan/Amazon+Titan+Text+Prompt+Engineering+Guidelines.pdf
    """
    
    # Determine the absolute path to the templates directory
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")
    
    def load_template(
        self, 
        prompt: Union[MessageBlock, List[Dict]], 
        system: Optional[str], 
        document: Optional[str],
    ) -> str:
        env = Environment(loader=FileSystemLoader(self.TEMPLATE_DIR))
        template = env.get_template("amazon_template.j2")
        prompt = template.render({"SYSTEM": system, "REQUEST": prompt, "DOCUMENT": document})
        return prompt.strip() + " "
    
    def prepare_request(
        self, 
        prompt: Union[str, MessageBlock, List[Dict]], 
        config: ModelConfig,    
        system: Optional[str]=None, 
        document: Optional[str]=None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, document)
        
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": config.max_tokens,
                "temperature": config.temperature,
                "topP": config.top_p,
                "stopSequences": config.stop_sequences
            }
        }
    
    async def prepare_request_async(
        self, 
        prompt: Union[str, MessageBlock, List[Dict]], 
        config: ModelConfig,    
        system: Optional[str]=None, 
        document: Optional[str]=None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if not isinstance(prompt, str):
            prompt = self.load_template(prompt, system, document)
            
        print(prompt)
        
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