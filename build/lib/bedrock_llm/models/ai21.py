import json

from typing import Any, AsyncGenerator, Tuple, List, Dict, Optional, Union

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock, DocumentBlock, SystemBlock
from src.bedrock_llm.schema.tools import ToolMetadata
from src.bedrock_llm.types.enums import StopReason


class JambaImplementation(BaseModelImplementation):
    
    def prepare_request(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare the request body for the AI21 API.

        Args:
            prompt (str | List[Dict]): The prompt to send to the AI21 API.
            config (ModelConfig): The configuration for the AI21 API.
            system (Optional[str]): The system prompt to send to the AI21 API.
            documents (Optional[str]): The context documents to send to the AI21 API.
            tools (Optional[List[Dict] | Dict]): The tools to send to the AI21 API.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The request body for the AI21 API.

        Raises:
            ValueError: If the prompt is not a string or a list of dictionaries.
            ValueError: If the instruction is not a string.
            ValueError: If tools are provided (not supported by Jamba Model).
            ValueError: If documents are provided (not supported in the request payload).

        See more: https://docs.ai21.com/docs/prompt-engineering
        """
        messages = []
        
        if tools:
            raise ValueError("AI21 Jamba Model does not support tools. Please use another model.")
        
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
            "top_p": config.top_p,
            "temperature": config.temperature,
            "stop": config.stop_sequences,
            "n": config.number_of_responses,
        }
        
        return request_body
    
    async def prepare_request_async(
        self, 
        config: ModelConfig, 
        prompt: Union[str, MessageBlock, List[Dict]],
        system: Optional[Union[str, SystemBlock]] = None,
        tools: Optional[Union[List[ToolMetadata], List[Dict]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare the request body for the AI21 API.

        Args:
            prompt (str | List[Dict]): The prompt to send to the AI21 API.
            config (ModelConfig): The configuration for the AI21 API.
            system (Optional[str]): The system prompt to send to the AI21 API.
            documents (Optional[str]): The context documents to send to the AI21 API.
            tools (Optional[List[Dict] | Dict]): The tools to send to the AI21 API.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: The request body for the AI21 API.

        Raises:
            ValueError: If the prompt is not a string or a list of dictionaries.
            ValueError: If the instruction is not a string.

        See more: https://docs.ai21.com/docs/prompt-engineering
        """
        messages = []
        
        if tools:
            raise ValueError("Jamba Model currently does not support tools, please use other LLM")
        
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
            "top_p": config.top_p,
            "temperature": config.temperature,
            "stop": config.stop_sequences,
            "n": config.number_of_responses,
        }
            
        # Conditionally add tools if it is not None
        if tools is not None:
            if isinstance(tools, dict):
                tools = [tools]
            request_body["tools"] = tools
            
        return request_body
       
    @staticmethod
    def _extract_chunk_data(chunk: dict) -> tuple[Optional[str], Optional[str]]:
        """Extract text content and stop reason from a chunk."""
        if not chunk.get("choices"):
            return None, None
        
        choice = chunk["choices"][0]
        return (
            choice["delta"].get("content"),
            choice.get("finish_reason")
        )
        
    def parse_response(
        self, 
        response: Any
    ) -> Tuple[MessageBlock, StopReason]:
        chunk = json.loads(response.read())
        chunk = chunk["choices"][0]
        message = MessageBlock(
            role=chunk["message"]["role"],
            content=chunk["message"]["content"].strip(),
            tool_calls=chunk["message"].get("tool_calls", None)
        )
        if chunk.get("finish_reason") == "stop":
            stop_reason = StopReason.END_TURN
        elif chunk.get("finish_reason") == "length":
            stop_reason = StopReason.MAX_TOKENS
        else:
            stop_reason = StopReason.ERROR

        return message, stop_reason

    async def parse_stream_response(
        self,
        stream: Any
    ) -> AsyncGenerator[Tuple[Optional[str], Optional[StopReason], Optional[MessageBlock]], None]:
        """
        Parse the response from the Bedrock API, handling both text content
        and tool call requests.

        Args:
            stream: The response stream from the Bedrock API.

        Yields:
            Tuple containing either:
            - (str, None): Regular text chunks
            - (MessageBlock, str): Final message with optional tool calls and stop reason
        """
        full_answer = []

        for event in stream:
            # yield event, None
            try:
                chunk = json.loads(event["chunk"]["bytes"])
                text_chunk, stop_reason = self._extract_chunk_data(chunk)
                
                if stop_reason:
                    message = MessageBlock(role="assistant", content="".join(full_answer).strip())
                    if stop_reason == "stop":
                        yield None, StopReason.END_TURN, message
                    elif stop_reason == "length":
                        yield None, StopReason.MAX_TOKENS, message
                    break
                
                if not text_chunk:
                    continue

                if not stop_reason:
                    yield text_chunk, None, None
                    full_answer.append(text_chunk)
                    
            except Exception as e:
                print(f"Unexpected error processing chunk: {str(e)}")
                continue