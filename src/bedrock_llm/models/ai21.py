import json

from typing import Any, AsyncGenerator, Tuple, List, Dict, Optional

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock, DocumentBlock, TextBlock


class JambaImplementation(BaseModelImplementation):
    
    async def prepare_request(
        self, 
        prompt: str | List[Dict],
        config: ModelConfig,
        system: Optional[str] = None,
        documents: Optional[List[DocumentBlock]] = None,
        tools: Optional[List[Dict] | Dict] = None,
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
        if isinstance(prompt, str):
            messages = [
                MessageBlock(
                    role="user", 
                    content=prompt
                ).model_dump()
            ]
        else:
            messages = prompt
        
        if system is not None:
            system = MessageBlock(
                role="system",
                content=system
            ).model_dump()
            messages.insert(0, system)
            
            print(messages)
        
        request_body = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "temperature": config.temperature,
            "stop": config.stop_sequences,
            "n": config.number_of_responses,
        }
        
        # Conditionally add tools if it is not None
        if documents is not None:
            request_body["documents"] = documents
            
        # Conditionally add tools if it is not None
        if tools is not None:
            if isinstance(tools, dict):
                tools = [tools]
            request_body["tools"] = tools
            
        return request_body
    
    
    async def parse_response(
        self, 
        stream: Any
    ) -> AsyncGenerator[Tuple[str | MessageBlock, Optional[str]], None]:
        """
        Parse the response from the AI21 API.

        Args:
            stream (Any): The response from the AI21 API.

        Yields:
            Tuple[str, None]: The generated text and None.

        Raises:
            ValueError: If the response is not a dictionary.
            ValueError: If the response does not contain the expected keys.

        See more: https://docs.ai21.com/reference/jamba-15-api-ref
        
        I have not do the tool calling.
        """
        full_answer = []
        
        for event in stream:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("choices"):
                text_chunk = chunk["choices"][0]["delta"].get("content")
                stop_reason = chunk["choices"][0]["finish_reason"]
                if text_chunk:
                    yield text_chunk, None
                    full_answer.append(text_chunk)
                elif stop_reason:
                    yield MessageBlock(
                        role="assistant", 
                        content="".join(full_answer)
                        ), stop_reason
        return