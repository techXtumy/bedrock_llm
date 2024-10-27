import json
import uuid

from typing import Any, AsyncGenerator, Tuple, List, Dict, Optional

from src.bedrock_llm.models.base import BaseModelImplementation, ModelConfig
from src.bedrock_llm.schema.message import MessageBlock, DocumentBlock, ToolCallBlock


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


    @staticmethod
    def _process_tool_calls(tool_calls_json: str) -> List[ToolCallBlock]:
        """Process and validate tool calls JSON."""
        try:
            tool_calls_data = json.loads(tool_calls_json)
            return [
                ToolCallBlock(
                    id=uuid.uuid4().hex,
                    type="function",
                    function=tool_call,
                ) for tool_call in tool_calls_data
            ]
        except json.JSONDecodeError:
            print("Error decoding tool calls")
            return []


    async def parse_response(
        self,
        stream: Any
    ) -> AsyncGenerator[Tuple[str | MessageBlock, Optional[str]], None]:
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
        full_answer: List[str] = []
        buffer: List[str] = []
        capturing_tool_calls = False
        tool_calls = None

        for event in stream:
            try:
                chunk = json.loads(event["chunk"]["bytes"])
                text_chunk, stop_reason = self._extract_chunk_data(chunk)
                
                if stop_reason:
                    yield MessageBlock(
                        role="assistant",
                        content="".join(full_answer).strip(),
                        tool_calls=tool_calls
                    ), stop_reason

                if not text_chunk:
                    continue

                if "<tool_calls>" in text_chunk:
                    capturing_tool_calls = True
                    content_after_tag = text_chunk.split("<tool_calls>")[1]
                    buffer.append(content_after_tag)
                    yield text_chunk, None
                    continue

                if "</tool_calls>" in text_chunk and capturing_tool_calls:
                    capturing_tool_calls = False
                    content_before_tag = text_chunk.split("</tool_calls>")[0]
                    buffer.append(content_before_tag)
                    
                    tool_calls = self._process_tool_calls("".join(buffer).strip())
                    buffer.clear()
                    
                    text_chunk = text_chunk.split("</tool_calls>")[1]
                    stop_reason = "tool_call"

                elif capturing_tool_calls:
                    buffer.append(text_chunk)
                    continue

                if text_chunk:
                    yield text_chunk, None
                    full_answer.append(text_chunk)

            except json.JSONDecodeError:
                print(f"Error decoding chunk: {event}")
                continue
            except Exception as e:
                print(f"Unexpected error processing chunk: {str(e)}")
                continue