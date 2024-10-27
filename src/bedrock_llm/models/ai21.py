import json
import uuid
import re

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
    def _process_tool_calls(tool_calls_json: str) -> List[str]:
        """Process and transform tool calls JSON to the required format."""
        try:
            # Step 1: Check if it uses single quotes instead of double quotes
            if "'" in tool_calls_json and '"' not in tool_calls_json:
                # Replace single quotes with double quotes
                cleaned_data = tool_calls_json.replace("'", '"')
            else:
                cleaned_data = tool_calls_json.strip()

            # Step 2: Remove any extraneous whitespace
            cleaned_data = re.sub(r'\s+', ' ', cleaned_data)

            # Step 3: Attempt to parse with json.loads()
            tool_calls_data = json.loads(cleaned_data)

            # Ensure it's a list, as expected
            if not isinstance(tool_calls_data, list):
                print("Expected a list, but got:", type(tool_calls_data))
                return []

            result = []
            for tool_call in tool_calls_data:
                call_id = tool_call.get("id")
                function_name = tool_call.get("function", {}).get("name")
                arguments = tool_call.get("function", {}).get("arguments", {})

                # Construct the function call string with arguments
                args_str = ', '.join([f'{key}="{value}"' for key, value in arguments.items()])
                function_call = f'{function_name}({args_str})'

                # Append to result in the desired format
                result.append([call_id, function_call])

            return result

        except json.JSONDecodeError:
            print(f"Error decoding tool calls {tool_calls_json}")
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
        full_answer = []

        for event in stream:
            try:
                chunk = json.loads(event["chunk"]["bytes"])
                text_chunk, stop_reason = self._extract_chunk_data(chunk)
                
                if stop_reason:
                    yield "".join(full_answer), stop_reason

                if not text_chunk:
                    yield text_chunk, None
                    
            except Exception as e:
                print(f"Unexpected error processing chunk: {str(e)}")
                continue