import asyncio
import sys
import os
import pytz
import json
import boto3

from termcolor import cprint
from datetime import datetime
import xml.etree.ElementTree as ET

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import LLMClient
from src.bedrock_llm.types.enums import ModelName
from src.bedrock_llm.config.base import RetryConfig
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.message import MessageBlock, ToolUseBlock, TextBlock, ToolResultBlock, ToolCallBlock

from typing import Literal, List, Optional, Callable, Coroutine, Any

"""
Watch this video for connecting to Outlook

https://www.youtube.com/watch?v=7Cve_k4C_Ts
"""

model_config = ModelConfig(
    temperature=0.6,
    max_tokens=2048,
    top_p=1
)
retry_config = RetryConfig(
    max_retries=3,
    retry_delay=0.5
)
runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

chat_history = []
tool_metadata_list = [
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a specified email address",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "The email address to send the email to"
                        },
                        "subject": {
                            "type": "string",
                            "description": "The subject of the email"
                        },
                        "body": {
                            "type": "string",
                            "description": "The body of the email. You need to generate this yourself"
                        }
                    },
                    "required": ["email", "subject", "body"]
                }
            }
        }, 
        {
            "type": "function",
            "function": {
                "name": "retrieve_information",
                "description": "Retrieve informations from the HR company Policies Knowledge Base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query for searching for the information"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
system = f"""You are a helpful assistant. This is the real time data {datetime.now(tz = pytz.timezone("Asia/Bangkok")).strftime('%Y-%m-%d %H:%M:%S %Z')}

Here is a list of functions in JSON format that you can invoke.
Please call the model follow by your own instruction that you have trained on.
{tool_metadata_list}"""


async def get_user_input(
    placeholder: str
) -> str:
    return input(placeholder)


async def send_email(
    email: str,
    subject: str,
    body: str
):
    kwargs = {
        "recipientEmail": to,
        "subject": subject,
        "body": body
    }
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: runtime.send_email(**kwargs))
    return result


async def retrieve_information(query: str):
    kwargs = {
        "knowledgeBaseId": "VSR83TL8CR",
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": 25,
                "overrideSearchType": "HYBRID"
            }
        },
        "retrievalQuery": {
            "text": query
        }
    }
    
    # Run boto3 call in a thread pool since it's blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: runtime.retrieve(**kwargs))
    return build_context_kb_prompt(result)


def build_context_kb_prompt(
    retrieved_json_file, 
    min_relevant_percentage: float = 0.3, 
    debug=False
):
    if not retrieved_json_file:
        return ""
    
    documents = ET.Element("documents")
    
    if retrieved_json_file["ResponseMetadata"]["HTTPStatusCode"] != 200:
        documents.text = "Error in getting data source from knowledge base. No context is provided"
    else:
        body = retrieved_json_file["retrievalResults"]
        for i, context_block in enumerate(body):
            if context_block["score"] < min_relevant_percentage:
                break
            document = ET.SubElement(documents, "document", {"index": str(i + 1)})
            source = ET.SubElement(document, "source")
            content = ET.SubElement(document, "document_content")
            source.text = iterate_through_location(context_block["location"])
            content.text = context_block["content"]["text"]
    
    return ET.tostring(documents, encoding="unicode", method="xml")


def iterate_through_location(location: dict):
    # Optimize by stopping early if uri or url is found
    for loc_data in location.values():
        if isinstance(loc_data, dict):
            uri = loc_data.get("uri")
            if uri:
                return uri
            url = loc_data.get("url")
            if url:
                return url
    return None


async def process_tools(
    tool_list: List[ToolCallBlock],
    timeout: float = 60.0
) -> List[MessageBlock]:
    
    for tool in tool_list:
        
        message = MessageBlock(
            role="tool", 
            content="", 
            tool_calls_id=""
        )
        
        result = ""

        tool_function = tool.function.name
        if tool_function:
            try:
                result += await globals()[tool_function](**tool.function.arguments)
                is_error = False
            except Exception as e:
                result = str(e)
                is_error = True
        else:
            result = f"Tool {tool.function.name} not found"
            is_error = True

        message.content.append(
            ToolResultBlock(
                type="tool_result",
                tool_use_id=tool.id,
                is_error=is_error,
                content=result
            )
        )

    return message


async def main():
    client = LLMClient(
        region_name="us-east-1",
        model_name=ModelName.JAMBA_1_5_LARGE,
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0)
    )
    
    
    
    while True:
        
        user_input = await get_user_input("Enter a prompt: ")
        
        # Add user message to chat history
        chat_history.append(MessageBlock(
            role="user",
            content=user_input
        ).model_dump())

    
        # For debugging
        print("\nCurrent Chat History:")
        print(json.dumps(chat_history, indent=2))
        
        async for chunk, stop_reason in client.generate(
                prompt=chat_history,
                system=system,
                config=model_config
            ):
                if stop_reason is None:
                    cprint(chunk, color="green", end="", flush=True)
                else:
                    # Always show the stop reason
                    cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")
                    
                    if isinstance(chunk, MessageBlock):
                        # Handle tool calls if present
                        if chunk.tool_calls:
                            cprint(f"Tool calls detected: {chunk.tool_calls}", color="cyan", flush=True)
                            chat_history.append(chunk.model_dump())
                            break
                        else:
                            # For regular responses, add to chat history
                            chat_history.append(chunk.model_dump())
                            break
        
        if user_input.lower() == "/bye":
            break
        

if __name__ == "__main__":
    asyncio.run(main())