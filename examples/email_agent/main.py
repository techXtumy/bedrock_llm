import asyncio
import sys
import os
import json
import traceback

from termcolor import cprint
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.bedrock_llm.client import LLMClient
from src.bedrock_llm.types.enums import ModelName
from src.bedrock_llm.config.base import RetryConfig
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.message import MessageBlock
from utils import get_user_input

from typing import Coroutine, List

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

tool_metadata_list = [{ 
    "name": "send_email",
    "description": "Sends an email with the specified details.",
    "parameters": {
        "type": "object",
        "properties": {
            "sender_email": {
                "type": "string",
                "description": "Email address of the sender."
            },
            "recipient_email": {
                "type": "string",
                "description": "Email address of the recipient."
            },
            "subject": {
                "type": "string",
                "description": "Subject of the email."
            },
            "body": {
                "type": "string",
                "description": "Content/body of the email."
            },
        },
        "required": [
            "sender_email",
            "recipient_email",
            "subject",
            "body"
        ]
    }
}]

tools = []
for i in tool_metadata_list:
    tools.append(str(i))

system = f"You are a helpful Agent Assistant. Here are tools that you can invoke: {"".join(tools)}"


async def process_tools(
    tool_list: List[Coroutine],
    timeout: float = 60.0
) -> List:
    result = []

    for coro in tool_list:
        try:
            # Get function name for better error reporting
            func_name = getattr(coro, '__qualname__', str(coro))
            
            # Handle the case where the coroutine might be malformed
            if not asyncio.iscoroutine(coro):
                error_info = {
                    'status': 'error',
                    'function': func_name,
                    'error': 'Invalid coroutine object',
                    'error_type': 'TypeError'
                }
                result.append(error_info)
                cprint(f"✗ Invalid coroutine: {func_name}", color="red")
                continue

            # Attempt to execute the coroutine with timeout
            try:
                tool_result = await asyncio.wait_for(coro, timeout=timeout)
                result.append({
                    'status': 'success',
                    'function': func_name,
                    'result': tool_result
                })
                cprint(f"✓ Successfully executed: {func_name}", color="green")
                
            except asyncio.TimeoutError:
                error_info = {
                    'status': 'error',
                    'function': func_name,
                    'error': f'Function timed out after {timeout} seconds',
                    'error_type': 'TimeoutError'
                }
                result.append(error_info)
                cprint(f"⚠ Timeout: {func_name}", color="yellow")
                
            except TypeError as e:
                error_info = {
                    'status': 'error',
                    'function': func_name,
                    'error': f'Invalid arguments: {str(e)}',
                    'error_type': 'TypeError'
                }
                result.append(error_info)
                cprint(f"✗ Argument Error in {func_name}: {str(e)}", color="red")
                
            except Exception as e:
                error_info = {
                    'status': 'error',
                    'function': func_name,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
                result.append(error_info)
                cprint(f"✗ Error in {func_name}: {str(e)}", color="red")
                
        except Exception as e:
            # Catch-all for any unexpected errors in error handling itself
            result.append({
                'status': 'error',
                'function': 'unknown',
                'error': f'Unexpected error: {str(e)}',
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
            cprint(f"✗ Unexpected error processing tool: {str(e)}", color="red")

    return result


async def main():
    client = LLMClient(
        region_name="us-east-1",
        model_name=ModelName.JAMBA_1_5_MINI,
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0)
    )
    
    chat_history = []
    
    user_input = "Send email to my boss on wishing him an happy birthday. My boss email is duy.doan@techxcorp.com, My name is Phicks."
    
    while True:
        
        # Add user message to chat history
        chat_history.append(MessageBlock(
            role="user",
            content=user_input
        ).model_dump())

        while True:
            
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
                        
            if stop_reason == "stop":
                break
            
        user_input = await get_user_input("\nEnter your next request or /bye to exit: ")
        
        if user_input.lower() == "/bye":
            break
        

if __name__ == "__main__":
    asyncio.run(main())