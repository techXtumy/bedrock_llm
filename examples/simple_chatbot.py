import asyncio
import sys
import os

from termcolor import cprint

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import LLMClient
from src.bedrock_llm.types.enums import ModelName
from src.bedrock_llm.config.base import RetryConfig
from src.bedrock_llm.schema.message import MessageBlock


# Function to handle user input asynchronously
async def get_user_input(placeholder: str) -> str:
    return input(placeholder)


async def chat_with_titan():
    
    """
    Read more about prompting for Titan Text:
    https://d2eo22ngex1n9g.cloudfront.net/Documentation/User+Guides/Titan/Amazon+Titan+Text+Prompt+Engineering+Guidelines.pdf
    """
    
    # Initialize the client
    client = LLMClient(
        region_name="us-east-1",
        model_name=ModelName.TITAN_PREMIER,
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0)
    )
    
    # Initialize the chat history
    chat_history = []
    
    # Receive first user input
    input_prompt = await get_user_input("Enter a prompt: ")
    
    while True:

        # Save the user input to chat history
        chat_history.append(f"User: {input_prompt}\nBot: ")

        # Simple text generation
        print(chat_history, end="", flush=True)
        async for chunk, stop_reason in client.generate(
            prompt="".join(chat_history)
        ):
            if isinstance(chunk, str):
                cprint(chunk, color="green", end="", flush=True)
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")

        # Save response to chat history
        chat_history.append(f"{chunk}\n\n")
        
        # Check for bye bye
        if input_prompt.lower() == "/bye":
            break
        
        # Receive user input
        input_prompt = await get_user_input("Enter a prompt: ")
        

async def chat_with_claude():
    # Initialize the client
    client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.CLAUDE_3_5_SONNET,
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0)
    )
    
    # Initialize the chat history
    chat_history = []
    
    while True:
        
        # Receive user input
        input_prompt = await get_user_input("Enter a prompt: ")
        
        # Save the user input to chat history
        chat_history.append(
            MessageBlock(
                role="user", 
                content=input_prompt
            ).model_dump()
        )
        
        # Simple text generation
        async for chunk, stop_reason in client.generate(
            prompt=chat_history
        ):
            if isinstance(chunk, str):
                cprint(chunk, color="green", end="", flush=True)
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")
        
        # Save response to chat history
        chat_history.append(chunk.model_dump())
        
        # Check for bye bye
        if input_prompt.lower() == "/bye":
            break


async def chat_with_llama():
    # Initialize the client
    client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.LLAMA_3_2_90B,
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0)
    )
    
    # Initialize the chat history as a list for efficient string handling
    chat_history = []
    
    # Predefine the system message to avoid repetition
    system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant<|eot_id|>"
    chat_history.append(system)

    while True:
        # Get user input asynchronously
        input = await get_user_input("Enter a prompt: ")
        
        # Format and save user message to chat history
        formatted_msg = f"<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        chat_history.append(formatted_msg)
        
        # Generate a response from the model
        async for chunk, stop_reason in client.generate(prompt="".join(chat_history)):
            if isinstance(chunk, str):
                cprint(chunk, color="green", end="", flush=True)
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")
        
        # Append the bot response to chat history
        chat_history.append(f"{chunk}<|eot_id|>")
        
        # Exit if user types "/bye"
        if input.lower() == "/bye":
            break


async def chat_with_mistral():
    # Initialize the client
    client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.MISTRAL_7B,
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0)
    )
    
    chat_history = []
    
    # Predefine the system message to avoid repetition
    system = "<s>[INST] You are a helpful AI assistant that can answer and keep the conversation going.\n\n"
    chat_history.append(system)

    while True:
        input = await get_user_input("Enter a prompt: ")
        
        # Format and save user message to chat history
        formatted_msg = f"{input} [/INST] "
        chat_history.append(formatted_msg)
        
        # Generate a response from the model
        async for chunk, stop_reason in client.generate(prompt="".join(chat_history)):
            if isinstance(chunk, str):
                cprint(chunk, color="green", end="", flush=True)
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")
        
        # Append the bot response to chat history
        chat_history.append(f"{chunk} </s>[INST] ")
        
        if input.lower() == "/bye":
            break


if __name__ == "__main__":
    
    mode_selection = input("Select mode (1 for Claude, 2 for Titan, 3 for Llama, and 4 for Mistral): ")
    if mode_selection == "1":
        asyncio.run(chat_with_claude())
    elif mode_selection == "2":
        asyncio.run(chat_with_titan())
    elif mode_selection == "3":
        asyncio.run(chat_with_llama())
    elif mode_selection == "4":
        asyncio.run(chat_with_mistral())