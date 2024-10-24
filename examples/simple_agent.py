import asyncio
import sys
import os

from termcolor import cprint

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import LLMClient
from src.bedrock_llm.types.enums import ModelName
from src.bedrock_llm.config.base import RetryConfig
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.tools import ToolMetadata, InputSchema, PropertyAttr


async def main():
    
    # Initialize the client
    client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.CLAUDE_3_5_SONNET,
        retry_config=RetryConfig(max_retries=3, retry_delay=1.0)
    )
    
    # Using Claude with tools
    system_prompt = "You are a helpful assistant."
    message = "What is the weather like now?"
    tool = ToolMetadata(
        name="get_weather",
        description="Get the current weather",
        input_schema=InputSchema(
            type="object",
            properties={
                "location": PropertyAttr(
                    type="string",
                    description="The city and state, e.g. San Francisco, CA"
                )
            },
            required=["location"]
        )
    ).model_dump()
    
    config = ModelConfig(temperature=0.7, max_tokens=512)
    async for chunk, stop_reason in client.generate(
        prompt=message,
        config=config,
        system=system_prompt,
        tools=tool,
    ):
        if isinstance(chunk, str):
            cprint(chunk, color="green", end="", flush=True)
        if stop_reason == "tool_use":
            cprint(f"\nGeneration stopped: {stop_reason}", color="cyan")
            print("Weather today: 36'C in Hanoi.")
        elif stop_reason:
            cprint(f"\nGeneration stopped: {stop_reason}\n", color="red")


if __name__ == "__main__":
    asyncio.run(main())