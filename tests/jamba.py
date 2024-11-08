import asyncio
import sys
import os
import random
import pytz
import traceback

from termcolor import cprint
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import LLMClient
from src.bedrock_llm.types.enums import ModelName
from src.bedrock_llm.config.base import RetryConfig
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.schema.tools import ToolMetadata, InputSchema, PropertyAttr
from src.bedrock_llm.schema.message import MessageBlock, ToolUseBlock, TextBlock, ToolResultBlock
from src.bedrock_llm.utils.prompt import llama_format, llama_tool_format

from typing import Literal, List, Optional, Coroutine


async def main():
    model_config = ModelConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=2048,
    )
    client = LLMClient(
        region_name="us-east-1",
        model_name=ModelName.JAMBA_1_5_LARGE,
        retry_config=RetryConfig(
            max_retries=3,
            retry_delay=1,
            exponential_backoff=True
        ),
    )
    get_weather_tool = ToolMetadata(
            name="get_weather",
            description="Get the weather of a city",
            parameters= InputSchema(
                type="object",
                properties={
                    "location": PropertyAttr(
                        type="string",
                        description="The city and state, e.g. San Francisco, CA"
                    )
                },
                required=["location"],
            )
        ).model_dump()
    
    async for message, stop_reason in client.generate(
        prompt="Hello, can you give me the weather in Hanoi",
        config=model_config,
        system="You are the weather forcaster, you have the tool get the weather of every location in the world.",
        tools=[get_weather_tool]
    ):
        print(message, end="", flush=True)
    


if __name__ == "__main__":
    asyncio.run(main())