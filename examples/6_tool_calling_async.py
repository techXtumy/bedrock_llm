# Add for print console with color
import json
from typing import List

from termcolor import cprint

from bedrock_llm import (LLMClient, MessageBlock, ModelConfig, ModelName,
                         RetryConfig)
from bedrock_llm.schema.tools import InputSchema, PropertyAttr, ToolMetadata
from bedrock_llm.types.enums import StopReason

system = "You are a helpful assistant. You have access to realtime information. You can use tools to get the real time data weather of a city."

# Create a LLM client
client = LLMClient(
    region_name="us-east-1",
    model_name=ModelName.LLAMA_3_2_3B,
    memory=[],
    retry_config=RetryConfig(max_attempts=3),
)

# Create a configuration for inference parameters
config = ModelConfig(temperature=0.1, top_p=0.9, max_tokens=512)

# Create tool definition
get_weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the real time weather of a city",
    input_schema=InputSchema(
        type="object",
        properties={
            "location": PropertyAttr(
                type="string", description="The city to get the weather of"
            )
        },
        required=["location"],
    ),
)

# Create a system prompt with a list of examples
history = [
    MessageBlock(role="user", content="What is the weather in Denmark"),
    MessageBlock(role="assistant", content="[get_weather(location='Denmark')]"),
    MessageBlock(role="tool", content="tools_result: Denmark is 10*C"),
    MessageBlock(role="assistant", content="The weather in Denmark is 10*C"),
    MessageBlock(role="user", content="What is the weather in New York and Toronto?"),
]


async def get_weather(location: str):
    # Mock function to get weather
    return f"tools_result: {location} is 20*C"


async def call_tools(response: str) -> MessageBlock:
    result_list = MessageBlock(role="tool", content=[])

    # Remove the tags and whitespace
    cleaned_response = response.strip("<|python_tag|>").strip()

    try:
        tools_list = eval(cleaned_response)

        # Validate that tools_list is actually a list
        if not isinstance(tools_list, list):
            raise ValueError("Expected a list of tools")

        # Process each function
        for func in tools_list:
            # Add validation for func
            if asyncio.iscoroutine(func):
                try:
                    tool_result = await asyncio.wait_for(func, timeout=10)
                    result_list.content.append(tool_result)
                except asyncio.TimeoutError:
                    raise Exception("Tool execution timed out")
                except Exception as e:
                    raise Exception(f"Tool execution failed: {str(e)}")
            else:
                raise Exception("Invalid tool format")

    except json.JSONDecodeError:
        raise Exception("Invalid JSON format in response")
    except Exception as e:
        raise Exception(f"Error processing tools: {str(e)}")

    return result_list


async def main():
    while True:
        # Invoke the model and get results
        async for token, stop_reason, response in client.generate_async(
            config=config, prompt=history, system=system, tools=[get_weather_tool]
        ):
            # Print out the results
            if not stop_reason:
                cprint(token, "green", end="", flush=True)

            if stop_reason == StopReason.TOOL_USE:
                history.append(response)
                # Call the tool
                result = await call_tools(response.content)
                history.append(result)
                print()  # Print break line for readability
                break

        if stop_reason == StopReason.END_TURN:
            history.append(response)
            print()
            break

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
