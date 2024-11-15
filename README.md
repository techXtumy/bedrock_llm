# Bedrock LLM

A Python library for building LLM applications using Amazon Bedrock Provider boto3 library. it aim to fast prototyping using variety of Foundation Model from Amazon Bedrock. It also aim to easy integration  Amazon Bedrock Foundation Model with other services. 

The library is crafted to create best practices, production ready on Anthropic Model Family, Llama, Amazon Titan Text, MistralAI and AI21.

![Conceptual Architecture](/assests/bedrock_llm.png)

## Features

- Support for Retrieval-Augmented Generation (RAG)
- Support for Agent-based interactions
- Support for Multi-Agent systems (in progress)
- Support for creating workflows, nodes, and event-based systems (coming soon)
- Support for image generated model and speech to text (STT), text to speech (TTS) (coming soon)
- Performance monitoring for both asynchronous and synchronous functions
- Logging functionality for tracking function calls, inputs, and outputs

## Installation

You can install the Bedrock LLM library using pip:

```
pip install bedrock-llm

This library requires Python 3.9 or later.
```

## Usage

Here's a quick example of how to use the Bedrock LLM library:

### Simple text generation

```python
from bedrock_llm import LLMClient, ModelName, ModelConfig

# Create a LLM client
client = LLMClient(
    region_name="us-east-1",
    model_name=ModelName.MISTRAL_7B
)

# Create a configuration for inference parameters
config = ModelConfig(
    temperature=0.1,
    top_p=0.9,
    max_tokens=512
)

# Create a prompt
prompt = "Who are you?"

# Invoke the model and get results
response, stop_reason = client.generate(config, prompt)

# Print out the results
cprint(response.content, "green")
cprint(stop_reason, "red")
```

### Simple tool calling

```python
from bedrock_llm import Agent, ModelName
from bedrock_llm.schema.tools import ToolMetadata, InputSchema, PropertyAttr

agent = Agent(
    region_name="us-east-1",
    model_name=ModelName.CLAUDE_3_5_HAIKU
)

# Define the tool description for the model
get_weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the weather in specific location",
    input_schema=InputSchema(
        type="object",
        properties={
            "location": PropertyAttr(
                type="string",
                description="Location to search for, example: New York, WashingtonDC, ..."
            )
        },
        required=["location"]
    )
)

# Define the tool
@Agent.tool(get_weather_tool)
async def get_weather(location: str):
    return f"{location} is 20*C"


async def main():
    prompt = input("User: ")

    async for token, stop_reason, response, tool_result in agent.generate_and_action_async(
        prompt=prompt,
        tools=["get_weather"]
    ):
        if token:
            print(token, end="", flush=True)
        if stop_reason:
            print(f"\n{stop_reason}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Monitoring and Logging

The `monitor`  decorators for monitoring the performance of both asynchronous and synchronous functions:

```python
from bedrock_llm.monitor import Monitor

@monitor_async
async def my_async_function():
    # Your async function code here

@monitor_sync
def my_sync_function():
    # Your sync function code here
```

The `log` decorators for logging function calls, inputs, and outputs:

```python
from bedrock_llm.monitor import Logging

@log_async
async def my_async_function():
    # Your async function code here

@log_sync
def my_sync_function():
    # Your sync function code here
```

These decorators are optimized for minimal performance impact on your application.

### More examples

For more detailed usage instructions and API documentation, please refer to our [documentation](https://github.com/yourusername/bedrock_llm/wiki).

You can also see some examples of how to use and build LLM flow using the libary 
- [basic](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/1_basic.py)
- [stream response](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/2_stream_response.py)
- [all support llm](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/3_all_llm.py)
- [simple chat bot](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/4_chatbot.py)
- [tool calling](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/5_tool_calling.py)
- [agent](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/7_agent.py)

and more to come, we are working on it :)

## Requirements

- Python 3.9+
- pydantic>=2.0.0
- boto3>=1.18.0
- botocore>=1.21.0
- jinja>=3.1.4

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.