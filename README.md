# Bedrock LLM

A Python library for building LLM applications using Amazon Bedrock Provider and boto3 library. It aims to create best practices and production-ready solutions for various LLM models, including Anthropic, Llama, Amazon Titan, MistralAI, and AI21.

The library is structured into two main components:
1. `bedrock_be`: Infrastructure and services for deploying LLM applications.
2. `bedrock_llm`: LLM orchestration and interaction logic.

This structure allows for seamless integration of LLM capabilities with robust deployment and infrastructure management.

![Conceptual Architecture](/assests/bedrock_llm.png)

## Features

- Support for multiple LLM models through Amazon Bedrock
- Efficient LLM orchestration with `bedrock_llm`
- Infrastructure and deployment services with `bedrock_be`
- Agent-based interactions and tool calling
- Asynchronous and synchronous function support
- Performance monitoring and logging functionality
- Support for Retrieval-Augmented Generation (RAG)
- Multi-Agent systems (in progress)
- Workflows, nodes, and event-based systems (coming soon)
- Image generation, speech-to-text (STT), and text-to-speech (TTS) support (coming soon)

## Installation

You can install the Bedrock LLM library using pip:

```bash
pip install bedrock-llm
```

This library requires Python 3.9 or later.

## AWS Credentials Setup

Before using the library, make sure you have your AWS credentials properly configured:

1. Create or update your AWS credentials file at `~/.aws/credentials`:
```ini
[bedrock]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

2. Create or update your AWS config file at `~/.aws/config`:
```ini
[profile bedrock]
region = us-east-1
```

3. When initializing the client, specify the profile name:
```python
from bedrock_llm import LLMClient, ModelName, ModelConfig

# Create a LLM client with specific AWS profile
client = LLMClient(
    region_name="us-east-1",
    model_name=ModelName.MISTRAL_7B,
    profile_name="bedrock"  # Specify your AWS profile name
)
```

You can verify your credentials by running:
```bash
aws bedrock list-foundation-models --profile bedrock
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

Use the `monitor` decorators for performance monitoring:

```python
from bedrock_llm.monitor import Monitor

@Monitor.monitor_async
async def my_async_function():
    # Your async function code here

@Monitor.monitor_sync
def my_sync_function():
    # Your sync function code here
```

Use the `log` decorators for logging function calls:

```python
from bedrock_llm.monitor import Logging

@Logging.log_async
async def my_async_function():
    # Your async function code here

@Logging.log_sync
def my_sync_function():
    # Your sync function code here
```

These decorators are optimized for minimal performance impact on your application.

## Architecture

For a detailed overview of the library's architecture, please see [ARCHITECTURE.md](ARCHITECTURE.md).

## Examples

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
