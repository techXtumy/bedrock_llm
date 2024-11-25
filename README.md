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
- Enhanced Agent-based interactions with:
  - Robust tool validation and execution
  - Comprehensive error handling and logging
  - Configurable memory management
  - Type-safe responses with `AgentResponse`
  - Support for multiple LLM tool-calling conventions (Claude, Llama, Mistral, etc.)
- Asynchronous and synchronous function support
- Performance monitoring and logging functionality
- Support for Retrieval-Augmented Generation (RAG)
- Optimized Pipeline System:
  - Modular node-based architecture
  - Batch processing with configurable parameters
  - In-memory caching with size management
  - Parallel processing with thread pools
  - Type-safe node connections
  - Event-driven data flow
  - Filter nodes for data validation
- Multi-Agent systems (in progress)
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

## Quick Start

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

### Pipeline Usage

```python
from bedrock_llm.pipeline import Pipeline, BatchNode, CachedNode

# Create a pipeline for efficient text processing
pipeline = Pipeline("text-processor")

# Add optimized nodes
batch_node = BatchNode(
    "batch-embeddings",
    embed_batch_func,
    batch_size=32
)

cache_node = CachedNode(
    "cached-process",
    process_func,
    cache_size=1000
)

# Connect nodes
pipeline.add_node(batch_node)
pipeline.add_node(cache_node)
batch_node.connect(cache_node)

# Process data
result = await pipeline.execute(input_data)
```

### Agent Features

The Agent class in `bedrock_llm` provides powerful capabilities for building LLM-powered applications:

#### Tool Management

```python
from bedrock_llm import Agent, ToolMetadata
from typing import Dict

# Define a tool with metadata
@Agent.tool(
    metadata=ToolMetadata(
        name="search",
        description="Search for information",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    )
)
async def search(query: str) -> Dict:
    # Tool implementation
    pass
```

#### Error Handling

The library provides comprehensive error handling with custom exceptions:

```python
try:
    result = await agent.generate_and_action_async(
        prompt="Search for Python tutorials",
        tools=["search"]
    )
except ToolExecutionError as e:
    print(f"Tool '{e.tool_name}' failed: {e.message}")
    if e.original_error:
        print(f"Original error: {e.original_error}")
```

#### Memory Management

Configure memory limits to manage conversation history:

```python
agent = Agent(
    region_name="us-west-2",
    model_name=ModelName.ANTHROPIC_CLAUDE_V2,
    memory_limit=100  # Keep last 100 messages
)
```

#### Type-Safe Responses

The library now provides type-safe responses using TypedDict:

```python
async for response in agent.generate_and_action_async(...):
    token: Optional[str] = response["token"]
    stop_reason: Optional[StopReason] = response["stop_reason"]
    message: Optional[MessageBlock] = response["message"]
    tool_results: Optional[List] = response["tool_results"]
```

#### Tool States

Support for different LLM tool-calling conventions:

- Claude/Llama style: Uses ToolUseBlock for tool execution
- Mistral/Jamba style: Uses ToolCallBlock for function calling

## Monitoring and Logging

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

The Bedrock LLM library is architected for scalability, reliability, and extensibility. Key architectural components include:

### Core Components

- **Client Layer**: Robust interfaces for Bedrock service interaction
  - Async/Sync clients with streaming support
  - Configurable retry logic
  - Memory management
  - Type-safe operations

- **Model Layer**: Flexible model implementation framework
  - Support for multiple LLM providers
  - Custom parameter optimization
  - Response formatting

- **Agent System**: Advanced autonomous capabilities
  - Tool management and execution
  - State preservation
  - Error handling
  - Type-safe responses

### Infrastructure (bedrock_be)

- AWS service integration
- Deployment automation
- Monitoring and scaling
- Security management

For a comprehensive architectural overview, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Examples

For more detailed usage instructions and API documentation, please refer to our [documentation](https://github.com/yourusername/bedrock_llm/LIBRARY_DOCUMENTATION.md).

You can also see some examples of how to use and build LLM flow using the libary

- [basic](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/1_basic.py)
- [stream response](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/2_stream_response.py)
- [all support llm](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/3_all_llm.py)
- [simple chat bot](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/4_chatbot.py)
- [tool calling](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/5_tool_calling.py)
- [agent](https://github.com/Phicks-debug/bedrock_llm/blob/main/examples/7_agent.py)

and more to come, we are working on it :)

## More Documents and wanna understand the project more?

For more detailed documentation, examples, and project insights, please refer to the following resources:

- **Documentation**: [https://github.com/Phicks-debug/bedrock_llm/LIBRARY_DOCUMENTATION.md](<https://github.com/Phicks-debug/bedrock_llm/LIBRARY_DOCUMENTATION.md>)
- **Examples**: [https://github.com/Phicks-debug/bedrock_llm/examples](https://github.com/Phicks-debug/bedrock_llm/examples)
- **Project Insights**: [https://github.com/Phicks-debug/bedrock_llm/docs](https://github.com/Phicks-debug/bedrock_llm/docs)

Feel free to reach out if you have any questions or need further assistance!

## Requirements

- python>=3.9
- pydantic>=2.0.0
- boto3>=1.18.0
- botocore>=1.21.0
- jinja2>=3.1.2
- psutil>=5.9.0
- pytz>=2023.3
- termcolor>=2.3.0
- databases[postgresql]>=0.7.0
- sqlalchemy>=2.0.0
- asyncpg>=0.27.0  # PostgreSQL async driver
- types-redis>=4.6.0
- types-pytz
- rx==3.2.0

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
