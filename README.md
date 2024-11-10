# Bedrock LLM

A Python library for building LLM applications using Amazon Bedrock Provider boto3 library. it aim to fast prototyping using variety of Foundation Model from Amazon Bedrock. It also aim to easy integration  Amazon Bedrock Foundation Model with other services. 

The library is crafted to create best practices, production ready on Anthropic Model Family, Llama, Amazon Titan Text, MistralAI and AI21.

## Features

- Support for Retrieval-Augmented Generation (RAG)
- Support for Agent-based interactions
- Support for Multi-Agent systems (in progress)
- Support for creating workflows, nodes, and event-based systems (comming soon)
- Support for image generated model and speech to text (STT), text to speech (TTS) (comming soon)

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

- Python 3.7+
- pydantic>=2.0.0
- boto3>=1.18.0
- botocore>=1.21.0
- jinja>=3.1.4

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.