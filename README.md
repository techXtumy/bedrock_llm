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
    model_name=ModelName.TITAN_PREMIER
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

### Simple agent

```python
from bedrock_llm import Agent, ModelName, 
from bedrock_llm import MessageBlock, ToolMetadata, InputSchema, PropertyAttr

system = "You are a helpful assistant. You have access to realtime information. You can use tools to get the real time data weather of a city."

# Create a LLM client
agent = Agent(
    region_name="us-east-1",
    model_name=ModelName.CLAUDE_3_5_HAIKU
)

# Create tool definition
get_weather_tool = ToolMetadata(
    name="get_weather",
    description="Get the real time weather of a city",
    input_schema=InputSchema(
        type="object",
        properties={
            "location": PropertyAttr(
                type="string",
                description="The city to get the weather of"
            )
        },
        required=["location"]
    )
)

# Create user prompt
prompt = MessageBlock(role="user", content="What is the weather in New York and Toronto?")

@Agent.tool(get_weather_tool)
async def get_weather(location: str):
    # Mock function to get weather
    return f"tools_result: {location} is 20*C"


async def main():
    # Invoke the model and get results
    async for token, stop_reason, response, tool_result in agent.generate_and_action_async(
        prompt=prompt,
        system=system,
        tools=["get_weather"]
    ):
        # Print out the results
        if token:
            print(token, end="", flush=True)
        
        if tool_result:
            print(f"\n{tool_result}")
        
        if stop_reason == StopReason.TOOL_USE:
            for x in range(1, len(response.content)):
                print(f"\n{response.content[x].model_dump()}", end="", flush=True)
            print(f"\n{stop_reason}")
        elif stop_reason:
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