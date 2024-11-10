import sys
import os

# Add for print console with color
from termcolor import cprint

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import LLMClient, ModelName, MessageBlock
from src.bedrock_llm.schema.tools import ToolMetadata, InputSchema, PropertyAttr
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.config.base import RetryConfig

# Create a LLM client
client = LLMClient(
    region_name="us-east-1",
    model_name=ModelName.LLAMA_3_2_3B,
    retry_config=RetryConfig(
        max_attempts=3
    )
)

# Create a configuration for inference parameters
config = ModelConfig(
    temperature=0.1,
    top_p=0.9,
    max_tokens=512
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

# Create a system prompt with a list of examples
prompt = [
    MessageBlock(role="user", content="What is the capital of France?"),
    MessageBlock(role="assistant", content="The capital of France is Paris."),
    MessageBlock(role="user", content="What is the capital of Germany?"),
    MessageBlock(role="assistant", content="The capital of Germany is Berlin."),
    MessageBlock(role="user", content="What is your name and what is the weather in the capital of Italy?")
]
system = "You are a helpful assistant. You have access to realtime information. You can use tools to get the real time data weather of a city."

# Invoke the model and get results
response, stop_reason = client.generate(
    config=config, 
    prompt=prompt,
    tools=[get_weather_tool]
)

# Print out the results
cprint(f"Calling function: {response.content}", "cyan")
cprint(stop_reason, "red")

prompt.append(MessageBlock(role="assistant", content=response.content))
prompt.append(MessageBlock(role="tool", content="20*C"))

response, stop_reason = client.generate(
    config=config, 
    prompt=prompt,
)

# Print out the results
cprint(response.content, "green")
cprint(stop_reason, "red")