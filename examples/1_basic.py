import sys
import os

# Add for print console with color
from termcolor import cprint

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import LLMClient, ModelName
from src.bedrock_llm.utils.prompt import llama_format
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.config.base import RetryConfig

# Create a LLM client
client = LLMClient(
    region_name="us-east-1",
    model_name=ModelName.LLAMA_3_2_1B,
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

# Create a prompt
prompt = llama_format("What is the capital of Vietnam?")

# Invoke the model and get results
response, stop_reason = client.generate(prompt,config)

# Print out the results
cprint(response.content, "green")
cprint(stop_reason, "red")