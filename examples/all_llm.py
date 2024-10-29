import sys
import os
import asyncio

# Add for print console with color
from termcolor import cprint

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bedrock_llm.client import LLMClient, ModelName
from src.bedrock_llm.utils.prompt import llama_format, mistral_format, titan_format
from src.bedrock_llm.config.model import ModelConfig
from src.bedrock_llm.config.base import RetryConfig


async def main():
    
    # Prompt format
    prompt = "Bạn là ai?."
    system = "You are a helpful AI. Answer only in Vietnamese"
    llama_prompt = llama_format(prompt, system)
    mistral_prompt = mistral_format(prompt, system)
    titan_prompt = titan_format(prompt, system)
    config = ModelConfig(
        temperature=0,
        max_tokens=512,
        top_p=1,
        top_k=70
    )
    retry_config = RetryConfig(
        max_retries=3,
        retry_delay=0.5
    )
    
    # Using Llama model
    for model in [
        ModelName.LLAMA_3_2_1B, 
        ModelName.LLAMA_3_2_3B, 
        ModelName.LLAMA_3_2_11B, 
        ModelName.LLAMA_3_2_90B
    ]:
        llama_client = LLMClient(
            region_name="us-west-2",
            model_name=model,
            retry_config=retry_config
        )
        print("Model: ", model)
        async for message, stop_reason in llama_client.generate(
            prompt=llama_prompt,
            config=config
        ):
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}", color="red")
                break
            cprint(message, color="yellow", end="", flush=True)
    
    
    # Using Titan model
    for model in [
        ModelName.TITAN_LITE, 
        ModelName.TITAN_EXPRESS, 
        ModelName.TITAN_PREMIER
    ]:
        titan_client = LLMClient(
            region_name="us-east-1",
            model_name=model,
            retry_config=retry_config
        )
        print("Model: ", model)
        async for message, stop_reason in titan_client.generate(
            prompt=titan_prompt,
            config=config
        ):
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}", color="red")
                break
            cprint(message, color="cyan", end="", flush=True)
    
    
    # Using Claude model
    for model in [
        ModelName.CLAUDE_3_5_HAIKU, 
        ModelName.CLAUDE_3_5_SONNET
    ]:
        claude_client = LLMClient(
            region_name="us-west-2",
            model_name=model,
            retry_config=retry_config
        )
        print("Model: ", model)
        async for message, stop_reason in claude_client.generate(
            prompt=prompt,
            config=config,
            system=system
        ):
            if isinstance(message, str):
                cprint(message, color="green", end="", flush=True)
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}", color="red")
                break
        
        
    # Using Jamba model
    for model in [ModelName.JAMBA_1_5_MINI, ModelName.JAMBA_1_5_LARGE]:
        jamba_client = LLMClient(
            region_name="us-east-1",
            model_name=model,
            retry_config=retry_config
        )
        print("Model: ", model)
        async for message, stop_reason in jamba_client.generate(
            prompt=prompt,
            config=config
        ):
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}", color="red")
                break
            if message is not None:
                cprint(message, color="grey", end="", flush=True)
        
        
    # Using Mistral 7B Instruct model
    mistral_client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.MISTRAL_7B,
        retry_config=retry_config
    )
    print("Model: ", ModelName.MISTRAL_7B)
    async for message, stop_reason in mistral_client.generate(
        prompt=mistral_prompt,
        config=config
    ):
        if stop_reason:
            cprint(f"\nGeneration stopped: {stop_reason}", color="red")
            break
        cprint(message, color="magenta", end="", flush=True)
        
        
    # Using Mistral Large V2 model
    mistral_client = LLMClient(
        region_name="us-west-2",
        model_name=ModelName.MISTRAL_LARGE_2,
        retry_config=retry_config
    )
    print("Model: ", ModelName.MISTRAL_LARGE_2)
    async for message, stop_reason in mistral_client.generate(
        prompt=prompt,
        config=config,
        system=system
    ):
        if stop_reason:
            cprint(f"\nGeneration stopped: {stop_reason}", color="red")
            break
        cprint(message, color="blue", end="", flush=True)
    
    
if __name__ == "__main__":
    asyncio.run(main())