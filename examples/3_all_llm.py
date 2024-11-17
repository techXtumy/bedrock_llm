import asyncio

# Add for print console with color
from termcolor import cprint

from bedrock_llm import AsyncClient, ModelConfig, ModelName, RetryConfig
from bedrock_llm.schema import MessageBlock


async def main():
    # Prompt format
    prompt = MessageBlock(role="user", content="Who are you and what can you do?")
    system = "You are a helpful AI. Answer only in Vietnamese"
    config = ModelConfig(temperature=0, max_tokens=512, top_p=1, top_k=70)
    retry_config = RetryConfig(max_retries=3, retry_delay=0.5)

    # Using Llama model
    for model in [
        ModelName.LLAMA_3_2_1B,
        ModelName.LLAMA_3_2_3B,
        ModelName.LLAMA_3_2_11B,
        ModelName.LLAMA_3_2_90B,
    ]:
        llama_client = AsyncClient(
            region_name="us-east-1", model_name=model, retry_config=retry_config
        )
        print("Model: ", model)
        async for token, stop_reason,_ in llama_client.generate_async(
            config=config, prompt=prompt, system=system
        ):
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}", color="red")
                break
            cprint(token, color="yellow", end="", flush=True)

    # Using Titan model
    for model in [
        ModelName.TITAN_LITE,
        ModelName.TITAN_EXPRESS,
        ModelName.TITAN_PREMIER,
    ]:
        titan_client = AsyncClient(
            region_name="us-east-1", model_name=model, retry_config=retry_config, profile_name="bedrock"
        )
        print("Model: ", model)
        async for token, stop_reason, message in titan_client.generate_async(
            config=config, prompt=prompt, system=system
        ):
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}", color="red")
                break
            cprint(token, color="cyan", end="", flush=True)

    # Using Claude model
    for model in [
        ModelName.CLAUDE_3_HAIKU,
        ModelName.CLAUDE_3_5_HAIKU,
        ModelName.CLAUDE_3_5_SONNET,
    ]:
        claude_client = AsyncClient(
            region_name="us-east-1", model_name=model, retry_config=retry_config, profile_name="bedrock"
        )
        print("Model: ", model)
        async for token, stop_reason, message in claude_client.generate_async(
            config=config, prompt=prompt, system=system
        ):
            if token:
                cprint(token, color="green", end="", flush=True)
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}", color="red")
                break

    # # Using Jamba model
    for model in [ModelName.JAMBA_1_5_MINI, ModelName.JAMBA_1_5_LARGE]:
        jamba_client = AsyncClient(
            region_name="us-east-1", model_name=model, retry_config=retry_config, profile_name="bedrock"
        )
        print("Model: ", model)
        async for token, stop_reason, message in jamba_client.generate_async(
            config=config, prompt=prompt, system=system
        ):
            if stop_reason:
                cprint(f"\nGeneration stopped: {stop_reason}", color="red")
                break
            if token:
                cprint(token, color="grey", end="", flush=True)

    # Using Mistral 7B Instruct model
    mistral_client = AsyncClient(
        region_name="us-west-2",
        model_name=ModelName.MISTRAL_7B,
        retry_config=retry_config,
        profile_name="bedrock"
    )
    print("Model: ", ModelName.MISTRAL_7B)
    async for token, stop_reason, message in mistral_client.generate_async(
        config=config, prompt=prompt, system=system
    ):
        if stop_reason:
            cprint(f"\nGeneration stopped: {stop_reason}", color="red")
            break
        cprint(token, color="magenta", end="", flush=True)

    # Using Mistral Large V2 model
    mistral_client = AsyncClient(
        region_name="us-west-2",
        model_name=ModelName.MISTRAL_LARGE_2,
        retry_config=retry_config,
        profile_name="bedrock"
    )
    print("Model: ", ModelName.MISTRAL_LARGE_2)
    async for token, stop_reason, message in mistral_client.generate_async(
        config=config, prompt=prompt, system=system
    ):
        if stop_reason:
            cprint(f"\nGeneration stopped: {stop_reason}", color="red")
            break
        cprint(token, color="blue", end="", flush=True)
        
    
    # Close the connection
    await llama_client.close()
    await titan_client.close()
    await claude_client.close()
    await jamba_client.close()
    await mistral_client.close()

if __name__ == "__main__":
    asyncio.run(main())
