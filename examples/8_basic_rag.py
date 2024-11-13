import boto3
from termcolor import cprint

from bedrock_llm import ModelName, MessageBlock, LLMClient, ModelConfig
from bedrock_llm.types.enums import StopReason

# Define the system prompt
system = "You are a RAG assistant. You have the user query and the search results from knowledge base. Base on the search results, please answer the user query only use the informations from it. If you do not know the answer, please said 'I do not know the answer to this question'"

# Define the runtime for knowledge base
runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

# Create a LLM client
client = LLMClient(
    region_name="us-east-1",
    model_name=ModelName.JAMBA_1_5_LARGE
)

# Create a configuration for inference parameters
config = ModelConfig(
    temperature=0.2,
    top_p=0.9,
    max_tokens=1024
)

# Define the search knowledge base
async def knowledge_base_retrieve(query: str):
    kwargs = {
        "knowledgeBaseId": "VSR83TL8CR",    # Insert your knowledge base ID
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {
                "numberOfResults": 25
            }
        },
        "retrievalQuery": {
            "text": query
        }
    }
    
    result = await asyncio.run(None, lambda: runtime.retrieve(**kwargs))
    return str(result["retrievalResults"])  # Make sure the return knowledge base is in string


async def main():
    
    # Get the user input
    user = input("User: ")
    
    cprint("Getting data from knowledge base ...", "cyan")
    # Query the user question into the knowledge base
    result = await knowledge_base_retrieve(user)
    
    cprint(result, "yellow") # In production need to parse and analyising this more.
    # Combine the user question with the result from the knowledge base
    prompt = MessageBlock(role="user", content=user+"\n\n"+result)
    
    cprint("Finallizing answer ...", "cyan")
    # Invoke the model and get results
    async for token, stop_reason, response in client.generate_async(
        config=config,
        prompt=prompt,
        system=system
    ):
        # Print out the results
        if token:
            cprint(token, "green", end="", flush=True)
        
        if stop_reason == StopReason.END_TURN:
            cprint(f"\n{stop_reason}", "red")
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())