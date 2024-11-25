import asyncio
from typing import Any, Dict

from .core import Pipeline, PipelineNode


async def text_preprocessing(text: str) -> str:
    """Example async node function for text preprocessing"""
    # Simulate some async processing
    await asyncio.sleep(1)
    return text.lower().strip()


def tokenization(text: str) -> list:
    """Example sync node function for tokenization"""
    return text.split()


async def llm_processing(tokens: list) -> Dict[str, Any]:
    """Example async node function for LLM processing"""
    # Simulate LLM processing
    await asyncio.sleep(2)
    return {
        "tokens": tokens,
        "embedding": [0.1, 0.2, 0.3],  # Example embedding
        "processed": True
    }


async def create_example_pipeline() -> Pipeline:
    # Create pipeline
    pipeline = Pipeline("text_processing")

    # Create nodes
    preprocess_node = PipelineNode(
        name="preprocess",
        func=text_preprocessing,
        is_async=True
    )

    tokenize_node = PipelineNode(
        name="tokenize",
        func=tokenization,
        is_async=False
    )

    llm_node = PipelineNode(
        name="llm_process",
        func=llm_processing,
        is_async=True
    )

    # Add nodes to pipeline
    pipeline.add_node(preprocess_node)
    pipeline.add_node(tokenize_node)
    pipeline.add_node(llm_node)

    # Set up node connections
    preprocess_node.connect(tokenize_node)
    tokenize_node.connect(llm_node)

    # Set start node
    pipeline.set_start_node(preprocess_node)

    return pipeline


async def run_example():
    pipeline = await create_example_pipeline()
    input_text = "  Hello World! This is an Example.  "

    try:
        results = await pipeline.execute(input_text)
        print("Pipeline Results:", results)
        print("Pipeline Status:", pipeline.get_pipeline_status())
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_example())
