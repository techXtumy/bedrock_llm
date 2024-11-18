import asyncio
import logging
import time
from typing import Any, Dict, List

from .optimized import (BatchConfig, BatchNode, CachedNode, FilterNode,
                        OptimizedPipeline, ParallelNode)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example node functions
async def embed_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Simulate batch embedding generation"""
    # In real application, this would call an embedding model
    await asyncio.sleep(0.5)  # Simulate API call
    return [
        {"text": text, "embedding": [0.1, 0.2, 0.3]}
        for text in texts
    ]


def preprocess_text(text: str) -> str:
    """CPU-intensive text preprocessing"""
    # Simulate CPU-intensive work
    time.sleep(0.1)
    return text.lower().strip()


def is_valid_text(text: str) -> bool:
    """Filter out empty or too short texts"""
    return len(text.strip()) > 5


async def create_optimized_pipeline() -> OptimizedPipeline:
    """Create an optimized pipeline for text processing"""
    pipeline = OptimizedPipeline("optimized_text_processing")

    # Create nodes with different optimizations
    preprocess_node = ParallelNode(
        "preprocess",
        func=preprocess_text,
        is_async=False,
        max_workers=4
    )

    filter_node = FilterNode(
        "filter",
        condition=is_valid_text
    )

    embed_node = BatchNode(
        "embed",
        func=embed_batch,
        config=BatchConfig(
            batch_size=16,
            max_wait_time=0.5,
            min_batch_size=4
        )
    )

    cache_node = CachedNode(
        "cache_results",
        func=lambda x: {"processed": x},
        cache_size=1000
    )

    # Add nodes to pipeline
    pipeline.add_node(preprocess_node)
    pipeline.add_node(filter_node)
    pipeline.add_node(embed_node)
    pipeline.add_node(cache_node)

    # Connect nodes in correct order
    preprocess_node.connect(filter_node)
    filter_node.connect(embed_node)
    embed_node.connect(cache_node)

    # Set start node
    pipeline.set_start_node(preprocess_node)

    # Enable parallel execution
    pipeline.enable_parallel_execution(True)

    return pipeline


async def run_optimized_example():
    """Run the optimized pipeline with example data"""
    pipeline = await create_optimized_pipeline()

    # Example input data
    texts = [
        "Hello World! This is an example.",
        "  Another example text  ",
        "hi",  # This should be filtered out
        "Processing with optimized pipeline",
        "Test text for processing"
    ]

    logger.info("Starting pipeline execution...")
    start_time = time.time()

    try:
        # Process each text and handle None results from filtering
        results = []
        for text in texts:
            result = await pipeline.execute(text)
            if result.get('filter') is not None:  # Text passed the filter
                results.append(result)

        logger.info("Pipeline execution completed!")
        logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
        logger.info(f"Number of processed texts: {len(results)}")
        logger.info(f"Results: {results}")
        logger.info(f"Pipeline status: {pipeline.get_pipeline_status()}")

    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_optimized_example())
