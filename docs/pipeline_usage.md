# Pipeline Orchestration System Documentation

## Overview

The Pipeline Orchestration System is a flexible, high-performance framework for building complex LLM workflows. It supports both synchronous and asynchronous operations, reactive programming patterns, and provides comprehensive status tracking and error handling.

## Features

- 🔄 Flexible node connections (one-to-many, many-to-one)
- ⚡ Async/sync function support
- 📊 Built-in status tracking and monitoring
- 🔍 Comprehensive error handling
- 🚀 Reactive programming patterns using RxPY
- 💪 High-performance execution with concurrent processing

## Installation

The pipeline system is included in the Bedrock LLM library. Ensure you have the required dependencies:

```bash
pip install rx==3.2.0  # Required for reactive programming
```

## Basic Usage

### 1. Creating a Simple Pipeline

```python
from bedrock_llm.pipeline import Pipeline, PipelineNode

# Create a pipeline
pipeline = Pipeline("my_pipeline")

# Define node functions
async def process_text(text: str) -> str:
    return text.lower()

def tokenize(text: str) -> list:
    return text.split()

# Create nodes
node1 = PipelineNode("preprocessor", process_text, is_async=True)
node2 = PipelineNode("tokenizer", tokenize, is_async=False)

# Add nodes to pipeline
pipeline.add_node(node1)
pipeline.add_node(node2)

# Connect nodes
node1.connect(node2)

# Set start node
pipeline.set_start_node(node1)

# Execute pipeline
async def run():
    results = await pipeline.execute("Hello World!")
    print(results)

# Run the pipeline
import asyncio
asyncio.run(run())
```

### 2. Advanced Pipeline Configuration

```python
from bedrock_llm.pipeline import Pipeline, PipelineNode
from typing import Dict, Any

# Define complex node functions
async def embed_text(tokens: list) -> Dict[str, Any]:
    # Simulate embedding generation
    return {
        "tokens": tokens,
        "embeddings": [0.1, 0.2, 0.3]
    }

async def classify_text(data: Dict[str, Any]) -> Dict[str, Any]:
    # Simulate classification
    return {
        **data,
        "classification": "positive"
    }

def aggregate_results(data: Dict[str, Any]) -> Dict[str, Any]:
    # Aggregate and format results
    return {
        "final_result": {
            "classification": data["classification"],
            "confidence": 0.95,
            "embeddings": data["embeddings"]
        }
    }

# Create pipeline with multiple paths
pipeline = Pipeline("advanced_pipeline")

# Create nodes
preprocess = PipelineNode("preprocess", lambda x: x.lower(), is_async=False)
tokenize = PipelineNode("tokenize", str.split, is_async=False)
embed = PipelineNode("embed", embed_text, is_async=True)
classify = PipelineNode("classify", classify_text, is_async=True)
aggregate = PipelineNode("aggregate", aggregate_results, is_async=False)

# Add nodes
for node in [preprocess, tokenize, embed, classify, aggregate]:
    pipeline.add_node(node)

# Create complex connections
preprocess.connect(tokenize)
tokenize.connect(embed)
embed.connect(classify)
classify.connect(aggregate)

# Set start node
pipeline.set_start_node(preprocess)
```

### 3. Error Handling and Monitoring

```python
from bedrock_llm.pipeline import Pipeline, PipelineNode, NodeStatus
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define a function that might fail
async def risky_operation(data: Any) -> Any:
    if some_condition:
        raise ValueError("Operation failed")
    return processed_data

# Create node with error handling
node = PipelineNode("risky_node", risky_operation, is_async=True)

# Monitor pipeline status
async def monitor_pipeline(pipeline: Pipeline, input_data: Any):
    try:
        results = await pipeline.execute(input_data)
        status = pipeline.get_pipeline_status()
        
        # Check status of each node
        for node_name, node_status in status.items():
            if node_status == NodeStatus.FAILED:
                logging.error(f"Node {node_name} failed")
            elif node_status == NodeStatus.COMPLETED:
                logging.info(f"Node {node_name} completed successfully")
                
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
```

## Best Practices

### 1. Node Design

- Keep node functions focused and single-purpose
- Use appropriate async/sync designation based on operation type
- Include proper error handling within node functions
- Document input/output types clearly

```python
from typing import TypeVar, Generic

T = TypeVar('T')
U = TypeVar('U')

class TypedNode(PipelineNode, Generic[T, U]):
    """Type-safe node with clear input/output types"""
    def __init__(
        self,
        name: str,
        func: Callable[[T], U],
        is_async: bool = False
    ):
        super().__init__(name, func, is_async)
        self.input_type = T
        self.output_type = U
```

### 2. Pipeline Organization

- Group related nodes into logical sub-pipelines
- Use clear naming conventions
- Implement proper logging and monitoring
- Consider using dependency injection for configuration

```python
class SubPipeline:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.pipeline = Pipeline(name)
        self.config = config
        self._setup_nodes()
        
    def _setup_nodes(self):
        # Create and connect nodes specific to this sub-pipeline
        pass
        
    async def execute(self, data: Any) -> Any:
        return await self.pipeline.execute(data)
```

### 3. Performance Optimization

- Use batching for similar operations
- Implement caching for expensive operations
- Consider using thread pools for CPU-bound tasks
- Use connection pooling for database operations

```python
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

class OptimizedNode(PipelineNode):
    def __init__(self, name: str, func: Callable, is_async: bool = False):
        super().__init__(name, func, is_async)
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    @lru_cache(maxsize=1000)
    async def process_with_cache(self, data: Any) -> Any:
        if self.is_async:
            return await self.func(data)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.func,
                data
            )
```

## Advanced Topics

### 1. Custom Node Types

```python
class BatchNode(PipelineNode):
    """Node that processes data in batches"""
    def __init__(self, name: str, func: Callable, batch_size: int = 32):
        super().__init__(name, func, is_async=True)
        self.batch_size = batch_size
        self._batch = []
        
    async def process(self, data: Any) -> Any:
        self._batch.append(data)
        if len(self._batch) >= self.batch_size:
            result = await self._process_batch()
            self._batch = []
            return result
        return None
        
    async def _process_batch(self) -> Any:
        return await self.func(self._batch)
```

### 2. Pipeline Patterns

#### Fan-out Pattern

```python
# Create multiple parallel processing paths
source_node.connect(process_node1)
source_node.connect(process_node2)
source_node.connect(process_node3)
```

#### Aggregation Pattern

```python
# Combine results from multiple nodes
process_node1.connect(aggregator_node)
process_node2.connect(aggregator_node)
process_node3.connect(aggregator_node)
```

#### Filter Pattern

```python
class FilterNode(PipelineNode):
    def __init__(self, name: str, condition: Callable[[Any], bool]):
        super().__init__(name, self._filter, is_async=False)
        self.condition = condition
        
    def _filter(self, data: Any) -> Optional[Any]:
        return data if self.condition(data) else None
```

## Performance Tips

1. **Batch Processing**
   - Group similar operations into batches
   - Use vectorized operations when possible
   - Implement proper batch size tuning

2. **Caching Strategy**
   - Use LRU cache for frequently accessed data
   - Implement Redis for distributed caching
   - Consider TTL for cached items

3. **Resource Management**
   - Use connection pooling for databases
   - Implement proper cleanup in node destructors
   - Monitor memory usage in long-running pipelines

4. **Monitoring and Debugging**
   - Implement comprehensive logging
   - Use performance profiling
   - Monitor system resources

## Common Pitfalls to Avoid

1. **Anti-patterns**
   - Avoid creating too many small nodes
   - Don't mix sync/async operations unnecessarily
   - Prevent circular dependencies

2. **Resource Leaks**
   - Always clean up resources
   - Use context managers for managed resources
   - Implement proper error handling

3. **Performance Issues**
   - Avoid unnecessary serialization/deserialization
   - Don't create too many thread pools
   - Watch out for memory leaks in long-running pipelines

## Example Use Cases

### 1. Text Processing Pipeline

```python
# Text processing pipeline example
async def create_text_pipeline():
    pipeline = Pipeline("text_processing")
    
    # Create nodes
    clean = PipelineNode("clean", lambda x: x.lower().strip(), is_async=False)
    tokenize = PipelineNode("tokenize", str.split, is_async=False)
    embed = PipelineNode("embed", get_embeddings, is_async=True)
    
    # Add and connect nodes
    pipeline.add_node(clean)
    pipeline.add_node(tokenize)
    pipeline.add_node(embed)
    
    clean.connect(tokenize)
    tokenize.connect(embed)
    
    pipeline.set_start_node(clean)
    return pipeline
```

### 2. LLM Processing Pipeline

```python
# LLM processing pipeline example
async def create_llm_pipeline(model_config: Dict[str, Any]):
    pipeline = Pipeline("llm_processing")
    
    # Create specialized nodes
    preprocess = PipelineNode("preprocess", preprocess_text, is_async=True)
    generate = PipelineNode("generate", lambda x: model.generate(x), is_async=True)
    postprocess = PipelineNode("postprocess", format_output, is_async=False)
    
    # Add and connect nodes
    pipeline.add_node(preprocess)
    pipeline.add_node(generate)
    pipeline.add_node(postprocess)
    
    preprocess.connect(generate)
    generate.connect(postprocess)
    
    pipeline.set_start_node(preprocess)
    return pipeline
```

## Contributing

We welcome contributions to improve the pipeline system! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your changes
4. Include tests for new functionality
5. Update documentation as needed

## Support

For issues and feature requests, please use the GitHub issue tracker.
