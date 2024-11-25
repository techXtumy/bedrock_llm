# Bedrock LLM Library Documentation

## Introduction

The Bedrock LLM library is a Python package that provides a high-level interface for interacting with various Large Language Models (LLMs) through Amazon Bedrock. This library simplifies the process of working with different LLM providers and offers consistent APIs for common operations like text generation, chat interactions, embeddings, and agent-based workflows.

## Overall Architecture

The library is organized into several key components:

1. **Client Layer**: Handles communication with Amazon Bedrock service
2. **Models**: Implements provider-specific model interfaces
3. **Configuration**: Manages model and runtime settings
4. **Schema**: Defines data structures for messages and tools
5. **Agent**: Implements autonomous agent capabilities
6. **Types**: Contains type definitions and enums
7. **Monitor**: Provides monitoring and observability features
8. **Pipeline**: Offers a powerful pipeline system for building efficient data processing workflows

## File Structure

```bash
src/bedrock_llm/
├── __init__.py           # Package initialization and version
├── agent.py             # Agent implementation for autonomous operations
├── aws_clients.py       # AWS client configuration and management
├── client/             # Client implementations
│   ├── __init__.py
│   ├── async_client.py  # Asynchronous client implementation
│   ├── base.py         # Base client abstract class
│   ├── embeddings.py   # Embeddings client implementation
│   ├── reranking.py    # Reranking client implementation
│   └── sync_client.py  # Synchronous client implementation
├── config/             # Configuration management
│   ├── __init__.py
│   ├── base.py        # Base configuration classes
│   └── model.py       # Model-specific configurations
├── models/            # Model implementations
│   ├── __init__.py
│   ├── ai21.py       # AI21 model implementation
│   ├── amazon.py     # Amazon Titan model implementation
│   ├── anthropic.py  # Anthropic Claude model implementation
│   ├── base.py       # Base model interface
│   ├── cohere.py     # Cohere model implementation
│   ├── embeddings.py # Embeddings model implementations
│   ├── meta.py       # Meta Llama model implementation
│   └── mistral.py    # Mistral model implementation
├── monitor/          # Monitoring and observability
├── pipeline/         # Pipeline components
│   ├── __init__.py
│   ├── core.py       # Core pipeline implementation
│   ├── optimized.py  # Optimized pipeline features
│   ├── examples.py   # Usage examples
│   └── optimized_example.py  # Optimized pipeline examples
├── schema/          # Data structures and validation
│   ├── __init__.py
│   ├── cache.py     # Caching implementations
│   ├── message.py   # Message type definitions
│   └── tools.py     # Tool schema definitions
└── types/          # Type definitions and enums
```

## Detailed File Analysis

### **init**.py

Contains package initialization code and exports main interfaces:

- Version information
- Public API exports
- Type hints
- Default configurations

### agent.py

Implements the autonomous agent system with following key components:

```python
class ToolState(Enum):
    """Tool execution states for tracking progress."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"

class Agent(AsyncClient):
    """Main agent implementation."""
    
    def tool(cls, metadata: ToolMetadata):
        """Decorator for registering tools."""
        
    async def generate_and_action_async(self, ...):
        """Core method for agent reasoning and action."""
```

Key features:

- Tool registration and management
- Memory management system
- Error handling
- State tracking
- Async execution support

### aws_clients.py

Manages AWS client configuration and session handling:

- Credential management
- Region configuration
- Session pooling
- Service endpoint configuration

### client/base.py

Core client implementation with essential features:

```python
class BaseClient(ABC):
    """Abstract base class for all clients."""
    
    async def _handle_retry_logic_stream(self, operation, *args, **kwargs):
        """Implements streaming retry logic."""
        
    def _process_prompt(self, messages: List[MessageBlock], **kwargs) -> str:
        """Processes messages into provider format."""
```

Features:

- Retry logic implementation
- Error handling
- Request/response processing
- Memory management
- Model implementation selection

### client/async_client.py

Asynchronous client implementation:

- Async API calls
- Streaming support
- Concurrent operations
- Resource management

### client/sync_client.py

Synchronous client implementation:

- Blocking API calls
- Batch processing
- Sequential operations

### config/base.py

Base configuration classes:

```python
class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_attempts: int
    base_delay: float
    max_delay: float

class ModelConfig(BaseModel):
    """Model-specific configuration."""
    provider: str
    model_id: str
    parameters: Dict[str, Any]
```

Features:

- Type validation
- Default values
- Configuration inheritance
- Environment variable support

### models/base.py

Defines the base interface for model implementations:

```python
class BaseModelImplementation(ABC):
    @abstractmethod
    def prepare_request(self, messages: List[MessageBlock], **kwargs) -> Any:
        """Convert messages to provider format."""
        
    @abstractmethod
    def parse_response(self, response: Any) -> Tuple[MessageBlock, StopReason]:
        """Parse provider response to standard format."""
```

Key aspects:

- Request preparation
- Response parsing
- Stream handling
- Provider-specific logic

### schema/message.py

Message type definitions and validation:

```python
class MessageBlock(BaseModel):
    """Base message structure."""
    role: str
    content: Union[str, List[Union[TextBlock, ImageBlock, ToolBlock]]]
    metadata: Optional[Dict[str, Any]]

class ToolBlock(BaseModel):
    """Tool interaction representation."""
    name: str
    parameters: Dict[str, Any]
    result: Optional[str]
```

Features:

- Type validation
- Serialization support
- Nested structure support
- Custom validation rules

## Pipeline Components

The library includes a powerful pipeline system for building efficient data processing workflows:

### Basic Pipeline Usage

```python
from bedrock_llm.pipeline import Pipeline, PipelineNode

# Create a simple pipeline
pipeline = Pipeline("text-processing")

# Add processing nodes
node1 = PipelineNode("tokenizer", tokenize_func)
node2 = PipelineNode("embedding", embed_func)

# Connect nodes
pipeline.add_node(node1)
pipeline.add_node(node2)
node1.connect(node2)

# Execute pipeline
result = await pipeline.execute(input_text)
```

### Optimized Pipeline Features

1. **Batch Processing**

    ```python
    from bedrock_llm.pipeline import BatchNode, BatchConfig

    batch_node = BatchNode(
        "batch-embeddings",
        embed_batch_func,
        BatchConfig(batch_size=32, max_wait_time=1.0)
    )
    ```

2. **Cached Processing**

    ```python
    from bedrock_llm.pipeline import CachedNode

    cached_node = CachedNode(
        "cached-embeddings",
        embed_func,
        cache_size=1000
    )
    ```

3. **Parallel Processing**

    ```python
    from bedrock_llm.pipeline import ParallelNode

    parallel_node = ParallelNode(
        "parallel-process",
        process_func,
        max_workers=4
    )
    ```

4. **Type-Safe Nodes**

    ```python
    from bedrock_llm.pipeline import TypedNode

    typed_node = TypedNode[str, float](
        "typed-process",
        process_func
    )
    ```

5. **Data Filtering**

    ```python
    from bedrock_llm.pipeline import FilterNode

    filter_node = FilterNode(
        "filter-data",
        lambda x: x > 0.5
    )
    ```

### Pipeline Optimization

The pipeline system includes several optimization features:

1. **Batch Processing**
   - Automatically batches inputs for improved throughput
   - Configurable batch sizes and timing
   - Handles partial batches efficiently

2. **Caching**
   - In-memory caching of results
   - LRU cache implementation
   - Configurable cache sizes

3. **Parallel Execution**
   - Thread pool-based parallel processing
   - Configurable worker count
   - Automatic resource management

4. **Type Safety**
   - Generic type support
   - Runtime type checking
   - Type-safe node connections

5. **Event-Driven Architecture**
   - Reactive programming model
   - Non-blocking execution
   - Event-based data flow

## Core Components

### Client Layer (client/)

The client layer is the foundation of the Bedrock LLM library, providing comprehensive functionality for interacting with Amazon Bedrock services. This layer is designed with modularity, type safety, and performance in mind.

#### BaseClient (client/base.py)

The `BaseClient` is an abstract base class that implements core functionality shared across all client types:

##### Authentication and Session Management

- Secure AWS credential handling
- Automatic session refresh and token management
- Region-specific endpoint configuration
- Support for custom endpoint URLs
- IAM role assumption capabilities

##### Retry Logic and Error Handling

- Configurable retry strategy with exponential backoff
- Customizable retry conditions and exceptions
- Circuit breaker pattern implementation
- Detailed error logging and diagnostics
- Automatic request ID tracking

##### Model Implementation Management

- Dynamic model loading based on configuration
- Version compatibility checking
- Model-specific parameter validation
- Automatic format conversion
- Response post-processing

##### Memory and Context Management

- Conversation history tracking
- Context window optimization
- Automatic memory pruning
- Token counting and limiting
- Memory persistence options

#### AsyncClient Implementation

The AsyncClient is optimized for high-throughput scenarios and provides:

##### Asynchronous Operations

- Non-blocking API calls
- Concurrent request handling
- Connection pooling
- Request queuing and prioritization
- Backpressure handling

##### Streaming Support

- Chunked response processing
- Stream backpressure handling
- Automatic reconnection
- Stream timeout management
- Partial response handling

##### Performance Features

- Request batching
- Connection reuse
- Resource cleanup
- Memory optimization
- Monitoring hooks

#### SyncClient Implementation

The SyncClient offers a simpler interface for basic use cases:

##### Synchronous Operations

- Direct request-response pattern
- Automatic retries
- Timeout handling
- Response validation
- Error propagation

##### Batch Processing

- Efficient batch requests
- Result aggregation
- Error handling per item
- Progress tracking
- Resource management

#### EmbeddingsClient Specialization

Dedicated client for vector embedding operations:

##### Embedding Features

- Multi-model support
- Batch processing optimization
- Dimension management
- Normalization options
- Caching support

##### Performance Optimization

- Request batching
- Memory efficiency
- Parallel processing
- Resource pooling
- Cache management

#### Best Practices for Client Usage

##### Client Selection

- Use AsyncClient for:
  - High-throughput applications
  - Streaming requirements
  - Concurrent operations
  - Resource-intensive tasks
- Use SyncClient for:
  - Simple applications
  - Sequential processing
  - Direct integrations
  - Development and testing

##### Configuration Guidelines

- Set appropriate timeouts
- Configure retry policies
- Implement error handling
- Monitor memory usage
- Enable logging when needed

##### Resource Management

- Implement proper cleanup
- Handle connection pooling
- Manage batch sizes
- Monitor performance
- Implement circuit breakers

### Models (models/)

The models package contains implementations for different LLM providers supported by Amazon Bedrock:

#### BaseModelImplementation (models/base.py)

Abstract base class defining the interface for all model implementations:

- Request preparation
- Response parsing
- Stream handling
- Provider-specific configurations

Supported Providers:

- Anthropic (Claude models)
- AI21
- Amazon (Titan models)
- Cohere
- Meta (Llama models)
- Mistral

Each provider implementation handles:

- Message formatting
- Token counting
- Response streaming
- Model-specific parameters

### Configuration (config/)

The configuration system provides a type-safe way to manage:

#### RetryConfig

Defines retry behavior for API calls:

- Max attempts
- Backoff factor
- Retry conditions

#### ModelConfig

Handles model-specific settings:

- Model parameters (temperature, top_p, etc.)
- Provider-specific configurations
- Runtime settings

### Schema (schema/)

The schema package defines the core data structures used throughout the library:

#### Message Types (message.py)

- `MessageBlock`: Base message structure
- `TextBlock`: Text content
- `ImageBlock`: Image content support
- `ToolUseBlock`: Tool usage records
- `ToolResultBlock`: Tool execution results
- `SystemBlock`: System message support

Key features:

- Pydantic models for validation
- JSON serialization support
- Type checking
- Extensible design

### Agent System (agent.py)

The Agent System is a sophisticated component that provides autonomous interaction capabilities, designed to seamlessly integrate with various LLM models while maintaining robust error handling and type safety.

#### Core Architecture

##### Component Design

The Agent system is built on a modular architecture:

```python
class Agent(AsyncClient):
    def __init__(self, tools: List[Tool] = None, **kwargs):
        super().__init__(**kwargs)
        self.tools = ToolRegistry(tools)
        self.memory = AgentMemory()
        self.state_manager = StateManager()
```

##### Key Components

1. **Tool Registry**: Manages tool registration and discovery
2. **Agent Memory**: Handles conversation and state persistence
3. **State Manager**: Coordinates tool states and transitions
4. **Response Processor**: Ensures type-safe response handling

#### Features and Implementation

##### Tool Management System

The tool management system provides a flexible framework for integrating custom tools:

```python
@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance between two points on Earth"""
    # Implementation
    return distance
```

###### Tool Registration

- Declarative tool registration using decorators
- Automatic parameter validation and type checking
- Documentation generation from docstrings
- Runtime tool discovery and validation

###### State Management

- Tool-specific state tracking
- Persistent state across invocations
- State validation and cleanup
- Error state handling

##### Memory Architecture

###### Conversation Management

- Efficient history tracking
- Context window optimization
- Automatic memory pruning
- Importance-based retention

###### State Persistence

- Durable state storage
- State recovery mechanisms
- Transaction support
- Conflict resolution

##### Error Handling Framework

###### Error Categories

1. Tool Execution Errors
   - Parameter validation failures
   - Runtime execution errors
   - Resource access issues
   - Timeout handling

2. State Management Errors
   - State transition failures
   - Persistence errors
   - Recovery failures
   - Consistency issues

3. Memory Management Errors
   - Context overflow
   - Storage failures
   - Retrieval errors
   - Pruning issues

###### Error Recovery Strategies

- Automatic retry with backoff
- Graceful degradation
- State rollback
- Alternative tool selection

#### Type Safety System

##### Response Types

```python
class AgentResponse(TypedDict):
    result: Any
    tool_calls: List[ToolCall]
    memory_updates: List[MemoryUpdate]
    state_changes: List[StateChange]
```

##### Validation Layer

- Runtime type checking
- Schema validation
- Contract enforcement
- Compatibility verification

##### Tool Implementation

1. **Design Guidelines**
   - Single responsibility principle
   - Clear input/output contracts
   - Comprehensive error handling
   - Performance optimization

2. **Testing Strategy**
   - Unit test coverage
   - Integration testing
   - Performance benchmarks
   - Error scenario testing

##### Memory Management

1. **Optimization Techniques**
   - Efficient storage formats
   - Intelligent pruning
   - Caching strategies
   - Compression methods

2. **Consistency Guidelines**
   - Transaction boundaries
   - State validation
   - Recovery procedures
   - Cleanup protocols

##### Error Handling

1. **Implementation Approach**
   - Specific error types
   - Detailed error messages
   - Recovery procedures
   - Logging and monitoring

2. **Recovery Strategy**
   - Graceful degradation
   - Alternative paths
   - State preservation
   - User communication

#### Performance Considerations

##### Optimization Strategies

1. **Memory Efficiency**
   - Optimized data structures
   - Memory pooling
   - Resource cleanup
   - Cache management

2. **Processing Speed**
   - Parallel execution
   - Batch processing
   - Response streaming
   - Load balancing

##### Monitoring and Metrics

1. **Key Metrics**
   - Response times
   - Memory usage
   - Error rates
   - Tool performance

2. **Observability**
   - Detailed logging
   - Performance tracing
   - Error tracking
   - Usage analytics

## Extending the Library

### Adding New Model Providers

To add support for a new model provider:

1. Create a new file in `models/` directory
2. Implement `BaseModelImplementation`:

   ```python
   class NewProviderImplementation(BaseModelImplementation):
       def prepare_request(self, messages, **kwargs):
           # Convert messages to provider format
           
       def parse_response(self, response):
           # Parse provider response to MessageBlock
   ```

3. Add provider configuration in `config/model.py`
4. Register the implementation in model factory

### Adding New Tools

Tools can be added to agents using the decorator pattern:

```python
@agent.tool(
    name="new_tool",
    description="Tool description",
    parameters={...}
)
async def new_tool(param1: str, param2: int) -> str:
    # Tool implementation
    return result
```

### Custom Message Types

The schema system can be extended with new message types:

1. Create new Pydantic model inheriting from appropriate base
2. Implement required validation and serialization
3. Register with message handler system

## Monitoring and Observability

The library includes built-in monitoring capabilities:

- Request/response logging
- Performance metrics
- Error tracking
- Cost monitoring
- Usage statistics

## Best Practices

### Memory Management Options

- Use appropriate memory settings for your use case
- Implement cleanup for long-running applications
- Monitor memory usage in high-throughput scenarios

### Error Hadeling Options

- Implement appropriate retry strategies
- Handle rate limits gracefully
- Log and monitor errors
- Use type checking for tool inputs/outputs

### Performance Optimization Options

- Use async operations for concurrent requests
- Implement caching where appropriate
- Batch requests when possible
- Monitor token usage

## Usage Examples

### Basic Text Generation

```python
from bedrock_llm import SyncClient
from bedrock_llm.schema.message import MessageBlock

# Initialize client
client = SyncClient()

# Create a simple prompt

# Generate response
response = client.generate(messages)
print(response.content)
```

### Chat with Memory

```python
from bedrock_llm import AsyncClient
from bedrock_llm.schema.message import MessageBlock
from bedrock_llm.config import ModelConfig

# Initialize with custom configuration
config = ModelConfig(
    provider="anthropic",
    model_id="claude-3-sonnet",
    parameters={
        "temperature": 0.7,
        "max_tokens": 1000
    }
)
client = AsyncClient(model_config=config)

# Multi-turn conversation
messages = [
    MessageBlock(role="system", content="You are a helpful assistant."),
    MessageBlock(role="user", content="What is Python?"),
    MessageBlock(role="assistant", content="Python is a high-level programming language..."),
    MessageBlock(role="user", content="Show me a simple example.")
]

response = await client.generate_async(messages)
```

### Using Tools

```python
from bedrock_llm import Agent
from bedrock_llm.schema.tools import ToolMetadata

agent = Agent()

@agent.tool(
    metadata=ToolMetadata(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "location": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }
    )
)
async def get_weather(location: str, units: str = "celsius") -> str:
    # Implement weather lookup logic
    return f"Weather in {location}: 22°{units[0].upper()}"

response = await agent.generate_and_action_async(
    "What's the weather in London?",
    available_tools=["get_weather"]
)
```

### Image Understanding

```python
from bedrock_llm import SyncClient

client = SyncClient()

messages = [
    MessageBlock(
        role="user",
        content=[
            TextBlock(text="What's in this image?"),
            ImageBlock(
                url="https://example.com/image.jpg",
                detail_level="auto"
            )
        ]
    )
]

response = client.generate(messages)
```

## API Reference

### Client Classes

#### SyncClient

```python
class SyncClient(BaseClient):
    def generate(
        self,
        messages: List[MessageBlock],
        **kwargs
    ) -> MessageBlock:
        """Generate a response synchronously."""
        
    def stream(
        self,
        messages: List[MessageBlock],
        **kwargs
    ) -> Iterator[MessageBlock]:
        """Stream responses synchronously."""
```

#### AsyncClient

```python
class AsyncClient(BaseClient):
    async def generate_async(
        self,
        messages: List[MessageBlock],
        **kwargs
    ) -> MessageBlock:
        """Generate a response asynchronously."""
        
    async def stream_async(
        self,
        messages: List[MessageBlock],
        **kwargs
    ) -> AsyncIterator[MessageBlock]:
        """Stream responses asynchronously."""
```

### Configuration Classes

#### ModelConfig

```python
class ModelConfig(BaseModel):
    provider: str
    model_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    streaming: bool = False
    timeout: Optional[float] = None
```

#### RetryConfig

```python
class RetryConfig(BaseModel):
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
```

### Message Schema

#### MessageBlock

```python
class MessageBlock(BaseModel):
    role: str
    content: Union[str, List[Union[TextBlock, ImageBlock, ToolBlock]]]
    metadata: Optional[Dict[str, Any]] = None
```

#### TextBlock

```python
class TextBlock(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
```

### Agent Classes

#### Agent

```python
class Agent(AsyncClient):
    async def generate_and_action_async(
        self,
        prompt: str,
        available_tools: Optional[List[str]] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate response and execute tools as needed."""
```

## Error Handling

### Common Errors

1. `ModelImplementationError`:
   - Raised when model-specific operations fail
   - Contains provider error details
   - Includes request/response context

2. `ToolExecutionError`:
   - Raised during tool execution failures
   - Contains tool name and parameters
   - Includes error context and stack trace

3. `ConfigurationError`:
   - Raised for invalid configurations
   - Includes validation details
   - Suggests correct configuration

### Error Handling Examples

```python
from bedrock_llm.exceptions import ModelImplementationError

try:
    response = client.generate(messages)
except ModelImplementationError as e:
    print(f"Model error: {e.message}")
    print(f"Provider: {e.provider}")
    print(f"Request ID: {e.request_id}")
```

## Community and Support

### Contributing

1. Fork the repository
2. Create a feature branch
3. Follow style guide and tests
4. Submit pull request

### Getting Help

- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and community support
- Documentation: Full API reference and examples
