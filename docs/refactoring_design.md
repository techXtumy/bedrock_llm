# Client Implementation Refactoring Design

## Current Structure Analysis

- The current `LLMClient` class in `client.py` mixes sync and async implementations
- Agent class inherits from LLMClient and primarily uses async methods
- Embedding functionality is currently part of the main LLMClient
- No separate client for reranking models exists

## Proposed Refactoring

1. Split into separate base classes:
   - `BaseClient` - Common functionality
   - `AsyncClient` - Async-specific implementation
   - `SyncClient` - Sync-specific implementation
   - `EmbeddingsClient` - Dedicated embeddings functionality
   - `RerankingClient` - New client for reranking operations

2. Inheritance Structure:

   ```bash
   BaseClient
   ├── AsyncClient
   │   └── Agent (existing)
   ├── SyncClient
   ├── EmbeddingsClient
   └── RerankingClient
   ```

3. File Organization:

   ```bash
   src/bedrock_llm/
   ├── client/
   │   ├── __init__.py
   │   ├── base.py
   │   ├── async_client.py
   │   ├── sync_client.py
   │   ├── embeddings.py
   │   └── reranking.py
   ├── agent.py
   └── ...
   ```

## Implementation Plan

1. Create new client package structure
2. Extract common base functionality
3. Split sync/async implementations
4. Create specialized embedding/reranking clients
5. Update agent.py to use new async client
6. Update documentation and examples
