# LLM Core Module

## Purpose

The LLM core module provides the foundational components for interacting with local large language models via Ollama. It includes the main LLMClient, configuration management, and conversation context handling.

## Components

### LLMClient (`client.py`)

Main interface for querying LLMs with multiple response modes:
- **Standard queries**: Conversational queries with context
- **Short responses**: Brief answers (< 150 tokens)
- **Long responses**: Detailed answers (> 500 tokens)
- **Structured responses**: JSON-formatted with schema validation
- **Raw queries**: Direct prompts without modification
- **Streaming**: Response generation

### LLMConfig (`config.py`)

Configuration management for LLM operations:
- Environment variable loading
- Model settings and generation defaults
- Response mode token limits
- System prompt configuration
- Per-query option creation

### GenerationOptions (`config.py`)

Per-query generation control:
- Temperature, max_tokens, top_p, top_k
- Seed for reproducibility
- Stop sequences
- Native JSON format mode
- Repeat penalty, num_ctx

### ConversationContext (`context.py`)

Multi-turn conversation management:
- Message history tracking
- Token limit enforcement
- Context pruning strategies
- System prompt preservation

### ResponseSaver (`response_saver.py`)

Response saving utilities for persistence and debugging.

### EmbeddingClient (`embedding_client.py`)

Client for generating text embeddings using Ollama embedding models.

**Key Features:**
- Single and batch embedding generation
- Automatic text chunking for large documents (handles 2048 token limit)
- Document-level embedding aggregation via mean pooling
- Embedding caching support
- Request monitoring with progress updates
- Automatic Ollama restart on timeout/hung state
- Adaptive timeout scaling based on text length

**Key Methods:**
- `generate_embedding(text)` - Generate embedding for single text
- `generate_embeddings_batch(texts)` - Generate embeddings for multiple texts
- `generate_document_embedding(text, chunk_size=None)` - Generate document-level embedding with automatic chunking
- `check_connection(timeout=5.0)` - Check Ollama server availability
- `check_embedding_endpoint(timeout=10.0)` - Verify embedding endpoint is working

**RequestMonitor (`embedding_client.py`)**

Background monitoring for long-running embedding requests:
- Periodic progress updates (heartbeat logging)
- Timeout warning thresholds (50%, 75%, 90%)
- Non-blocking background thread monitoring

**Behavior:**
- Automatically started for requests with timeout > 30s
- Heartbeat interval adapts based on text length:
  - Very long documents (>50K chars): 5s intervals
  - Medium documents (10K-50K chars): 10s intervals
  - Shorter documents: 20s intervals
- Runs in daemon thread, automatically stops when request completes
- Provides visibility into long-running operations that might appear hung

**Log Messages:**
When a request monitor is started:
```
  → Request monitor started (heartbeat every 20s)
```

During processing, periodic heartbeats indicate the request is still active:
```
  ↻ Still processing embedding request... (15.2s elapsed, 77.6s remaining, text length: 3,979 chars)
```

Timeout warnings are logged at 50%, 75%, and 90% of timeout elapsed:
```
  ⚠ Embedding request at 50% of timeout (60.0s/120.0s elapsed, 60.0s remaining, text length: 50,000 chars)
```

The monitor automatically stops when the request completes or fails.

**Usage:**
```python
from infrastructure.llm.core.embedding_client import EmbeddingClient

# Initialize client
client = EmbeddingClient(
    embedding_model="embeddinggemma",
    cache_dir=Path("data/embeddings"),
    chunk_size=2000,
    timeout=120.0
)

# Generate single embedding
embedding = client.generate_embedding("Machine learning is fascinating")

# Generate batch embeddings
embeddings = client.generate_embeddings_batch(["Text 1", "Text 2", "Text 3"])

# Generate document-level embedding (auto-chunks large texts)
doc_embedding = client.generate_document_embedding(long_text)
```

**Configuration:**
- Uses `LLMConfig` for base configuration
- Embedding-specific settings via constructor parameters
- Environment variables: `LITERATURE_EMBEDDING_MODEL`, `LITERATURE_EMBEDDING_CACHE_DIR`, etc.

## Usage Examples

### Basic Client Usage

```python
from infrastructure.llm.core.client import LLMClient
from infrastructure.llm.core.config import LLMConfig

# Initialize with default config
client = LLMClient()

# Or with custom config
config = LLMConfig(
    base_url="http://localhost:11434",
    default_model="gemma3:4b",
    temperature=0.7
)
client = LLMClient(config)
```

### Query Modes

```python
# Standard query
response = client.query("What is machine learning?")

# Short response
answer = client.query_short("Define AI")

# Long response
explanation = client.query_long("Explain neural networks in detail")

# Structured response
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array"}
    }
}
result = client.query_structured("Analyze...", schema=schema)
```

### Context Management

```python
# Multi-turn conversation
response1 = client.query("What is X?")
response2 = client.query("Can you elaborate?")  # Context maintained

# Reset context
client.reset()

# Change system prompt
client.set_system_prompt("You are an expert researcher.")
```

### Generation Options

```python
from infrastructure.llm.core.config import GenerationOptions

opts = GenerationOptions(
    temperature=0.0,      # Deterministic
    seed=42,              # Reproducibility
    max_tokens=1000,      # Limit output
    stop=["END", "STOP"]  # Stop sequences
)
response = client.query("...", options=opts)
```

## Configuration

### Environment Variables

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gemma3:4b
export LLM_TEMPERATURE=0.7
export LLM_MAX_TOKENS=2048
export LLM_CONTEXT_WINDOW=131072
export LLM_TIMEOUT=60
export LLM_SEED=42
export LLM_SYSTEM_PROMPT="You are an expert research assistant."
```

## Error Handling

```python
from infrastructure.core.exceptions import (
    LLMConnectionError,
    LLMError,
    ContextLimitError
)

try:
    response = client.query("...")
except LLMConnectionError as e:
    print(f"Connection failed: {e.context}")
except ContextLimitError as e:
    print(f"Context limit exceeded: {e.context}")
```

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../llm/AGENTS.md`](../../llm/AGENTS.md) - LLM module overview
- [`client.py`](client.py) - LLMClient implementation
- [`embedding_client.py`](embedding_client.py) - EmbeddingClient implementation

