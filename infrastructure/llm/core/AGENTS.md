# LLM Core Module

## Purpose

The LLM core module provides the foundational components for interacting with local large language models via Ollama. It includes the main LLMClient, configuration management, and conversation context handling.

## Components

### LLMClient (`client.py`)

Main interface for querying LLMs with multiple response modes:
- **Standard queries**: Conversational queries with context
- **Short responses**: Brief answers (< 150 tokens)
- **Long responses**: Comprehensive answers (> 500 tokens)
- **Structured responses**: JSON-formatted with schema validation
- **Raw queries**: Direct prompts without modification
- **Streaming**: Real-time response generation

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
- [`../client.py`](client.py) - LLMClient implementation

