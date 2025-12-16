# LLM Core Module

Core components for LLM integration: client, config, and context management.

## Quick Start

```python
from infrastructure.llm.core import LLMClient, LLMConfig

client = LLMClient()
response = client.query("What is machine learning?")
```

## Components

- **client.py** - LLMClient main interface
- **config.py** - LLMConfig and GenerationOptions
- **context.py** - ConversationContext management
- **response_saver.py** - Response persistence

## Features

- Multiple response modes (short, long, structured, raw)
- Streaming support
- Context management
- Per-query configuration
- Error handling

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation
- [`../../llm/README.md`](../../llm/README.md) - LLM module overview



