# LLM Utils Module

Utility functions for Ollama server and model management.

## Quick Start

```python
from infrastructure.llm.utils.ollama import is_ollama_running, select_best_model

if is_ollama_running():
    model = select_best_model()
    print(f"Using model: {model}")
```

## Functions

- `is_ollama_running()` - Check server status
- `get_available_models()` - List models
- `select_best_model()` - Select best model
- `preload_model()` - Preload model

## See Also

- [`AGENTS.md`](AGENTS.md) - Complete documentation
- [`../../llm/README.md`](../../llm/README.md) - LLM module overview


