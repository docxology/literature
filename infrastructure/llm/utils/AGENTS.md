# LLM Utils Module

## Purpose

The utils module provides utility functions for Ollama server management, model discovery, and connection handling.

## Components

### Ollama Utilities (`ollama.py`)

Functions for Ollama server and model management:

#### Model Discovery

- **`is_ollama_running()`** - Check if Ollama server is running
- **`get_available_models()`** - List all available Ollama models
- **`select_best_model()`** - Select best model based on preferences
- **`get_model_info()`** - Get detailed information about a model

#### Server Management

- **`start_ollama_server()`** - Start Ollama server if not running
- **`check_ollama_health()`** - Comprehensive health check

#### Model Operations

- **`preload_model()`** - Preload a model to reduce first-query latency
- **`ensure_model_available()`** - Ensure a model is available, install if needed

## Usage Examples

### Checking Ollama Status

```python
from infrastructure.llm.utils.ollama import is_ollama_running

if is_ollama_running():
    print("Ollama is ready")
else:
    print("Ollama is not running")
```

### Model Discovery

```python
from infrastructure.llm.utils.ollama import (
    get_available_models,
    select_best_model
)

# List all models
models = get_available_models()
print(f"Available models: {models}")

# Select best model
best_model = select_best_model()
print(f"Best model: {best_model}")
```

### Model Preloading

```python
from infrastructure.llm.utils.ollama import preload_model

success, error = preload_model("llama3.2:3b", timeout=60.0)
if success:
    print("Model preloaded successfully")
else:
    print(f"Preload failed: {error}")
```

## Model Preferences

The module uses a default preference list for model selection:
1. `llama3-gradient:latest` - Large context (256K), reliable
2. `llama3.1:latest` - Good balance
3. `llama2:latest` - Widely available
4. `gemma2:2b` - Fast, small
5. `gemma3:4b` - Medium size
6. `mistral:latest` - Alternative
7. `codellama:latest` - Code-focused

## Configuration

Environment variables:
- `OLLAMA_HOST` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_TIMEOUT` - Connection timeout (default: 2.0)

## Error Handling

Functions return detailed error information:
- Connection errors with helpful messages
- Model availability checks
- Timeout handling with retries

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../llm/AGENTS.md`](../../llm/AGENTS.md) - LLM module overview
- [`ollama.py`](ollama.py) - Implementation



