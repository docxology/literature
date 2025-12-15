# Prompt Compositions

## Purpose

The compositions directory contains pre-composed prompt structures for specific use cases, particularly retry scenarios and complex prompt assemblies.

## Composition Files

### retry_prompts.json

Prompts designed for retry scenarios:
- Format enforcement prompts
- Error correction prompts
- Quality improvement prompts
- Retry-specific instructions

## Usage

Compositions are loaded via `PromptFragmentLoader`:

```python
from infrastructure.llm.prompts.loader import PromptFragmentLoader

loader = PromptFragmentLoader()
composition = loader.load_composition("retry_prompts.json#retry_template")
```

## Composition Structure

Compositions are complete prompt structures that can be used directly or as building blocks for more complex prompts.

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Prompts module documentation


