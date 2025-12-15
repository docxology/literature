# Prompt Compositions

Pre-composed prompt structures for specific use cases.

## Composition Files

- **retry_prompts.json** - Prompts for retry scenarios

## Usage

Compositions are loaded via `PromptFragmentLoader`:

```python
loader = PromptFragmentLoader()
composition = loader.load_composition("retry_prompts.json#retry_template")
```

## See Also

- [`../AGENTS.md`](../AGENTS.md) - Prompts module documentation



