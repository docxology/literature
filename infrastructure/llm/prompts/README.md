# LLM Prompts Module

Composable prompt fragment system for building LLM prompts.

## Quick Start

```python
from infrastructure.llm.prompts import PromptFragmentLoader, PromptComposer

loader = PromptFragmentLoader()
composer = PromptComposer(loader=loader)

prompt = composer.compose_template(
    "manuscript_reviews.json#template",
    text=content,
    max_tokens=1000
)
```

## Components

- **loader.py** - Load fragments and templates from JSON
- **composer.py** - Compose prompts from fragments
- **fragments/** - Reusable prompt fragments
- **templates/** - Complete prompt templates

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation
- [`../../llm/README.md`](../../llm/README.md) - LLM module overview



