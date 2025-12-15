# Prompt Fragments

## Purpose

The fragments directory contains reusable prompt fragments stored as JSON files. These fragments are building blocks for composing complete prompts.

## Fragment Files

### content_requirements.json

Defines content requirements and standards for prompts:
- Section requirements
- Quality standards
- Coverage expectations
- Content guidelines

### format_requirements.json

Specifies formatting requirements:
- Markdown structure
- Header requirements
- Citation formats
- Formatting standards

### section_structures.json

Defines section structures for different document types:
- Manuscript sections
- Review sections
- Summary sections
- Document templates

### system_prompts.json

Pre-defined system prompts for different use cases:
- Manuscript review prompts
- Paper summarization prompts
- Research assistance prompts
- Domain-specific prompts

### token_budget_awareness.json

Helps adjust prompts based on available token budget:
- Dynamic content selection
- Truncation strategies
- Priority-based inclusion
- Token optimization

### validation_hints.json

Provides hints for output validation:
- Expected formats
- Quality criteria
- Common issues to check
- Validation patterns

## Usage

Fragments are loaded via `PromptFragmentLoader`:

```python
from infrastructure.llm.prompts.loader import PromptFragmentLoader

loader = PromptFragmentLoader()
fragment = loader.load_fragment("system_prompts.json#manuscript_review")
```

## Fragment Reference Format

Fragments use the format `"filename.json#key"`:
- `"system_prompts.json#manuscript_review"` - Loads the `manuscript_review` key
- `"content_requirements.json#section_requirements"` - Loads specific requirements

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Prompts module documentation



