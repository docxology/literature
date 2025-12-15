# Prompt Fragments

Reusable prompt fragments stored as JSON files.

## Fragment Files

- **content_requirements.json** - Content requirements and standards
- **format_requirements.json** - Formatting specifications
- **section_structures.json** - Section structure definitions
- **system_prompts.json** - System prompt definitions
- **token_budget_awareness.json** - Token budget management
- **validation_hints.json** - Validation criteria

## Usage

Fragments are loaded via `PromptFragmentLoader`:

```python
loader = PromptFragmentLoader()
fragment = loader.load_fragment("system_prompts.json#manuscript_review")
```

## See Also

- [`../AGENTS.md`](../AGENTS.md) - Prompts module documentation


