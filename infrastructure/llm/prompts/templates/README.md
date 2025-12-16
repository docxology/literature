# Prompt Templates

Complete prompt templates for common LLM operations.

## Template Files

- **manuscript_reviews.json** - Manuscript review templates
- **paper_summarization.json** - Paper summarization templates

## Usage

Templates are loaded and composed via `PromptComposer`:

```python
composer = PromptComposer()
prompt = composer.compose_template(
    "manuscript_reviews.json#template_name",
    **variables
)
```

## See Also

- [`../AGENTS.md`](../AGENTS.md) - Prompts module documentation






