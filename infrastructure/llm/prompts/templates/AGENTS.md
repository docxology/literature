# Prompt Templates

## Purpose

The templates directory contains complete prompt templates for common LLM operations. Templates are JSON files that define full prompt structures.

## Template Files

### manuscript_reviews.json

Complete templates for manuscript review operations:
- Executive summary templates
- Quality review templates
- Methodology review templates
- Improvement suggestion templates

### paper_summarization.json

Templates for paper summarization:
- Abstract summarization
- Full paper summarization
- Section-specific summaries

## Usage Examples

Templates are loaded and composed via `PromptComposer`:

```python
from infrastructure.llm.prompts.composer import PromptComposer
from infrastructure.llm.prompts.loader import PromptFragmentLoader

loader = PromptFragmentLoader()
composer = PromptComposer(loader=loader)

prompt = composer.compose_template(
    "manuscript_reviews.json#template_name",
    text=content,
    max_tokens=1000
)
```

## Template Structure

Templates define:
- Prompt structure
- Variable placeholders
- Fragment references
- Format requirements
- Content requirements

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Prompts module documentation



