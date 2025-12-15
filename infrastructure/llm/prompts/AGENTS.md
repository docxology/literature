# LLM Prompts Module

## Purpose

The prompts module provides a composable prompt fragment system for building complex LLM prompts from reusable components. It supports loading fragments from JSON files, composing templates, and managing prompt structures.

## Components

### PromptFragmentLoader (`loader.py`)

Loads prompt fragments, templates, and compositions from JSON files:
- **Fragment loading**: Load specific sections from JSON files using references like `"file.json#key"`
- **Template loading**: Load complete template definitions
- **Caching**: LRU cache for loaded fragments to improve performance
- **Validation**: Validates loaded fragments and templates

### PromptComposer (`composer.py`)

Composes prompts from fragments and templates:
- **Template composition**: Assembles complete prompts from template definitions
- **Variable substitution**: Replaces template variables with actual values
- **Fragment composition**: Combines multiple fragments into single prompts
- **Token budget awareness**: Adjusts prompts based on available token budget

### Directory Structure

```
prompts/
├── fragments/          # Reusable prompt fragments
│   ├── content_requirements.json
│   ├── format_requirements.json
│   ├── section_structures.json
│   ├── system_prompts.json
│   ├── token_budget_awareness.json
│   └── validation_hints.json
├── templates/         # Complete prompt templates
│   ├── manuscript_reviews.json
│   └── paper_summarization.json
└── compositions/      # Composed prompt structures
    └── retry_prompts.json
```

## Usage Examples

### Loading Fragments

```python
from infrastructure.llm.prompts.loader import PromptFragmentLoader

loader = PromptFragmentLoader()

# Load a system prompt fragment
system_prompt = loader.load_fragment("system_prompts.json#manuscript_review")

# Load a template
template = loader.load_template("manuscript_reviews.json#manuscript_executive_summary")
```

### Composing Prompts

```python
from infrastructure.llm.prompts.composer import PromptComposer
from infrastructure.llm.prompts.loader import PromptFragmentLoader

loader = PromptFragmentLoader()
composer = PromptComposer(loader=loader)

# Compose a template with variables
prompt = composer.compose_template(
    "manuscript_reviews.json#manuscript_executive_summary",
    text=manuscript_text,
    max_tokens=1000
)
```

### Fragment References

Fragment references use the format `"filename.json#key"`:
- `"system_prompts.json#manuscript_review"` - Loads the `manuscript_review` key from `system_prompts.json`
- `"templates.json#template_name"` - Loads a template definition

## Fragment Types

### Content Requirements

Defines what content should be included in prompts:
- Section requirements
- Quality standards
- Coverage expectations

### Format Requirements

Specifies formatting requirements:
- Markdown structure
- Header requirements
- Citation formats

### Section Structures

Defines section structures for different document types:
- Manuscript sections
- Review sections
- Summary sections

### System Prompts

Pre-defined system prompts for different use cases:
- Manuscript review
- Paper summarization
- Research assistance

### Token Budget Awareness

Helps adjust prompts based on available token budget:
- Dynamic content selection
- Truncation strategies
- Priority-based inclusion

### Validation Hints

Provides hints for output validation:
- Expected formats
- Quality criteria
- Common issues to check

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../llm/AGENTS.md`](../../llm/AGENTS.md) - LLM module overview
- [`loader.py`](loader.py) - Fragment loader implementation
- [`composer.py`](composer.py) - Prompt composer implementation



