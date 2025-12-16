# LLM Templates Module

## Purpose

The templates module provides pre-built prompt templates for common research tasks. Templates are Python classes that can be rendered with variables to create complete prompts.

## Components

### ResearchTemplate (`base.py`)

Base class for all research templates:
- **Template rendering**: Variable substitution using Python's Template class
- **Error handling**: Validates required variables
- **Extensibility**: Easy to create custom templates

### Template Classes

- **PaperSummarization** (`research.py`) - Paper summarization templates
- **ManuscriptExecutiveSummary** (`manuscript.py`) - Executive summary templates
- **ManuscriptQualityReview** (`manuscript.py`) - Quality review templates
- **ManuscriptMethodologyReview** (`manuscript.py`) - Methodology review templates
- **ManuscriptImprovementSuggestions** (`manuscript.py`) - Improvement suggestion templates
- **ManuscriptTranslationAbstract** (`manuscript.py`) - Translation abstract templates

### Template Helpers (`helpers.py`)

Utility functions for template operations:
- Template registry
- Template discovery
- Template validation

## Usage Examples

### Using Built-in Templates

```python
from infrastructure.llm.templates import PaperSummarization

template = PaperSummarization()
prompt = template.render(
    text=paper_text,
    focus="methodology"
)
```

### Creating Custom Templates

```python
from infrastructure.llm.templates.base import ResearchTemplate

class MyTemplate(ResearchTemplate):
    template_str = (
        "Analyze the following with focus on ${aspect}:\n\n"
        "${content}"
    )

template = MyTemplate()
prompt = template.render(aspect="limitations", content="...")
```

### Template Registry

```python
from infrastructure.llm.templates import get_template

template = get_template("summarize_abstract")
prompt = template.render(text=abstract)
```

## Available Templates

### Research Templates

- `summarize_abstract` - Summarize research abstracts
- `literature_review` - Synthesize multiple summaries
- `code_doc` - Generate Python docstrings
- `data_interpret` - Interpret statistical results

### Manuscript Templates

- `manuscript_executive_summary` - Executive summary generation
- `manuscript_quality_review` - Quality review generation
- `manuscript_methodology_review` - Methodology review generation
- `manuscript_improvement_suggestions` - Improvement suggestions
- `manuscript_translation_abstract` - Abstract translation

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../llm/AGENTS.md`](../../llm/AGENTS.md) - LLM module overview
- [`base.py`](base.py) - Base template class
- [`research.py`](research.py) - Research templates
- [`manuscript.py`](manuscript.py) - Manuscript templates




