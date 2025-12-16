# LLM Templates Module

Pre-built prompt templates for research tasks.

## Quick Start

```python
from infrastructure.llm.templates import PaperSummarization

template = PaperSummarization()
prompt = template.render(text=paper_text)
```

## Available Templates

- Research templates (summarization, literature review)
- Manuscript templates (reviews, summaries, translations)

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation
- [`../../llm/README.md`](../../llm/README.md) - LLM module overview



