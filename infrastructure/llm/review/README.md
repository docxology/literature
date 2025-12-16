# LLM Review Module

Manuscript review generation and management.

## Quick Start

```python
from infrastructure.llm.review import generate_review

review = generate_review(manuscript_text, review_type="executive_summary")
```

## Components

- **generator.py** - Review generation
- **io.py** - Review I/O operations
- **metrics.py** - Review metrics tracking

## Review Types

- Executive summary
- Quality review
- Methodology review
- Improvement suggestions

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation
- [`../../llm/README.md`](../../llm/README.md) - LLM module overview



