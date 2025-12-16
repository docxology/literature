# LLM Validation Module

Output validation for LLM responses.

## Quick Start

```python
from infrastructure.llm.validation import OutputValidator

validator = OutputValidator()
is_valid, errors = validator.validate_complete(response, mode="structured")
```

## Components

- **core.py** - Main OutputValidator class
- **format.py** - Format compliance checking
- **repetition.py** - Repetition detection
- **structure.py** - Structure validation

## Validation Types

- JSON validation
- Length validation
- Format compliance
- Repetition detection
- Structure validation

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation
- [`../../llm/README.md`](../../llm/README.md) - LLM module overview



