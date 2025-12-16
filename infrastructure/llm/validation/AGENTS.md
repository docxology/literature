# LLM Validation Module

## Purpose

The validation module provides output validation for LLM responses, including JSON validation, structure checking, format compliance, and repetition detection.

## Components

### OutputValidator (`core.py`)

Main validation class with methods for:
- **JSON validation**: Parse and validate JSON responses
- **Length validation**: Check response length against requirements
- **Structure validation**: Validate against JSON schemas
- **Citation extraction**: Extract citations from responses
- **Validation**: Validation pipeline

### Format Validation (`format.py`)

Format compliance checking:
- **Emoji detection**: Detect emoji usage
- **Table detection**: Detect markdown tables
- **Conversational phrases**: Detect AI assistant phrases
- **Hallucination detection**: Detect fictional section references
- **Off-topic detection**: Detect off-topic responses

### Repetition Detection (`repetition.py`)

Repetition analysis:
- **Sentence similarity**: Detect repeated sentences
- **Paragraph similarity**: Detect repeated paragraphs
- **Deduplication**: Remove duplicate content
- **Repetition scoring**: Score repetition levels

### Structure Validation (`structure.py`)

Structure validation:
- **Section completeness**: Check required sections present
- **Section extraction**: Extract structured sections
- **Response structure**: Validate overall structure

## Usage Examples

### JSON Validation

```python
from infrastructure.llm.validation import OutputValidator

validator = OutputValidator()
data = validator.validate_json(response)
```

### Validation

```python
from infrastructure.llm.validation import OutputValidator

validator = OutputValidator()
is_valid, errors = validator.validate_complete(
    response,
    mode="structured",
    schema=my_schema
)
```

### Format Compliance

```python
from infrastructure.llm.validation.format import check_format_compliance

is_compliant, issues, details = check_format_compliance(response)
```

### Repetition Detection

```python
from infrastructure.llm.validation.repetition import detect_repetition

has_repetition, score = detect_repetition(text)
```

## Validation Modes

### Short Response Validation

Validates responses < 150 tokens:
- Length check
- Completeness check
- Format validation

### Long Response Validation

Validates responses > 500 tokens:
- Length check
- Structure validation
- Quality checks

### Structured Response Validation

Validates JSON responses:
- JSON parsing
- Schema validation
- Structure validation

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../llm/AGENTS.md`](../../llm/AGENTS.md) - LLM module overview
- [`core.py`](core.py) - OutputValidator implementation



