# LLM Review Module

## Purpose

The review module provides utilities for generating manuscript reviews using LLM integration. It includes review generation, I/O operations, and metrics tracking.

## Components

### ReviewGenerator (`generator.py`)

Generates manuscript reviews with:
- **Executive summaries**: High-level overview of manuscripts
- **Quality reviews**: Detailed quality assessments
- **Methodology reviews**: Methodology-focused analysis
- **Improvement suggestions**: Actionable improvement recommendations
- **Translation abstracts**: Abstract translations

### ReviewIO (`io.py`)

Handles review input/output operations:
- **Review loading**: Load reviews from files
- **Review saving**: Save reviews to files
- **Action item extraction**: Extract actionable items from reviews
- **Format compliance**: Check review format compliance

### ReviewMetrics (`metrics.py`)

Tracks and analyzes review metrics:
- **Streaming metrics**: Real-time generation metrics
- **Quality metrics**: Review quality measurements
- **Performance metrics**: Generation performance tracking

## Usage Examples

### Generating Reviews

```python
from infrastructure.llm.review.generator import generate_review

review = generate_review(
    manuscript_text=text,
    review_type="executive_summary",
    llm_client=client
)
```

### Review I/O

```python
from infrastructure.llm.review.io import save_review, load_review

# Save review
save_review(review, output_path)

# Load review
review = load_review(input_path)
```

### Metrics Tracking

```python
from infrastructure.llm.review.metrics import StreamingMetrics

metrics = StreamingMetrics()
# Track metrics during generation
```

## Review Types

### Executive Summary

High-level overview covering:
- Overview
- Key contributions
- Methodology summary
- Principal results
- Significance and impact

### Quality Review

Detailed quality assessment with:
- Scoring (1-5 scale)
- Strengths and weaknesses
- Priority issues
- Recommendations

### Methodology Review

Methodology-focused analysis:
- Experimental design
- Statistical methods
- Reproducibility
- Methodological rigor

## Configuration

Environment variables:
- `LLM_MAX_INPUT_LENGTH` - Maximum input length (default: 500000)
- `LLM_REVIEW_MAX_TOKENS` - Maximum tokens for reviews (default: 16384)
- `LLM_REVIEW_TIMEOUT` - Review generation timeout (default: 600)

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../llm/AGENTS.md`](../../llm/AGENTS.md) - LLM module overview
- [`generator.py`](generator.py) - Review generation implementation



