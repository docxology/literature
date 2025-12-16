# Summarizing Papers Guide

Guide to generating LLM-powered summaries for academic papers.

## Prerequisites

- Ollama server running (`ollama serve`)
- Model installed (e.g., `ollama pull gemma3:4b`)
- PDF files in `data/pdfs/` directory

## Quick Start

### Command Line

```bash
# Generate summaries for existing PDFs
python3 scripts/literature_search.py --summarize
```

### Python API

```python
from infrastructure.literature.summarization import SummarizationEngine
from infrastructure.llm import LLMClient
from pathlib import Path

# Initialize LLM client
llm_client = LLMClient()

# Initialize summarization engine
engine = SummarizationEngine(llm_client)

# Summarize a paper
result = engine.summarize_paper(
    result=search_result,
    pdf_path=Path("data/pdfs/paper.pdf")
)

# Check result
if result.summary_text:
    print(f"Summary: {result.output_words} words")
    print(f"Quality score: {result.quality_score:.2f}")
```

## Summarization Process

The summarization system uses a multi-stage approach:

1. **PDF Text Extraction** - Extract text with section prioritization
2. **Context Extraction** - Extract key sections (abstract, intro, conclusion)
3. **Draft Generation** - Generate initial summary using LLM
4. **Quality Validation** - Validate summary quality
5. **Refinement** - Refine summary if needed (with retries)

## Configuration

### Environment Variables

```bash
# Maximum parallel summaries
export MAX_PARALLEL_SUMMARIES=1

# Summarization timeout
export LLM_SUMMARIZATION_TIMEOUT=600

# LLM model
export OLLAMA_MODEL=gemma3:4b
```

### Engine Configuration

```python
from infrastructure.literature.summarization import (
    SummarizationEngine,
    SummaryQualityValidator
)

# Custom validator
validator = SummaryQualityValidator(min_words=200)

# Custom engine
engine = SummarizationEngine(
    llm_client=llm_client,
    quality_validator=validator,
    max_pdf_chars=100000  # Limit PDF characters
)
```

## Progress Tracking

### Progress

```python
def progress_callback(event):
    print(f"{event.stage}: {event.status} - {event.message}")

result = engine.summarize_paper(
    result=search_result,
    pdf_path=pdf_path,
    progress_callback=progress_callback
)
```

### Progress Events

- `pdf_extraction` - PDF text extraction
- `context_extraction` - Context extraction
- `draft_generation` - Draft generation
- `validation` - Quality validation
- `refinement` - Summary refinement

## Quality Validation

Summaries are validated for:
- **Length** - Minimum word count requirements
- **Title matching** - Title consistency
- **Repetition** - Sentence/paragraph repetition detection
- **Quotes** - Presence of direct quotes
- **Structure** - Required sections present

### Validation Results

```python
result = engine.summarize_paper(...)

if result.success:
    print("Summary accepted")
else:
    print(f"Validation issues: {result.validation_errors}")
    # Summary is still saved for review
```

## Output

Summaries are saved to `data/summaries/{citation_key}_summary.md` with:
- Generated summary text
- Validation metadata (status, score, errors)
- Generation statistics (time, tokens, words)

## Best Practices

1. **Use appropriate models** - Larger models produce better summaries
2. **Monitor progress** - Use progress callbacks for long operations
3. **Review validation** - Check validation errors for quality issues
4. **Parallel processing** - Use `MAX_PARALLEL_SUMMARIES` for multiple papers
5. **Model selection** - Choose models with sufficient context window

## Troubleshooting

### Low Quality Scores

- Check PDF text extraction quality
- Verify context extraction is finding key sections
- Review validation errors
- Consider adjusting refinement attempts

### Slow Generation

- Reduce `max_pdf_chars` to process less text
- Use parallel processing for multiple papers
- Check LLM connection and model performance
- Consider using faster LLM models

### Missing Sections

- Check PDF text extraction (may be missing sections)
- Verify context extraction is finding all sections
- Increase `max_pdf_chars` if truncation is occurring

### Repetition Issues

**Symptoms**: Validation errors like "Severe repetition detected: Same sentence appears X times"

**Automatic Handling**:
- System detects severe repetition and applies aggressive deduplication
- Lower temperature (0.2) used during refinement for repetition issues
- More aggressive similarity thresholds (0.75) for deduplication

**Manual Solutions**:
- Check PDF text quality - some PDFs may have duplicate content
- Try a different LLM model
- Review failed summaries (saved with quality score 0.00) for patterns
- Consider manual editing of problematic summaries

### Failure Analysis

At the end of each summarization run, the system provides:

1. **Failure Summary**: Count of failures by category
   - Repetition Issues
   - LLM Connection Error
   - Context Limit Exceeded
   - PDF Extraction Error
   - Title Mismatch
   - Other Errors

2. **Examples**: Sample failed papers with error messages

3. **Suggestions**: Specific recommendations based on failure patterns

**Example Output**:
```
FAILURE ANALYSIS
------------------------------------------------------------
Total failures: 5

Failures by category:
  Repetition Issues: 3 (60.0%)
    - paper1: Severe repetition detected: Same sentence appears 4 times
    - paper2: Severe repetition detected: Same phrase appears 6 times
    ... and 1 more
  LLM Connection Error: 2 (40.0%)
    - paper3: LLM connection unavailable after 2 attempts
    - paper4: LLM connection error after 1 attempts

Suggestions:
  • Repetition issues: Consider using lower temperature or different model
  • Connection errors: Check Ollama is running and accessible
```

### Progress Tracking

The progress bar now shows:
- Success/failure counts: `[████████░░░░] 281/513 (54%) ✓:275 ✗:6`
- ETA calculation that accounts for success vs failure rates
- More accurate time estimates for remaining work

## See Also

- **[Summarization Module Documentation](../modules/literature.md)** - Module documentation
- **[API Reference](../reference/api-reference.md)** - API documentation
- **[Guides AGENTS.md](AGENTS.md)** - Guide organization and standards

