# Validation Module

PDF text extraction for literature processing.

## Quick Start

```python
from infrastructure.validation import extract_text_from_pdf

text = extract_text_from_pdf(Path("paper.pdf"))
```

## Functions

- `extract_text_from_pdf()` - Extract text from PDF files

## Dependencies

- **pdfplumber** (optional, recommended): Best quality extraction
- **pypdf** (required): Included in project dependencies
- **PyPDF2** (optional, legacy): Fallback option

Install optional dependencies:
```bash
uv pip install pdfplumber
# Or: uv pip install -e ".[pdf]"
```

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation

