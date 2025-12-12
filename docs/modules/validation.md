# Validation Module

PDF validation and text extraction utilities.

## Overview

The validation module provides PDF text extraction functionality needed by the literature summarization system. It supports multiple PDF parsing libraries with automatic fallback.

## Key Components

### PDF Text Extraction (`pdf_validator.py`)

Extracts text content from PDF files with automatic fallback:
- **Text extraction**: Extracts all text from PDF files
- **Multi-library support**: Tries libraries in order: pdfplumber → pypdf → PyPDF2
- **Warning suppression**: Suppresses harmless pypdf warnings
- **Error handling**: Comprehensive error handling for PDF issues
- **Fallback logic**: Automatically tries alternative libraries if one fails

## Usage Examples

### Extracting Text from PDF

```python
from infrastructure.validation import extract_text_from_pdf
from pathlib import Path

pdf_path = Path("data/pdfs/paper.pdf")
text = extract_text_from_pdf(pdf_path)
```

### Error Handling

```python
from infrastructure.validation import (
    extract_text_from_pdf,
    PDFValidationError
)

try:
    text = extract_text_from_pdf(pdf_path)
except PDFValidationError as e:
    print(f"PDF extraction failed: {e}")
```

## Dependencies

The module supports multiple PDF parsing libraries with automatic fallback:
- **pdfplumber** (optional, recommended): Best quality extraction, tries first
- **pypdf** (required): Modern PyPDF2 replacement, tries second
- **PyPDF2** (optional, legacy): Legacy library, tries third

At least one of these libraries must be installed. The module will automatically use the first available library in the order listed above.

**Installation examples:**
```bash
# Recommended: install pdfplumber for best quality (optional dependency)
uv pip install pdfplumber
# Or: pip install pdfplumber

# Or install via optional dependencies
uv pip install -e ".[pdf]"

# pypdf is included in project dependencies (installed automatically)
# Or install legacy PyPDF2
pip install PyPDF2
```

## Integration

The validation module is used by:
- Literature summarization system for PDF text extraction
- PDF processing workflows

## See Also

- **[Validation Module Documentation](../infrastructure/validation/AGENTS.md)** - Complete documentation
- **[API Reference](../reference/api-reference.md)** - API documentation

