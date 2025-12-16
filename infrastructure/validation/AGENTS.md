# Validation Module

## Purpose

The validation module provides PDF text extraction functionality needed by the literature summarization system. This is a minimal subset focused on PDF processing.

## Components

### PDF Text Extraction (`pdf_validator.py`)

Extracts text content from PDF files with automatic fallback:
- **Text extraction**: Extracts all text from PDF files
- **Multi-library support**: Tries libraries in order: pdfplumber → pypdf → PyPDF2
- **Warning suppression**: Suppresses harmless pypdf warnings
- **Error handling**: Error handling for PDF issues
- **Fallback logic**: Automatically tries alternative libraries if one fails

## Configuration

The validation module uses environment variables for configuration:

### PDF Library Selection

The module automatically tries PDF parsing libraries in order:
1. `pdfplumber` (recommended for best quality)
2. `pypdf` (modern PyPDF2 replacement)
3. `PyPDF2` (legacy library)

At least one of these libraries must be installed. The module will automatically use the first available library.

### Installation

```bash
# Recommended: install pdfplumber for best quality
pip install pdfplumber

# Or install via optional dependencies
pip install -e ".[pdf]"

# pypdf is included in project dependencies (installed automatically)
```

## Usage Examples

### Extracting Text from PDF

```python
from infrastructure.validation.pdf_validator import extract_text_from_pdf
from pathlib import Path

pdf_path = Path("data/pdfs/paper.pdf")
text = extract_text_from_pdf(pdf_path)
```

## Error Handling

```python
from infrastructure.validation.pdf_validator import (
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
- **pdfplumber** (recommended): Best quality extraction, tries first
- **pypdf**: Modern PyPDF2 replacement, tries second
- **PyPDF2**: Legacy library, tries third

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

**Note:** `pdfplumber` is now available as an optional dependency group `[pdf]` in `pyproject.toml`. 
Both `pdfplumber` and `pypdf` are installed and working in the current environment.

## See Also

- [`README.md`](README.md) - Quick reference
- [`pdf_validator.py`](pdf_validator.py) - Implementation

