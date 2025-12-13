# PDF Module

PDF handling, downloading, extraction, and fallback strategies.

## Components

- **handler.py**: Main `PDFHandler` class for downloading PDFs
- **downloader.py**: PDF download implementation with retry logic
- **extractor.py**: Text extraction and HTML parsing utilities
- **fallbacks.py**: Fallback strategies for PDF URL resolution
- **html_extractor.py**: HTML-to-text extraction for fallback when PDFs unavailable

## Quick Start

```python
from infrastructure.literature.pdf import PDFHandler
from infrastructure.literature.core import LiteratureConfig

config = LiteratureConfig()
handler = PDFHandler(config)

# Download PDF
pdf_path = handler.download_paper(search_result)
```

## Features

- Automatic retry with exponential backoff
- User-Agent rotation for downloads
- HTML-to-PDF URL extraction
- Multiple fallback strategies (Unpaywall, arXiv, bioRxiv, OSF.io)
- **HTML text extraction fallback**: When PDFs are unavailable, extracts text from HTML pages
- OSF.io (Open Science Framework) support with direct download URL transformation
- Enhanced publisher-specific URL patterns (MDPI, IEEE, Preprints.org)
- Citation key-based file naming

## See Also

- [`AGENTS.md`](AGENTS.md) - Complete documentation
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


