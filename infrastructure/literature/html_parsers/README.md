# HTML Parsers Module

Publisher-specific HTML parsers for PDF URL extraction.

## Components

- **base.py**: Base parser class
- **osf.py**: OSF.io (Open Science Framework) parser
- **elsevier.py**: Elsevier/ScienceDirect parser
- **springer.py**: Springer parser
- **ieee.py**: IEEE parser
- **acm.py**: ACM parser
- **wiley.py**: Wiley parser
- **generic.py**: Generic fallback parser

## Quick Start

```python
from infrastructure.literature.html_parsers import extract_pdf_urls_modular

urls = extract_pdf_urls_modular(html_content, base_url)
```

## Features

- Publisher-specific parsing (OSF.io, Elsevier, Springer, IEEE, ACM, Wiley)
- Automatic parser selection by URL pattern
- OSF.io direct download URL extraction
- Generic fallback for unknown publishers

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


