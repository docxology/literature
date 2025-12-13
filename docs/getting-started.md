# Getting Started

Quick start guide for the Literature Search and Management System.

## Installation

### Prerequisites

- Python >= 3.10
- Ollama (optional, for AI summarization)
- Internet connection (for API searches)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd literature

# Install dependencies
uv sync
# or
pip install -e .
```

## Quick Start

### 1. Search for Papers

```bash
# Interactive search
python3 scripts/07_literature_search.py --search

# Search with keywords
python3 scripts/07_literature_search.py --search --keywords "machine learning,deep learning"
```

### 2. Download PDFs

PDFs are automatically downloaded when papers are added to the library. You can also download for existing entries:

```bash
python3 scripts/07_literature_search.py --download-only
```

### 3. Generate Summaries

```bash
# Generate summaries for existing PDFs
python3 scripts/07_literature_search.py --summarize
```

**Note:** Requires Ollama to be running. Start with `ollama serve`.

### 4. View Library

```bash
# Show library statistics
python3 -m infrastructure.literature.core.cli library stats

# List papers
python3 -m infrastructure.literature.core.cli library list
```

## Python API

### Basic Usage

```python
from infrastructure.literature import LiteratureSearch

# Initialize
searcher = LiteratureSearch()

# Search
papers = searcher.search("machine learning", limit=10)

# Process results
for paper in papers:
    print(f"{paper.title} ({paper.year})")
    
    # Add to library
    citation_key = searcher.add_to_library(paper)
    
    # Download PDF
    if paper.pdf_url:
        searcher.download_paper(paper)
```

### Using LLM

```python
from infrastructure.llm import LLMClient

# Initialize LLM client
client = LLMClient()

# Query
response = client.query("What is machine learning?")
print(response)
```

## Configuration

### Environment Variables

```bash
# Search settings
export LITERATURE_DEFAULT_LIMIT=25
export LITERATURE_SOURCES="arxiv,semanticscholar"

# LLM settings
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="gemma3:4b"

# Logging
export LOG_LEVEL=1  # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
```

## Output Files

All outputs are saved to the `data/` directory:

- `data/library.json` - Paper metadata index
- `data/references.bib` - BibTeX bibliography
- `data/pdfs/` - Downloaded PDFs
- `data/summaries/` - AI-generated summaries
- `data/output/` - Meta-analysis outputs

## Next Steps

- **[Architecture Overview](architecture.md)** - Understand system design
- **[Searching Papers](guides/search-papers.md)** - Detailed search guide
- **[Summarizing Papers](guides/summarize-papers.md)** - Summarization guide
- **[API Reference](reference/api-reference.md)** - Complete API documentation

## Troubleshooting

### Ollama Not Running

```bash
# Start Ollama
ollama serve

# Check status
ollama ps
```

### PDF Download Failures

Check `data/failed_downloads.json` for failed downloads and retry:

```bash
python3 scripts/07_literature_search.py --download-only
```

### Library Issues

```bash
# Validate library
python3 -m infrastructure.literature.core.cli library validate

# Clean up
python3 scripts/07_literature_search.py --cleanup
```

## See Also

- **[Architecture Overview](architecture.md)** - System architecture
- **[Configuration Guide](guides/configuration.md)** - Detailed configuration
- **[Module Documentation](../infrastructure/)** - Module-specific docs

