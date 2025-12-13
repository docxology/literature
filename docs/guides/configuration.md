# Configuration Guide

Complete guide to configuring the Literature Search and Management System.

## Configuration Methods

The system supports three configuration methods (in priority order):
1. **Environment variables** (highest priority)
2. **Configuration files** (YAML)
3. **Default values** (in code)

## Environment Variables

### Search Settings

```bash
# Default limit per source
export LITERATURE_DEFAULT_LIMIT=25

# Maximum total results
export LITERATURE_MAX_RESULTS=100

# Sources to use
export LITERATURE_SOURCES="arxiv,semanticscholar,pubmed"

# Search delays (seconds)
export LITERATURE_ARXIV_DELAY=3.0
export LITERATURE_SEMANTICSCHOLAR_DELAY=1.5
```

### PDF Settings

```bash
# Download directory
export LITERATURE_DOWNLOAD_DIR="data/pdfs"

# Parallel downloads (default: 4 workers)
export LITERATURE_MAX_PARALLEL_DOWNLOADS=4

# PDF download retry settings
export LITERATURE_DOWNLOAD_RETRY_ATTEMPTS=2
export LITERATURE_DOWNLOAD_RETRY_DELAY=2.0

# PDF download attempt limits (to prevent excessive retries)
export LITERATURE_MAX_URL_ATTEMPTS_PER_PDF=8
export LITERATURE_MAX_FALLBACK_STRATEGIES=3

# Use Unpaywall for open access
export LITERATURE_USE_UNPAYWALL=true
export UNPAYWALL_EMAIL=your@email.com
```

### LLM Settings

```bash
# Ollama connection
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="gemma3:4b"

# Generation defaults
export LLM_TEMPERATURE=0.7
export LLM_MAX_TOKENS=2048
export LLM_CONTEXT_WINDOW=131072
export LLM_TIMEOUT=60

# Summarization
export MAX_PARALLEL_SUMMARIES=1
export LLM_SUMMARIZATION_TIMEOUT=600
```

### Logging

```bash
# Log level (0=DEBUG, 1=INFO, 2=WARN, 3=ERROR)
export LOG_LEVEL=1

# Disable emoji output
export NO_EMOJI=1
```

## Programmatic Configuration

### Literature Config

```python
from infrastructure.literature import LiteratureConfig

config = LiteratureConfig(
    default_limit=50,
    sources=["arxiv", "semanticscholar"],
    arxiv_delay=2.0,
    download_dir="custom/pdfs"
)

# Or load from environment
config = LiteratureConfig.from_env()
```

### LLM Config

```python
from infrastructure.llm import LLMConfig

config = LLMConfig(
    base_url="http://localhost:11434",
    default_model="gemma3:4b",
    temperature=0.7,
    max_tokens=2048
)

# Or load from environment
config = LLMConfig.from_env()
```

## Configuration Files

### .env File

Create a `.env` file in the repository root:

```env
LITERATURE_DEFAULT_LIMIT=25
LITERATURE_SOURCES=arxiv,semanticscholar
LITERATURE_USE_UNPAYWALL=true
UNPAYWALL_EMAIL=your@email.com
OLLAMA_MODEL=gemma3:4b
LLM_TEMPERATURE=0.7
LOG_LEVEL=1
```

## Source-Specific Configuration

### API Keys

Some sources require API keys:

```bash
# Semantic Scholar (optional, for higher rate limits)
export SEMANTICSCHOLAR_API_KEY=your-api-key
```

### Rate Limits

Sources have different rate limits:
- **arXiv**: 3 seconds between requests
- **Semantic Scholar**: 1.5 seconds (longer with API key)
- **PubMed**: ~3 requests/second
- **CrossRef**: Varies by plan

## See Also

- **[API Reference](../reference/api-reference.md)** - Configuration API documentation
- **[Literature Config](../infrastructure/literature/core/config.py)** - Config implementation

