# Configuration Guide

Guide to configuring the Literature Search and Management System.

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

# Retry settings for API requests
export LITERATURE_RETRY_ATTEMPTS=3
export LITERATURE_RETRY_DELAY=5.0

# Request timeout (seconds)
export LITERATURE_TIMEOUT=30.0

# User agent string for API requests
export LITERATURE_USER_AGENT="Research-Template-Bot/1.0 (mailto:admin@example.com)"
```

### PDF Settings

```bash
# Download directory
export LITERATURE_DOWNLOAD_DIR="data/pdfs"

# Parallel downloads (default: 4 workers)
export LITERATURE_MAX_PARALLEL_DOWNLOADS=4

# PDF download timeout (seconds, larger files need more time)
export LITERATURE_PDF_DOWNLOAD_TIMEOUT=60.0

# PDF download retry settings
export LITERATURE_DOWNLOAD_RETRY_ATTEMPTS=2
export LITERATURE_DOWNLOAD_RETRY_DELAY=2.0

# PDF download attempt limits (to prevent excessive retries)
export LITERATURE_MAX_URL_ATTEMPTS_PER_PDF=8
export LITERATURE_MAX_FALLBACK_STRATEGIES=3

# Use browser-like User-Agent for downloads (helps avoid 403 errors)
export LITERATURE_USE_BROWSER_USER_AGENT=true

# HTML text extraction validation
export LITERATURE_HTML_TEXT_MIN_LENGTH=2000  # Minimum characters for extracted HTML text to be considered valid

# Use Unpaywall for open access
export LITERATURE_USE_UNPAYWALL=true
export UNPAYWALL_EMAIL=your@email.com
```

### File Path Settings

```bash
# BibTeX bibliography file
export LITERATURE_BIBTEX_FILE="data/references.bib"

# JSON library index file
export LITERATURE_LIBRARY_INDEX="data/library.json"
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
export LLM_NUM_CTX=131072  # Alternative name for context_window (Ollama parameter)
export LLM_TIMEOUT=60
export LLM_SEED=42  # Optional: seed for reproducibility

# Response length settings
export LLM_LONG_MAX_TOKENS=16384  # Maximum tokens for long responses

# Summarization
export MAX_PARALLEL_SUMMARIES=1
export LLM_SUMMARIZATION_TIMEOUT=600

# Embedding settings (for meta-analysis with embeddings)
export LITERATURE_EMBEDDING_MODEL="embeddinggemma"  # Embedding model name (default: embeddinggemma)
export LITERATURE_EMBEDDING_DIMENSION=768  # Embedding dimension (default: 768)
export LITERATURE_EMBEDDING_TIMEOUT=120.0  # Timeout for embedding requests (longer than LLM timeout for large texts)
export LITERATURE_EMBEDDING_RETRY_ATTEMPTS=3  # Number of retry attempts for failed embedding requests
export LITERATURE_EMBEDDING_RETRY_DELAY=2.0  # Initial delay between retries (exponential backoff)
export LITERATURE_EMBEDDING_RESTART_OLLAMA_ON_TIMEOUT=true  # Whether to attempt Ollama restart on timeout
export LITERATURE_EMBEDDING_FORCE_RESTART_ON_TIMEOUT=true  # Whether to force kill hung Ollama processes on timeout
export LITERATURE_EMBEDDING_TEST_ENDPOINT_ON_RESTART=true  # Whether to test embedding endpoint when restarting Ollama
export LITERATURE_EMBEDDING_CACHE_DIR="data/embeddings"  # Directory for caching embeddings
export LITERATURE_EMBEDDING_CHUNK_SIZE=2000  # Maximum tokens per chunk for text splitting
export LITERATURE_EMBEDDING_BATCH_SIZE=10  # Number of texts to process in each batch
export LITERATURE_EMBEDDING_MAX_TEXT_LENGTH=250000  # Maximum character length for documents to embed (documents exceeding this are skipped)
export LITERATURE_EMBEDDING_TIMEOUT_MULTIPLIER_FOR_LONG_DOCS=2.0  # Timeout multiplier for long documents (default: 2.0)
export LITERATURE_EMBEDDING_CHUNK_SIZE_REDUCTION_THRESHOLD=100000  # Character threshold for reducing chunk size (default: 100000)
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

Sources have different rate limits (default delays in seconds):
- **arXiv**: 3.0 seconds between requests
- **Semantic Scholar**: 1.5 seconds (longer with API key)
- **PubMed**: 0.34 seconds (~3 requests/second, NCBI requirement)
- **Europe PMC**: 0.5 seconds between requests
- **CrossRef**: 1.0 seconds between requests
- **OpenAlex**: 0.5 seconds between requests
- **DBLP**: 1.0 seconds between requests
- **bioRxiv/medRxiv**: 1.0 seconds between requests

These delays are configurable via source-specific settings in `LiteratureConfig.source_configs`.

## See Also

- **[API Reference](../reference/api-reference.md)** - Configuration API documentation
- **[Module Documentation](../modules/)** - Module documentation
- **[Guides AGENTS.md](AGENTS.md)** - Guide organization and standards

