# Core Module - Documentation

## Purpose

The core module provides the main entry point for literature search functionality, configuration management, and command-line interface.

## Components

### LiteratureSearch (core.py)

Main class for searching papers across multiple sources, downloading PDFs, and managing library entries.

**Key Methods:**
- `search(query, limit=10, sources=None, return_stats=False)` - Search across enabled sources. Returns list of SearchResult or tuple of (results, SearchStatistics) if return_stats=True
- `download_paper(result)` - Download PDF for a search result
- `download_paper_with_result(result)` - Download PDF with detailed result tracking (returns DownloadResult)
- `add_to_library(result)` - Add paper to BibTeX and JSON index, returns citation key
- `export_library(path=None, format="json")` - Export library to JSON file
- `get_library_stats()` - Get library statistics
- `get_library_entries()` - Get all entries in the library as dictionaries
- `get_source_health_status()` - Check health of all sources (returns detailed status dict)
- `check_all_sources_health()` - Check health of all sources (returns boolean dict)
- `remove_paper(citation_key)` - Remove a paper from the library by citation key
- `cleanup_library(remove_missing_pdfs=True)` - Clean up library by removing entries without PDFs

### LiteratureConfig (config.py)

Configuration dataclass with environment variable support.

**Configuration Options:**
- Search settings (default_limit, max_results)
- API settings (delays, retry attempts)
- File paths (download_dir, bibtex_file, library_index_file)
- Source selection (sources list)

**Environment Variables:**

All configuration options can be overridden via environment variables. Key variables:

**Search Settings:**
- `LITERATURE_DEFAULT_LIMIT` - Results per source per search (default: 25)
- `LITERATURE_MAX_RESULTS` - Maximum total results (default: 100)
- `LITERATURE_SOURCES` - Comma-separated sources (default: arxiv,semanticscholar)
- `LITERATURE_ARXIV_DELAY` - Seconds between arXiv requests (default: 3.0)
- `LITERATURE_SEMANTICSCHOLAR_DELAY` - Seconds between Semantic Scholar requests (default: 1.5)
- `SEMANTICSCHOLAR_API_KEY` - Semantic Scholar API key (optional)

**Request Settings:**
- `LITERATURE_RETRY_ATTEMPTS` - Retry attempts for failed requests (default: 3)
- `LITERATURE_RETRY_DELAY` - Base delay for exponential backoff (default: 5.0)
- `LITERATURE_TIMEOUT` - Request timeout in seconds (default: 30.0)

**File Paths:**
- `LITERATURE_DOWNLOAD_DIR` - PDF download directory (default: data/pdfs)
- `LITERATURE_BIBTEX_FILE` - BibTeX file path (default: data/references.bib)
- `LITERATURE_LIBRARY_INDEX` - JSON index file path (default: data/library.json)

**PDF Download Settings:**
- `LITERATURE_PDF_DOWNLOAD_TIMEOUT` - PDF download timeout (default: 60.0)
- `LITERATURE_DOWNLOAD_RETRY_ATTEMPTS` - Retry attempts for PDF downloads (default: 2)
- `LITERATURE_DOWNLOAD_RETRY_DELAY` - Base delay for download retry (default: 2.0)
- `LITERATURE_MAX_PARALLEL_DOWNLOADS` - Maximum parallel download workers (default: 4)
- `LITERATURE_MAX_URL_ATTEMPTS_PER_PDF` - Maximum URL attempts per PDF (default: 8)
- `LITERATURE_MAX_FALLBACK_STRATEGIES` - Maximum fallback strategy attempts (default: 3)
- `LITERATURE_USE_BROWSER_USER_AGENT` - Use browser User-Agent for downloads (default: true)
- `LITERATURE_HTML_TEXT_MIN_LENGTH` - Minimum HTML text length for extraction (default: 2000)

**Unpaywall Settings:**
- `LITERATURE_USE_UNPAYWALL` - Enable Unpaywall fallback (default: true)
- `UNPAYWALL_EMAIL` - Email for Unpaywall API (required if enabled)

**Embedding Settings:**
- `LITERATURE_EMBEDDING_MODEL` - Embedding model name (default: embeddinggemma)
- `LITERATURE_EMBEDDING_DIMENSION` - Embedding vector dimension (default: 768)
- `LITERATURE_EMBEDDING_CACHE_DIR` - Embedding cache directory (default: data/embeddings)
- `LITERATURE_EMBEDDING_CHUNK_SIZE` - Text chunk size for embeddings (default: 2000)
- `LITERATURE_EMBEDDING_BATCH_SIZE` - Batch size for embedding generation (default: 10)
- `LITERATURE_EMBEDDING_TIMEOUT` - Embedding request timeout (default: 120.0)
- `LITERATURE_EMBEDDING_RETRY_ATTEMPTS` - Retry attempts for embedding requests (default: 3)
- `LITERATURE_EMBEDDING_RETRY_DELAY` - Base delay for embedding retry (default: 2.0)
- `LITERATURE_EMBEDDING_RESTART_OLLAMA_ON_TIMEOUT` - Restart Ollama on timeout (default: true)
- `LITERATURE_EMBEDDING_FORCE_RESTART_ON_TIMEOUT` - Force restart Ollama on timeout (default: true)
- `LITERATURE_EMBEDDING_TEST_ENDPOINT_ON_RESTART` - Test embedding endpoint after restart (default: true)
- `LITERATURE_EMBEDDING_MAX_TEXT_LENGTH` - Maximum text length for embeddings (default: 250000)
- `LITERATURE_EMBEDDING_TIMEOUT_MULTIPLIER_FOR_LONG_DOCS` - Timeout multiplier for long documents (default: 2.0)
- `LITERATURE_EMBEDDING_CHUNK_SIZE_REDUCTION_THRESHOLD` - Chunk size reduction threshold (default: 100000)

See `config.py` for complete list of all configuration options.

### CLI (cli.py)

Command-line interface for interactive literature management.

**Commands:**
- `search` - Search for papers
- `library list` - List papers in library
- `library stats` - Show library statistics
- `library export` - Export library to JSON

## Usage Examples

### Basic Search

```python
from infrastructure.literature.core import LiteratureSearch

searcher = LiteratureSearch()
results = searcher.search("active inference", limit=5)
```

### Custom Configuration

```python
from infrastructure.literature.core import LiteratureConfig, LiteratureSearch

config = LiteratureConfig(
    default_limit=50,
    sources=["arxiv", "semanticscholar"],
    arxiv_delay=2.0
)
searcher = LiteratureSearch(config)
```

### Environment Configuration

```bash
export LITERATURE_DEFAULT_LIMIT=50
export LITERATURE_SOURCES=arxiv,semanticscholar
python3 -m infrastructure.literature.core.cli search "query"
```

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


