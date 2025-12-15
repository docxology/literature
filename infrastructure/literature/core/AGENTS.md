# Core Module - Complete Documentation

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
All configuration options can be overridden via environment variables (see config.py for complete list).

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


