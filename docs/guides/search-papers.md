# Searching Papers Guide

Guide to searching for academic papers using the Literature Search system.

## Quick Start

### Command Line

```bash
# Interactive search
python3 scripts/literature_search.py --search

# Search with keywords
python3 scripts/literature_search.py --search --keywords "machine learning,deep learning"
```

### Python API

```python
from infrastructure.literature import LiteratureSearch

searcher = LiteratureSearch()
papers = searcher.search("machine learning", limit=10)
```

## Search Sources

The system supports multiple academic databases:

- **arXiv** - Preprints in physics, mathematics, computer science
- **Semantic Scholar** - AI-powered research tool
- **PubMed** - Biomedical literature (NCBI E-utilities)
- **CrossRef** - DOI resolution and metadata
- **OpenAlex** - Open scholarly metadata (free API)
- **DBLP** - Computer science bibliography
- **bioRxiv/medRxiv** - Biology and medicine preprints
- **Europe PMC** - European biomedical literature
- **Unpaywall** - Open access PDF resolution (lookup-only, requires email)

### Configuring Sources

```bash
# Set sources via environment variable
export LITERATURE_SOURCES="arxiv,semanticscholar,pubmed"
```

```python
# Set sources programmatically
from infrastructure.literature import LiteratureConfig, LiteratureSearch

config = LiteratureConfig(sources=["arxiv", "semanticscholar"])
searcher = LiteratureSearch(config)
```

## Search Options

### Limit Results

```python
# Limit per source
papers = searcher.search("query", limit=25)

# Limit via environment
export LITERATURE_DEFAULT_LIMIT=25
```

### Source Selection

```python
# Use specific sources
papers = searcher.search(
    "query",
    sources=["arxiv", "semanticscholar"]
)
```

### Statistics

```python
# Get search statistics
papers, stats = searcher.search(
    "query",
    return_stats=True
)

print(f"Total results: {stats.total_results}")
print(f"Sources used: {stats.sources_used}")
```

## Processing Results

### Add to Library

```python
for paper in papers:
    citation_key = searcher.add_to_library(paper)
    print(f"Added: {citation_key}")
```

### Download PDFs

```python
for paper in papers:
    if paper.pdf_url:
        result = searcher.download_paper(paper)
        if result:
            print(f"Downloaded: {result}")
```

### Workflow

```python
papers = searcher.search("active inference", limit=10)

for paper in papers:
    # Add to library
    citation_key = searcher.add_to_library(paper)
    
    # Download PDF
    if paper.pdf_url:
        pdf_path = searcher.download_paper(paper)
        print(f"Downloaded: {pdf_path}")
```

## Deduplication

The system automatically deduplicates results:
- Exact DOI matching (highest priority)
- Fuzzy title similarity matching
- Relevance ranking to prioritize results

## Health Checking

Check source availability before searching:

```python
health_status = searcher.get_source_health_status()
for source, status in health_status.items():
    print(f"{source}: {'healthy' if status['healthy'] else 'unhealthy'}")
```

## Best Practices

1. **Use specific keywords** - More specific queries yield more relevant results
2. **Combine sources** - Use multiple sources for coverage
3. **Check source health** - Verify sources are available before large searches
4. **Limit results** - Use appropriate limits to avoid overwhelming results
5. **Review before adding** - Check results before adding to library

## Troubleshooting

### No Results

- Check source health status
- Verify internet connection
- Try different keywords
- Check source-specific requirements (API keys, etc.)

### Slow Searches

- Reduce limit per source
- Use fewer sources
- Check network connectivity
- Verify source rate limits

### PDF Download Failures

The system automatically tracks failed PDF downloads and **skips them by default** in subsequent runs to avoid repeated failures on access-restricted papers.

**Understanding Failed Downloads:**

- Failed downloads are tracked in `data/failed_downloads.json`
- Each failure includes: citation key, failure reason, attempted URLs, timestamp, and retriable status
- Common failure reasons:
  - `access_denied` - PDF requires authentication or subscription (typically not retriable)
  - `network_error` - Network connectivity issue (retriable)
  - `timeout` - Request timed out (retriable)
  - `not_found` - PDF URL does not exist (not retriable)
  - `html_response` - Received HTML instead of PDF (not retriable)

**Retrying Failed Downloads:**

To retry previously failed downloads, use the `--retry-failed` flag:

```bash
# Retry failed downloads
python3 scripts/literature_search.py --search --retry-failed --keywords "machine learning"

# Or for download-only operation
python3 scripts/literature_search.py --download-only --retry-failed
```

**Best Practices:**

- Use `--retry-failed` for network errors or timeouts (retriable failures)
- Skip retrying `access_denied` failures unless you have institutional access
- Check `data/failed_downloads.json` to understand why downloads failed
- Manually download access-restricted papers if you have institutional access

## See Also

- **[API Reference](../reference/api-reference.md)** - API documentation
- **[Literature Module Documentation](../modules/literature.md)** - Module documentation
- **[Guides AGENTS.md](AGENTS.md)** - Guide organization and standards

