# Sources Module - Documentation

## Purpose

The sources module provides API adapters for multiple academic databases, normalizing results into a common `SearchResult` format.

## Available Sources

### ArxivSource (arxiv.py)
- Public API: http://export.arxiv.org/api/query
- Rate limit: 3 seconds between requests
- Features: Full text links, primary categories, DOI extraction

### SemanticScholarSource (semanticscholar.py)
- API: Graph API (https://api.semanticscholar.org/graph/v1)
- Auth: Optional API key for higher rate limits
- Rate limit: 1.5 seconds between requests

### UnpaywallSource (unpaywall.py)
- API: Unpaywall API (https://api.unpaywall.org/v2)
- Purpose: Open access PDF resolution
- Auth: Requires email address

### BiorxivSource (biorxiv.py)
- API: bioRxiv/medRxiv preprint API (https://api.biorxiv.org/)
- Rate limit: 1.0 seconds between requests (default)
- Features: Preprint metadata and PDFs, DOI-based search, title-based search with similarity matching, biology and medicine preprints

### PubMedSource (pubmed.py)
- API: PubMed E-utilities (https://eutils.ncbi.nlm.nih.gov/entrez/eutils)
- Rate limit: 0.34 seconds between requests (~3 requests/second, NCBI requirement)
- Features: Abstract extraction, DOI and PMC ID extraction, PDF links from PMC

### EuropePMCSource (europepmc.py)
- API: Europe PMC REST API (https://www.ebi.ac.uk/europepmc/webservices/rest/search)
- Rate limit: 0.5 seconds between requests (default)
- Features: Full-text access, open access PDF links, citation counts

### CrossRefSource (crossref.py)
- API: CrossRef Works API (https://api.crossref.org/works)
- Rate limit: 1.0 seconds between requests (default)
- Features: DOI resolution and metadata (authors, venues, dates), open access information

### OpenAlexSource (openalex.py)
- API: OpenAlex API (https://api.openalex.org/works)
- Rate limit: 0.5 seconds between requests (default)
- Features: Metadata, PDF links, citation counts, open access information (free API, no key required)

### DBLPSource (dblp.py)
- API: DBLP API (https://dblp.org/search/publ/api)
- Rate limit: 1.0 seconds between requests (default)
- Features: Computer science bibliography, author and venue information, metadata

## Base Classes

### LiteratureSource (base.py)
Abstract base class for all sources.

**Key Methods:**
- `search()` - Search for papers
- `check_health()` - Check source health
- `get_health_status()` - Get health status

### SearchResult (base.py)
Normalized search result dataclass.

**Attributes:**
- title, authors, year, abstract
- url, doi, pdf_url
- source, venue, citation_count

## Usage Examples

### Using a Source

```python
from infrastructure.literature.sources import ArxivSource

source = ArxivSource()
results = source.search("active inference", limit=10)

for result in results:
    print(f"{result.title} ({result.year})")
```

### Health Checking

```python
from infrastructure.literature.sources import SemanticScholarSource

source = SemanticScholarSource()
is_healthy = source.check_health()
status = source.get_health_status()
```

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


