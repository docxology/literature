# PDF Module - Documentation

## Purpose

The PDF module handles all aspects of PDF downloading, text extraction, and URL resolution with fallback strategies.

## Components

### PDFHandler (handler.py)

Main interface for PDF downloading and management.

**Key Methods:**
- `download_paper()` - Download PDF for a search result
- `download_pdf()` - Download PDF from URL with citation key naming

**Features:**
- Automatic retry with exponential backoff
- User-Agent rotation (limited to 2 attempts)
- Multiple fallback strategies (configurable limits)
- Citation key-based file naming
- Configurable retry limits to prevent excessive attempts

### PDFDownloader (downloader.py)

Low-level PDF download implementation.

**Features:**
- Content type validation
- HTML response detection
- Recursive HTML parsing (limited depth)
- Error categorization
- Optimized retry logic with configurable limits
- Early exit for clearly unavailable URLs (404)
- Smart detection of when to stop retrying

### PDF Extractor (extractor.py)

Text extraction and HTML parsing utilities.

**Functions:**
- `extract_pdf_urls_from_html()` - Extract PDF URLs from HTML content
- `extract_citations()` - Extract citations from PDF (placeholder)

### PDF Fallbacks (fallbacks.py)

Fallback strategies for PDF URL resolution.

**Strategies:**
- URL transformation (PMC, arXiv, bioRxiv, OSF.io patterns)
- Unpaywall lookup
- DOI-based URL generation
- Publisher-specific patterns (MDPI, IEEE, Preprints.org, OSF.io)

**OSF.io Support:**
- Automatic detection of OSF.io DOIs (`10.31234/osf.io/...` or `10.31219/osf.io/...`)
- Direct download URL transformation: `https://osf.io/{project_id}/download`
- MDPI URL patterns with case-insensitive DOI handling
- IEEE Xplore and Preprints.org URL pattern improvements

### HTML Text Extractor (html_extractor.py)

Fallback text extraction when PDFs are unavailable.

**Features:**
- Extracts main content from HTML pages
- Removes navigation, headers, footers, scripts, and styles
- Preserves document structure (headings, paragraphs)
- Uses BeautifulSoup4 when available, falls back to regex parsing
- Validates extracted text to ensure it's a full paper (not just headers/footers)
- Saves extracted text as `.txt` files in `data/pdfs/` (only if validation passes)

**Validation:**
The `is_valid_paper_content()` method validates extracted text by checking:
- Minimum length (configurable via `html_text_min_length`, default: 2000 characters)
- Presence of common academic paper sections (abstract, introduction, methods, etc.)
- Absence of excessive repetition (indicates navigation/header text)

**Usage:**
Automatically used by `PDFHandler` when PDF download fails but HTML content is available. Only saves text files that pass validation.

### Failed Download Tracker (failed_tracker.py)

Tracks failed PDF download attempts for retry capability and automatic skip behavior.

**Class:** `FailedDownloadTracker`

**Key Methods:**
- `save_failed()` - Save a failed download to the tracker
- `load_failed()` - Load all failed downloads
- `get_retriable_failed()` - Get only retriable failed downloads (network errors, timeouts)
- `is_failed()` - Check if a citation key has a failed download
- `has_failures()` - Check if tracker has any failed downloads
- `remove_successful()` - Remove a citation key from tracker when download succeeds
- `clear_failed()` - Clear failed downloads from tracker

**Features:**
- Automatic tracking of all download failures
- Retriable detection (network errors, timeouts are retriable; access denied, not found are not)
- Persistent storage in `data/failed_downloads.json`
- Atomic file writes for data integrity
- Automatic cleanup when downloads succeed

**Failure Categories:**
- `network_error` - Connection errors (retriable)
- `timeout` - Request timeout (retriable)
- `access_denied` - HTTP 403 Forbidden (not retriable)
- `not_found` - HTTP 404 Not Found (not retriable)
- `html_response` - HTML received instead of PDF (not retriable)
- `no_pdf_url` - No PDF URL available (not tracked, just a warning)

**File Format:**
The tracker saves failures to `data/failed_downloads.json`:
```json
{
  "version": "1.0",
  "updated": "2025-12-13T14:21:29.308815",
  "failures": {
    "citation_key": {
      "citation_key": "citation_key",
      "title": "Paper Title",
      "failure_reason": "access_denied",
      "failure_message": "Detailed error message",
      "attempted_urls": ["url1", "url2"],
      "source": "arxiv",
      "timestamp": "2025-12-13T14:21:29.308815",
      "retriable": false
    }
  }
}
```

**Usage:**
```python
from infrastructure.literature.pdf.failed_tracker import FailedDownloadTracker
from infrastructure.literature.core.config import LiteratureConfig

config = LiteratureConfig()
tracker = FailedDownloadTracker(config)

# Save a failed download
tracker.save_failed(citation_key, download_result, title="Paper Title", source="arxiv")

# Check if a download previously failed
if tracker.is_failed(citation_key):
    print(f"Download for {citation_key} previously failed")

# Get retriable failures for retry
retriable = tracker.get_retriable_failed()

# Remove successful download from tracker
tracker.remove_successful(citation_key)
```

**Integration:**
The tracker is integrated into all download operations:
- `workflow.py` - `_download_papers_sequential` and `_download_papers_parallel`
- `meta_analysis.py` - Regular downloads and retry downloads
- `download.py` - Regular downloads and retry downloads

All operations automatically save failures and skip previously failed downloads by default (unless `retry_failed=True`).

## Usage Examples

### Basic Download

```python
from infrastructure.literature.pdf import PDFHandler
from infrastructure.literature.core import LiteratureConfig

config = LiteratureConfig()
handler = PDFHandler(config)

pdf_path = handler.download_paper(search_result)
```

### Fallback Strategies

The handler automatically tries multiple strategies:
1. Primary URL from search result
2. Transformed URLs (PMC variants, OSF.io, arXiv, etc.)
3. Unpaywall lookup
4. arXiv title search
5. bioRxiv/medRxiv DOI lookup
6. **HTML text extraction** (when PDF unavailable but HTML is accessible)

## Error Handling

PDF downloads categorize failures:
- `access_denied` - HTTP 403 Forbidden (tries multiple User-Agents and referers)
- `not_found` - HTTP 404 Not Found (early exit, no retries)
- `html_response` - HTML received instead of PDF (tries transformed URLs, then HTML text extraction)
- `html_no_pdf_link` - HTML page contains no working PDF URLs (tries HTML text extraction)
- `timeout` - Request timeout
- `network_error` - Connection errors

**HTML Text Extraction Fallback:**
When PDF downloads fail with `html_response` or `html_no_pdf_link`, the system automatically attempts to extract text content from the HTML page. The extracted text is validated to ensure it contains a full academic paper (not just webpage headers/footers) before being saved.

**Validation Criteria:**
- Minimum length: Extracted text must be at least `html_text_min_length` characters (default: 2000)
- Content quality: Must contain at least one common academic paper section (abstract, introduction, methods, results, discussion, conclusion, references, etc.)
- Repetition check: Rejects text with excessive repetition of short phrases (likely navigation/header text)

If validation passes, the extracted text is saved as a `.txt` file in `data/pdfs/` with the same naming convention as PDFs (citation key). If validation fails, a warning is logged explaining why (too short, missing sections, etc.) and the file is not saved. This prevents saving incomplete extractions that contain only webpage headers/footers.

Error messages include helpful context and troubleshooting suggestions.

## Configuration

### Retry Limits

To prevent excessive retry attempts, configure:

```python
config = LiteratureConfig(
    max_url_attempts_per_pdf=8,  # Maximum total URL attempts per PDF (default: 8)
    max_fallback_strategies=3,   # Maximum fallback strategy attempts (default: 3)
)
```

Or via environment variables:
```bash
export LITERATURE_MAX_URL_ATTEMPTS_PER_PDF=8
export LITERATURE_MAX_FALLBACK_STRATEGIES=3
export LITERATURE_HTML_TEXT_MIN_LENGTH=2000  # Minimum characters for HTML text extraction (default: 2000)
```

### Retry Behavior

The downloader uses optimized retry logic:
- **Standard download**: 1 attempt
- **Transformed URLs**: Up to 2 variants (for HTML responses)
- **User-Agent rotation**: Up to 2 different User-Agents
- **Minimal headers**: 1 attempt
- **HEAD request**: Skipped for persistent 403 errors
- **Referer spoofing**: 1 attempt (Google referer)
- **Academic referers**: 1 attempt (Scholar/Semantic Scholar)
- **Standard retries**: Up to `download_retry_attempts` (default: 2)

Total attempts are limited by `max_url_attempts_per_pdf` (default: 8).

## Troubleshooting

### PDF Download Failures

**Common Issues:**

1. **403 Forbidden errors**
   - The server may be blocking automated requests
   - Try accessing the URL manually in a browser
   - Check if the PDF requires authentication
   - The downloader automatically tries multiple User-Agents and referers

2. **HTML received instead of PDF**
   - The URL points to a web page, not a direct PDF link
   - The downloader automatically tries transformed URLs
   - Some publishers require clicking a download button (not supported)

3. **404 Not Found**
   - The PDF URL does not exist or has been moved
   - The downloader exits early for 404 errors (no retries)
   - Check the source URL or try alternative sources

4. **Excessive retry attempts**
   - Configure `max_url_attempts_per_pdf` to limit total attempts
   - Default is 8 attempts per PDF
   - Reduce if downloads are taking too long

### Missing PDF Libraries

If you see "No PDF parsing library available":
- Install one of: `pdfplumber`, `pypdf`, or `PyPDF2`
- `pypdf>=5.0` should be installed automatically with the project
- If missing, try: `pip install -e .` or `pip install pypdf`
- For best quality, install: `pip install pdfplumber`

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


