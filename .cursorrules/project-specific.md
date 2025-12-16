# Project-Specific Rules

## Literature Search and Management System

### Standalone Repository Design

#### Independence Requirements
- This repository is **completely independent and self-contained**
- No dependencies on external template or manuscript systems
- Duplicated shared infrastructure (`infrastructure/core/`, `infrastructure/llm/`) for independence
- Separate bibliography (`data/references.bib`) from any manuscript system
- Complete test suite for all functionality

#### Self-Contained Architecture
- All required modules must be duplicated within this repository
- Avoid external dependencies on template or manuscript systems
- Maintain complete independence from other systems
- Ensure all functionality is testable within this repository

## Directory Structure Conventions

### Root Level
- `run_literature.sh` - Main orchestrator with interactive menu
- `scripts/` - Orchestrator scripts (e.g., `literature_search.py`, `bash_utils.sh`)
- `infrastructure/` - Core modules (core, llm, literature)
- `tests/` - Test suite mirroring infrastructure structure
- `data/` - All data files (library.json, references.bib, pdfs/, summaries/, etc.)
- `docs/` - Documentation files

### Infrastructure Organization
- `infrastructure/core/` - Foundation utilities (duplicated for independence)
- `infrastructure/llm/` - LLM integration (duplicated for independence)
- `infrastructure/literature/` - Literature-specific modules
  - `core/` - Core search functionality
  - `sources/` - API adapters (arXiv, Semantic Scholar, PubMed, etc.)
  - `pdf/` - PDF handling
  - `library/` - Library management
  - `workflow/` - Workflow orchestration
  - `summarization/` - AI summarization
  - `meta_analysis/` - Meta-analysis tools

### Data Directory Structure
- `data/library.json` - Paper metadata index (JSON format)
- `data/references.bib` - BibTeX bibliography (separate from manuscript systems)
- `data/summarization_progress.json` - Summarization progress tracking (auto-generated)
- `data/failed_downloads.json` - Failed download tracking (auto-generated)
- `data/pdfs/` - Downloaded PDFs (named by citation key)
- `data/summaries/` - AI-generated summaries
- `data/extracted_text/` - Extracted PDF text
- `data/embeddings/` - Cached embedding files (JSON) for semantic analysis
- `data/output/` - Meta-analysis outputs and visualizations
- `literature/llm_outputs/` - Advanced LLM operation results (created at repo root, not in data/)

## Bibliography Management

### Bibliography Independence
- Bibliography (`data/references.bib`) is **separate** from any manuscript system
- Maintained independently in this repository
- Can be manually copied to manuscript systems if needed
- No automatic synchronization with external systems

### BibTeX Generation
- Generate BibTeX entries automatically from library entries
- Use consistent citation key format (e.g., `smith2024machine`)
- Include all relevant metadata (title, authors, year, DOI, URL, venue)
- Ensure BibTeX entries are valid and complete

### Library Index
- Maintain JSON-based library index in `data/library.json`
- Track complete paper metadata
- Include citation keys, PDF paths, added dates
- Support library queries and updates

## PDF Management

### PDF Naming Convention
- PDFs must be named by citation key (e.g., `smith2024machine.pdf`)
- Store all PDFs in `data/pdfs/` directory
- Maintain consistency between library entries and PDF filenames
- Handle PDF naming conflicts appropriately

### PDF Download
- Implement automatic PDF retrieval with retry logic
- Use Unpaywall for open access fallback when available
- Handle download failures gracefully

### Failed Downloads Tracking

**Automatic Failure Tracking:**
- All download failures are automatically saved to `data/failed_downloads.json`
- Failures are tracked in all download operations:
  - Workflow sequential and parallel downloads
  - Meta-analysis pipeline downloads
  - Download-only operation downloads
- Exceptions: "no_pdf_url" failures are warnings, not tracked as failures

**Default Skip Behavior:**
- By default, previously failed downloads are automatically skipped
- Skip happens in `find_papers_needing_pdf()` function
- Skip message: "Skipped X paper(s) with previously failed downloads (use --retry-failed to retry)"
- This prevents wasting time on papers that are likely to fail again (e.g., access-restricted papers)

**Retry Mechanism:**
- Use `retry_failed=True` parameter or `--retry-failed` flag to retry previously failed downloads
- Only retriable failures (network errors, timeouts) are retried by default
- All failures can be retried if explicitly requested
- Successful retries automatically remove entries from the tracker

**Failure Categories:**
- **Retriable**: `network_error`, `timeout` (may succeed on retry)
- **Not Retriable**: `access_denied`, `not_found`, `html_response` (unlikely to succeed)
- **Not Tracked**: `no_pdf_url` (just a warning, not a failure)

**File Format (`data/failed_downloads.json`):**
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

**Integration:**
- All download operations use `workflow.failed_tracker.save_failed()` to track failures
- Operations check `workflow.failed_tracker.is_failed()` to skip previously failed downloads
- The `FailedDownloadTracker` class is in `infrastructure/literature/pdf/failed_tracker.py`

### PDF Processing
- Extract text from PDFs for summarization
- Store extracted text in `data/extracted_text/` directory
- Use consistent text extraction methods
- Handle extraction errors appropriately

## Literature Search

### Multi-Source Search
- Support multiple search sources (arXiv, Semantic Scholar, PubMed, CrossRef, etc.)
- Implement unified search interface across sources
- Handle source-specific API differences
- Provide consistent result format

### Deduplication
- Implement automatic deduplication of search results
- Use DOI, title, and author matching for deduplication
- Handle variations in metadata across sources
- Maintain single source of truth for each paper

### Search Configuration
- Use environment variables for search configuration
- Support configurable search limits and sources
- Allow customization of search parameters
- Document all configuration options

## AI Summarization

### LLM Integration
- Use local LLM (Ollama) for paper summarization
- Require Ollama server to be running
- Support configurable model selection
- Handle LLM timeouts and errors gracefully

### Summary Generation
- Generate markdown summaries in `data/summaries/` directory
- Use citation key for summary filenames (e.g., `smith2024machine_summary.md`)
- Include quality validation for summaries
- Support batch summarization with progress tracking

### Summary Quality
- Validate summary completeness and accuracy
- Ensure summaries are informative and useful
- Handle summarization failures appropriately
- Provide feedback on summary quality

## Meta-Analysis

### Analysis Tools
- Implement PCA analysis of paper metadata
- Support keyword evolution analysis
- Provide author contribution analysis
- Generate visualizations for analysis results

### Output Management
- Save all meta-analysis outputs to `data/output/` directory
- Generate visualizations (PNG, PDF formats)
- Create summary reports (JSON, Markdown)
- Organize outputs by analysis type

### Visualization
- Generate publication trends visualizations
- Create keyword frequency charts
- Produce author network visualizations
- Support multiple visualization formats

## Workflow Orchestration

### Script Organization
- Use `scripts/` directory for orchestrator scripts
- Implement interactive menu in `run_literature.sh`
- Support both interactive and command-line workflows
- Provide clear workflow options and feedback

### Workflow Operations
- Support full pipeline (search → download → extract → summarize)
- Allow individual operations (search-only, download-only, etc.)
- Implement workflow state tracking
- Provide progress feedback for long-running operations

## Configuration Management

### Environment Variables
- Use environment variables for configuration
- Support `.env` file for local configuration
- Document all configuration options
- Provide sensible defaults

### Configuration Priority
- Environment variables override file configuration
- Support configuration validation
- Provide clear error messages for invalid configuration
- Document configuration precedence

## Testing Requirements

### Test Organization
- Mirror infrastructure structure in tests
- Test all literature-specific functionality
- Include tests for core and LLM modules (filtered as needed)
- Maintain comprehensive test coverage

### Test Data
- Use real data for testing (no mocks)
- Test with actual API responses when possible
- Include test fixtures for reproducible tests
- Handle test data cleanup appropriately

## Documentation Requirements

### Module Documentation
- Every folder level must have AGENTS.md and README.md
- AGENTS.md provides comprehensive module documentation
- README.md provides usage and quick reference
- Keep documentation synchronized with code

### System Documentation
- Maintain root-level AGENTS.md with complete system documentation
- Document all workflows and operations
- Provide troubleshooting guides
- Include configuration and setup instructions

