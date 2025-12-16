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
  - `sources/` - API adapters (arXiv, Semantic Scholar, PubMed, CrossRef, OpenAlex, DBLP, bioRxiv, Europe PMC, Unpaywall)
  - `pdf/` - PDF handling (downloading, extraction, failed download tracking)
  - `library/` - Library management (indexing, BibTeX generation)
  - `workflow/` - Workflow orchestration with progress tracking
  - `summarization/` - AI summarization (individual paper summaries)
  - `meta_analysis/` - Meta-analysis tools (bibliographic and statistical analysis)
  - `analysis/` - Paper analysis (paper analyzer, domain detector, context builder)
  - `html_parsers/` - Publisher-specific PDF URL extraction (Elsevier, Springer, IEEE, ACM, Wiley, generic)
  - `reporting/` - Multi-format export (JSON, CSV, HTML)
  - `llm/` - Advanced LLM operations (multi-paper synthesis: literature review, science communication, comparative analysis, research gaps, citation network analysis)
- `infrastructure/validation/` - PDF validation and text extraction

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

### Standard Meta-Analysis
- Bibliographic analysis (publication trends, venue distribution, author contributions)
- Citation analysis (citation distribution, metadata completeness)
- Keyword analysis (keyword frequency, keyword evolution over time)
- PCA analysis (text feature extraction and clustering)
- Visualization generation (PNG, PDF formats)
- No LLM/Ollama required

### Meta-Analysis with Embeddings
- Includes all standard meta-analysis features
- Requires Ollama server running and embedding model installed
- Generates semantic embeddings using Ollama embeddinggemma model
- Computes similarity matrices and clustering
- Provides comprehensive validation and quality metrics
- Enhanced visualizations (embedding clusters, similarity networks, quality plots)
- Automatic retry logic and Ollama health checks
- Embedding caching to avoid regeneration
- See [Meta-Analysis Guide](docs/guides/meta-analysis.md) for details

### Output Management
- Save all meta-analysis outputs to `data/output/` directory
- Generate visualizations (PNG, PDF formats)
- Create summary reports (JSON, Markdown)
- Organize outputs by analysis type
- Embedding analysis outputs include: embeddings.json, similarity matrices, clustering results, validation reports, statistics, and enhanced visualizations

### Visualization
- Generate publication trends visualizations
- Create keyword frequency charts
- Produce author network visualizations
- Support multiple visualization formats
- Embedding analysis includes: cluster visualizations, similarity heatmaps, quality metrics plots, dimensionality analysis

## Advanced LLM Operations

The system provides advanced LLM operations for multi-paper literature analysis. All operations use local Ollama models to generate text-based analyses.

### Available Operations

1. **Summarize Papers** (Individual)
   - Generate comprehensive summaries for individual papers with PDFs
   - Output: `data/summaries/{citation_key}_summary.md`
   - Length: 600-1000 words per paper
   - Requires: PDFs downloaded and text extracted

2. **Literature Review Synthesis**
   - Synthesize multiple papers into cohesive literature review paragraphs
   - Output: `literature/llm_outputs/review_outputs/literature_review_{timestamp}.md`
   - Length: 300-500 words
   - Configurable focus: methodology, results, theory, general
   - Uses paper summaries or abstracts

3. **Science Communication Narrative**
   - Create accessible science communication narratives for general audiences
   - Output: `literature/llm_outputs/communication_outputs/science_communication_{timestamp}.md`
   - Length: 600-800 words
   - Configurable audience: general_public, students, researchers
   - Narrative styles: storytelling, explanation, timeline

4. **Comparative Analysis**
   - Compare methods, findings, datasets, or performance across papers
   - Output: `literature/llm_outputs/compare_outputs/comparative_analysis_{timestamp}.md`
   - Length: 500-700 words
   - Configurable aspect: methods, results, datasets, performance

5. **Research Gap Identification**
   - Identify unanswered questions, methodological gaps, and future research directions
   - Output: `literature/llm_outputs/gaps_outputs/research_gaps_{timestamp}.md`
   - Length: 400-600 words
   - Configurable domain context for focused analysis

6. **Citation Network Analysis**
   - Analyze intellectual connections and relationships between papers (text-based)
   - Output: `literature/llm_outputs/network_outputs/citation_network_{timestamp}.md`
   - Length: 500-700 words
   - Note: This is text-based analysis, not graph visualization

### Paper Selection
- All multi-paper operations (2-6) use paper selection configuration
- Create `literature/paper_selection.yaml` for filtering
- Selection criteria: citation keys, year range, source filtering, PDF/summary availability, keyword matching, limit
- See [LLM Operations Guide](docs/guides/llm-operations.md) for details

## HTML Parsers

### Publisher-Specific PDF URL Extraction
- Extract PDF URLs from publisher landing pages when direct PDF links unavailable
- Modular parser system with publisher-specific implementations:
  - **Elsevier** - Elsevier journal pages
  - **Springer** - Springer journal pages
  - **IEEE** - IEEE Xplore pages
  - **ACM** - ACM Digital Library pages
  - **Wiley** - Wiley Online Library pages
  - **Generic** - Fallback parser for other publishers
- Automatic parser selection based on URL patterns
- Handles HTML parsing, link extraction, and PDF URL validation

## Reporting Module

### Multi-Format Export
- Export library data in multiple formats:
  - **JSON** - Complete library data with all metadata
  - **CSV** - Tabular format for spreadsheet analysis
  - **HTML** - Formatted reports for web viewing
- Library statistics and summaries
- Configurable export options
- See `infrastructure/literature/reporting/` for implementation

## Analysis Module

### Paper Analysis Tools
- **PaperAnalyzer** - Structure and content analysis of papers
- **DomainDetector** - Automatic domain detection from paper content
- **ContextBuilder** - Rich context generation for papers
- Supports enhanced paper understanding and categorization
- See `infrastructure/literature/analysis/` for implementation

## Workflow Orchestration

### Script Organization
- Use `scripts/` directory for orchestrator scripts (thin coordinators)
- Implement interactive menu in `run_literature.sh`
- Support both interactive and command-line workflows
- Provide clear workflow options and feedback
- Scripts delegate to infrastructure modules (thin orchestrator pattern)

### Workflow Operations
- Support full pipeline (search → download → extract → summarize)
- Allow individual operations (search-only, download-only, etc.)
- Implement workflow state tracking
- Provide progress feedback for long-running operations
- Support advanced LLM operations and meta-analysis workflows

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

