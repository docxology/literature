# Workflow Module - Complete Documentation

## Purpose

The workflow module orchestrates multi-paper operations, tracks progress, and manages search workflows.

## Components

### LiteratureWorkflow (workflow.py)

High-level workflow orchestration for literature processing.

**Key Methods:**
- `execute_search_and_summarize(keywords, limit_per_keyword, sources=None, retry_failed=False, max_parallel_summaries=1)` - Complete workflow: search → download → summarize
- `set_summarizer(summarizer)` - Set the summarizer for this workflow
- `set_progress_tracker(progress_tracker)` - Set the progress tracker for this workflow
- `_search_papers(keywords, limit_per_keyword, sources=None)` - Internal: Search for papers (returns List[SearchResult])
- `_download_papers(search_results, retry_failed=False)` - Internal: Download PDFs (returns tuple of (downloaded, results))
- `_summarize_papers_parallel(downloaded, max_parallel_summaries)` - Internal: Generate summaries in parallel

### ProgressTracker (progress.py)

Progress tracking for resumable operations.

**Key Methods:**
- `start_new_run(keywords, total_papers)` - Start a new summarization run
- `add_paper(citation_key, pdf_path)` - Add a paper to progress tracking
- `update_entry_status(citation_key, status, **kwargs)` - Update status for a paper entry
- `get_entry(citation_key)` - Get progress entry for a citation key
- `save_progress()` - Save progress to disk
- `load_progress()` - Load progress from disk
- `get_progress_summary()` - Get summary statistics as dictionary
- `archive_progress()` - Archive current progress to timestamped file

**Features:**
- Resumable operations
- Per-paper status tracking (pending, downloaded, processing, summarized, failed)
- Progress persistence to JSON file
- Automatic skip of existing summaries

### Failed Download Tracker Integration

The workflow module integrates with `FailedDownloadTracker` from `infrastructure.literature.pdf.failed_tracker` to track and manage failed PDF downloads.

**Automatic Failure Tracking:**
- All download failures are automatically saved to `data/failed_downloads.json`
- Failures are tracked in all download operations:
  - `_download_papers_sequential()` - Sequential downloads
  - `_download_papers_parallel()` - Parallel downloads
  - Meta-analysis pipeline downloads
  - Download-only operation downloads

**Skip Behavior:**
- By default, previously failed downloads are automatically skipped
- Skip happens in `find_papers_needing_pdf()` function
- Skip message: "Skipped X paper(s) with previously failed downloads (use --retry-failed to retry)"
- This prevents wasting time on papers that are likely to fail again (e.g., access-restricted papers)

**Retry Mechanism:**
- Use `retry_failed=True` parameter to retry previously failed downloads
- Only retriable failures (network errors, timeouts) are retried by default
- All failures can be retried if explicitly requested
- Successful retries automatically remove entries from the tracker

**Failure Categories:**
- **Retriable**: `network_error`, `timeout` (may succeed on retry)
- **Not Retriable**: `access_denied`, `not_found`, `html_response` (unlikely to succeed)
- **Not Tracked**: `no_pdf_url` (just a warning, not a failure)

**Integration Points:**
- `LiteratureWorkflow` initializes `FailedDownloadTracker` on creation
- All download operations use `workflow.failed_tracker.save_failed()` to track failures
- Operations check `workflow.failed_tracker.is_failed()` to skip previously failed downloads

### Search Orchestrator (orchestrator.py)

Thin orchestrator that imports and re-exports functions from operations submodules for backward compatibility.

**Note:** All operation logic has been moved to the `operations/` subdirectory for better modularity. The orchestrator.py file is now only 78 lines (down from 2099 lines), serving as a thin coordinator.

**Key Functions (re-exported from operations):**
- `run_search_only()` - Search without downloading
- `run_download_only()` - Download PDFs only
- `run_search()` - Full search workflow
- `run_cleanup()` - Library cleanup
- `run_meta_analysis()` - Meta-analysis pipeline
- `run_llm_operation()` - Advanced LLM operations

### Operations Submodules (operations/)

Operation-specific modules split from orchestrator for better modularity and testability:

**operations/search.py** - Search operations:
- `run_search_only()` - Execute search only (add to bibliography)
- `run_search()` - Execute full search workflow
- `get_keywords_input()` - Prompt for keywords
- `get_limit_input()` - Prompt for search limit
- `get_clear_options_input()` - Prompt for clear options

**operations/download.py** - Download operations:
- `run_download_only()` - Download PDFs for existing bibliography entries
- `find_papers_needing_pdf()` - Find entries needing PDF downloads
- `library_entry_to_search_result()` - Convert library entry to search result
- `failed_download_to_search_result()` - Convert failed download to search result
- `get_pdf_path_for_entry()` - Get PDF path for library entry

**operations/cleanup.py** - Cleanup operations:
- `run_cleanup()` - Clean up library (remove papers without PDFs, delete orphaned files)
- `find_orphaned_files()` - Find orphaned files not in bibliography
- `find_orphaned_pdfs()` - Find orphaned PDFs for meta-analysis
- `delete_orphaned_files()` - Delete orphaned files with error handling

**operations/meta_analysis.py** - Meta-analysis operations:
- `run_meta_analysis()` - Execute literature search workflow with meta-analysis
  - Runs search → download → extract → meta-analysis pipeline
  - Performs PCA analysis, keyword analysis, author analysis, and visualizations

**operations/llm_operations.py** - LLM operations:
- `run_llm_operation()` - Execute advanced LLM operation on selected papers
  - Supports: review, communication, compare, gaps, network operations

**operations/utils.py** - Common utilities:
- `get_source_descriptions()` - Get descriptions for all available sources
- `display_sources_with_status()` - Display sources with health status
- `get_source_selection_input()` - Prompt user to select sources
- `display_file_locations()` - Display file location summary

## Usage Examples

### Workflow Operations

```python
from infrastructure.literature.workflow import LiteratureWorkflow, LiteratureSearch
from infrastructure.literature.core import LiteratureConfig

# Initialize workflow
config = LiteratureConfig()
literature_search = LiteratureSearch(config)
workflow = LiteratureWorkflow(literature_search)

# Execute complete workflow (search → download → summarize)
result = workflow.execute_search_and_summarize(
    keywords=["active inference"],
    limit_per_keyword=10,
    max_parallel_summaries=2
)

print(f"Found {result.papers_found} papers")
print(f"Downloaded {result.papers_downloaded} PDFs")
print(f"Generated {result.summaries_generated} summaries")
```

### Progress Tracking

```python
from infrastructure.literature.workflow import ProgressTracker
from pathlib import Path

tracker = ProgressTracker(progress_file=Path("data/summarization_progress.json"))
tracker.load_progress()

if tracker.current_progress:
    summary = tracker.get_progress_summary()
    print(f"Processed: {summary['completed_summaries']}/{summary['total_papers']}")
```

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


