# Workflow Module - Complete Documentation

## Purpose

The workflow module orchestrates multi-paper operations, tracks progress, and manages search workflows.

## Components

### LiteratureWorkflow (workflow.py)

High-level workflow orchestration for literature processing.

**Key Methods:**
- `search_and_add()` - Search and add papers to library
- `download_pdfs()` - Download PDFs for library entries
- `summarize_papers()` - Generate summaries for papers

### ProgressTracker (progress.py)

Progress tracking for resumable operations.

**Key Methods:**
- `get_progress()` - Get current progress
- `update_progress()` - Update progress for a paper
- `save_progress()` - Save progress to disk
- `load_progress()` - Load progress from disk

**Features:**
- Resumable operations
- Per-paper status tracking
- Progress persistence

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
from infrastructure.literature.workflow import LiteratureWorkflow

workflow = LiteratureWorkflow()

# Search and add
result = workflow.search_and_add(
    keywords=["active inference"],
    limit=10
)

# Download PDFs
download_result = workflow.download_pdfs()
```

### Progress Tracking

```python
from infrastructure.literature.workflow import ProgressTracker

tracker = ProgressTracker()
progress = tracker.get_progress()
print(f"Processed: {progress.completed}/{progress.total}")
```

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


