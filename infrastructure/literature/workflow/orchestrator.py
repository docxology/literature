"""Search workflow orchestration for literature processing.

This module serves as a thin orchestrator that imports and re-exports
functions from the operations submodules for backward compatibility.

All actual operation logic has been moved to:
- operations/search.py - Search operations
- operations/download.py - Download operations
- operations/cleanup.py - Cleanup operations
- operations/meta_analysis.py - Meta-analysis operations
- operations/llm_operations.py - LLM operations
- operations/utils.py - Common utilities
"""
from __future__ import annotations

# Import all operations from submodules
from infrastructure.literature.workflow.operations.search import (
    get_keywords_input,
    get_limit_input,
    get_clear_options_input,
    run_search_only,
    run_search,
    DEFAULT_LIMIT_PER_KEYWORD,
)
from infrastructure.literature.workflow.operations.download import (
    library_entry_to_search_result,
    failed_download_to_search_result,
    get_pdf_path_for_entry,
    find_papers_needing_pdf,
    run_download_only,
)
from infrastructure.literature.workflow.operations.cleanup import (
    find_orphaned_pdfs,
    find_orphaned_files,
    delete_orphaned_files,
    run_cleanup,
)
from infrastructure.literature.workflow.operations.meta_analysis import run_meta_analysis
from infrastructure.literature.workflow.operations.llm_operations import run_llm_operation
from infrastructure.literature.workflow.operations.utils import (
    get_source_descriptions,
    display_sources_with_status,
    get_source_selection_input,
    display_file_locations,
    SOURCE_DESCRIPTIONS,
)

# Re-export everything for backward compatibility
__all__ = [
    # Search operations
    "get_keywords_input",
    "get_limit_input",
    "get_clear_options_input",
    "run_search_only",
    "run_search",
    "DEFAULT_LIMIT_PER_KEYWORD",
    # Download operations
    "library_entry_to_search_result",
    "failed_download_to_search_result",
    "get_pdf_path_for_entry",
    "find_papers_needing_pdf",
    "run_download_only",
    # Cleanup operations
    "find_orphaned_pdfs",
    "find_orphaned_files",
    "delete_orphaned_files",
    "run_cleanup",
    # Meta-analysis
    "run_meta_analysis",
    # LLM operations
    "run_llm_operation",
    # Utils
    "get_source_descriptions",
    "display_sources_with_status",
    "get_source_selection_input",
    "display_file_locations",
    "SOURCE_DESCRIPTIONS",
]
