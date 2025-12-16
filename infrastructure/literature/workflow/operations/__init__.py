"""Workflow operations module - split from orchestrator for better modularity."""
from __future__ import annotations

# Re-export all operation functions for backward compatibility
from infrastructure.literature.workflow.operations.search import (
    get_keywords_input,
    get_limit_input,
    get_clear_options_input,
    run_search_only,
    run_search,
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
from infrastructure.literature.workflow.operations.utils import (
    get_source_descriptions,
    display_sources_with_status,
    get_source_selection_input,
    display_file_locations,
    SOURCE_DESCRIPTIONS,
)

# Import meta_analysis and llm_operations
from infrastructure.literature.workflow.operations.meta_analysis import run_meta_analysis
from infrastructure.literature.workflow.operations.llm_operations import run_llm_operation
from infrastructure.literature.workflow.operations.enrichment import run_enrich_dois

__all__ = [
    # Search operations
    "get_keywords_input",
    "get_limit_input",
    "get_clear_options_input",
    "run_search_only",
    "run_search",
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
    # Utils
    "get_source_descriptions",
    "display_sources_with_status",
    "get_source_selection_input",
    "display_file_locations",
    "SOURCE_DESCRIPTIONS",
    # Meta-analysis (when created)
    "run_meta_analysis",
    # LLM operations (when created)
    "run_llm_operation",
    # DOI enrichment
    "run_enrich_dois",
]

