"""Search operation functions for literature workflow."""
from __future__ import annotations

import os
from typing import List, Optional

from infrastructure.core.logging_utils import get_logger, log_header, log_success
from infrastructure.literature.workflow.workflow import LiteratureWorkflow
from infrastructure.literature.workflow.operations.utils import (
    get_source_selection_input,
)

logger = get_logger(__name__)

DEFAULT_LIMIT_PER_KEYWORD = int(os.environ.get("LITERATURE_DEFAULT_LIMIT", "25"))


def get_keywords_input() -> List[str]:
    """Prompt user for comma-separated keywords.
    
    Multi-word terms are automatically quoted (e.g., "free energy principle").
    Users don't need to type quotes themselves.
    
    Returns:
        List of keyword strings, with multi-word terms automatically quoted.
    """
    try:
        keywords_str = input("Enter keywords (comma-separated, multi-word terms auto-quoted): ").strip()
        if not keywords_str:
            return []
        
        # Split by comma and process each keyword
        keywords = []
        for k in keywords_str.split(','):
            k = k.strip()
            if not k:
                continue
            
            # Remove existing quotes if user added them (we'll add our own)
            k = k.strip('"\'')
            
            # If keyword contains spaces, wrap it in quotes
            if ' ' in k:
                k = f'"{k}"'
            
            keywords.append(k)
        
        return keywords
    except (EOFError, KeyboardInterrupt):
        return []


def get_limit_input(default: int = DEFAULT_LIMIT_PER_KEYWORD) -> int:
    """Prompt user for search limit."""
    try:
        limit_str = input(f"Results per keyword [{default}]: ").strip()
        if not limit_str:
            return default
        return int(limit_str)
    except (ValueError, EOFError, KeyboardInterrupt):
        return default


def get_clear_options_input() -> tuple:
    """Prompt user for clear options.
    
    Returns:
        Tuple of (clear_pdfs, clear_summaries, clear_library).
    """
    try:
        print("\nClear options (default: No - incremental/additive behavior):")
        clear_pdfs_str = input("  Clear PDFs before download? [y/N]: ").strip().lower()
        clear_pdfs = clear_pdfs_str in ('y', 'yes')
        
        clear_summaries_str = input("  Clear summaries before generation? [y/N]: ").strip().lower()
        clear_summaries = clear_summaries_str in ('y', 'yes')
        
        print("  ⚠️  WARNING: Total clear will delete library index, PDFs, summaries, and progress file")
        clear_library_str = input("  Clear library completely (total clear)? [y/N]: ").strip().lower()
        clear_library = clear_library_str in ('y', 'yes')
        
        return (clear_pdfs, clear_summaries, clear_library)
    except (EOFError, KeyboardInterrupt):
        return (False, False, False)


def run_search_only(
    workflow: LiteratureWorkflow,
    keywords: Optional[List[str]] = None,
    limit: Optional[int] = None,
    sources: Optional[List[str]] = None,
    interactive: bool = True,
) -> int:
    """Execute literature search only (add to bibliography).

    Args:
        workflow: Configured LiteratureWorkflow instance.
        keywords: Optional keywords list (prompts if not provided).
        limit: Optional limit per keyword (prompts if not provided).
        sources: Optional list of sources to use (prompts if not provided and interactive=True).
        interactive: Whether to prompt for source selection if sources not provided.

    Returns:
        Exit code (0=success, 1=failure).
    """
    log_header("LITERATURE SEARCH (ADD TO BIBLIOGRAPHY)")

    # Get source selection if not provided
    if sources is None and interactive:
        sources = get_source_selection_input(workflow, default_enabled=True)
        if not sources:
            logger.error("No sources selected. Exiting.")
            return 1
    elif sources is None:
        # Non-interactive mode: use all available searchable sources
        enabled_sources = list(workflow.literature_search.sources.keys())
        sources = [s for s in enabled_sources 
                   if hasattr(workflow.literature_search.sources[s], 'search')]
    
    # Format sources display
    if not sources:
        sources_display = "no sources"
    elif len(sources) <= 8:
        # Show all sources if 8 or fewer
        sources_display = ', '.join(sources)
    else:
        # For many sources, show first few and count
        sources_display = f"{', '.join(sources[:5])}, and {len(sources) - 5} more"

    logger.info("\nThis will:")
    logger.info(f"  1. Search {sources_display} for papers")
    logger.info("  2. Add papers to bibliography (no download or summarization)")
    logger.info("")

    # Get limit if not provided
    if limit is None:
        limit = get_limit_input()

    # Get keywords if not provided
    if keywords is None:
        keywords = get_keywords_input()
        if not keywords:
            logger.info("No keywords provided. Exiting.")
            return 1

    # Execute search only
    log_header("SEARCHING FOR PAPERS")
    logger.info(f"Search keywords: {', '.join(keywords)}")
    logger.info(f"Results per keyword: {limit}")
    logger.info(f"Sources: {', '.join(sources)}")

    try:
        # Search for papers with selected sources
        search_results = workflow._search_papers(keywords, limit, sources=sources)
        papers_found = len(search_results)

        if not search_results:
            logger.warning("No papers found for the given keywords")
            return 1

        # Add all results to library
        log_header("ADDING TO BIBLIOGRAPHY")
        added_count = 0
        already_existed_count = 0

        for result in search_results:
            try:
                citation_key = workflow.literature_search.add_to_library(result)
                added_count += 1
                logger.info(f"Added: {citation_key}")
            except Exception:
                already_existed_count += 1
                logger.debug(f"Already exists: {result.title[:50]}...")

        # Get source information
        source_health = workflow.literature_search.get_source_health_status()
        
        # Display results
        logger.info(f"\n{'=' * 60}")
        logger.info("SEARCH COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Keywords searched: {', '.join(keywords)}")
        logger.info(f"Sources used: {', '.join(sources)}")
        logger.info(f"Papers found: {papers_found}")
        logger.info(f"Papers added to bibliography: {added_count}")
        if already_existed_count > 0:
            logger.info(f"Papers already in bibliography: {already_existed_count}")
        logger.info(f"Success rate: {(added_count / papers_found) * 100:.1f}%")
        
        # Display source health status
        unhealthy_sources = [name for name, status in source_health.items() 
                          if not status.get('healthy', True)]
        if unhealthy_sources:
            logger.warning(f"\n⚠️  Note: Some sources had issues: {', '.join(unhealthy_sources)}")
        
        log_success("Literature search complete!")
        return 0

    except Exception as e:
        logger.error(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_search(
    workflow: LiteratureWorkflow,
    keywords: Optional[List[str]] = None,
    limit: Optional[int] = None,
    max_parallel_summaries: int = 1,
    clear_pdfs: bool = False,
    clear_summaries: bool = False,
    clear_library: bool = False,
    interactive: bool = True,
    sources: Optional[List[str]] = None,
    retry_failed: bool = False,
) -> int:
    """Execute literature search workflow.
    
    Args:
        workflow: Configured LiteratureWorkflow instance.
        keywords: Optional keywords list (prompts if not provided).
        limit: Optional limit per keyword (prompts if not provided).
        max_parallel_summaries: Maximum parallel summarization workers.
        clear_pdfs: Clear PDFs before download (default: False).
        clear_summaries: Clear summaries before generation (default: False).
        clear_library: Perform total clear (library index, PDFs, summaries, progress file) 
                      before operations (default: False). If True, skips individual clear operations.
        interactive: Whether in interactive mode.
        sources: Optional list of sources to use (prompts if not provided and interactive=True).
        
    Returns:
        Exit code (0=success, 1=failure).
    """
    log_header("Literature Search and PDF Download")
    
    # Handle clear operations
    from infrastructure.literature.library.clear import clear_pdfs, clear_summaries, clear_library
    
    # If clear_library is True, it performs a total clear (library, PDFs, summaries, progress)
    # So we skip individual clear operations to avoid redundancy
    if clear_library:
        result = clear_library(confirm=True, interactive=interactive)
        if not result["success"]:
            logger.info("Library clear cancelled")
            return 1
        logger.info(f"Total clear completed: {result['message']}")
        # Skip individual clears since total clear already did everything
        clear_pdfs = False
        clear_summaries = False
    else:
        # Individual clear operations (only if not doing total clear)
        if clear_pdfs:
            result = clear_pdfs(confirm=True, interactive=interactive)
            if not result["success"]:
                logger.info("PDF clear cancelled")
                return 1
            logger.info(f"Cleared PDFs: {result['message']}")
        
        if clear_summaries:
            result = clear_summaries(confirm=True, interactive=interactive)
            if not result["success"]:
                logger.info("Summary clear cancelled")
                return 1
            logger.info(f"Cleared summaries: {result['message']}")
    
    # Get source selection if not provided
    if sources is None and interactive:
        sources = get_source_selection_input(workflow, default_enabled=True)
        if not sources:
            logger.error("No sources selected. Exiting.")
            return 1
    elif sources is None:
        # Non-interactive mode: use all available searchable sources
        enabled_sources = list(workflow.literature_search.sources.keys())
        sources = [s for s in enabled_sources 
                   if hasattr(workflow.literature_search.sources[s], 'search')]
    
    # Format sources display
    if not sources:
        sources_display = "no sources"
    elif len(sources) <= 8:
        # Show all sources if 8 or fewer
        sources_display = ', '.join(sources)
    else:
        # For many sources, show first few and count
        sources_display = f"{', '.join(sources[:5])}, and {len(sources) - 5} more"

    logger.info("\nThis will:")
    logger.info(f"  1. Search {sources_display} for papers")
    logger.info("  2. Download PDFs and add to BibTeX library")
    logger.info("  3. Generate AI summaries for each paper")
    logger.info(f"  4. Process up to {max_parallel_summaries} papers in parallel")
    logger.info("")
    
    # Get limit if not provided
    if limit is None:
        limit = get_limit_input()
    
    # Get keywords if not provided
    if keywords is None:
        keywords = get_keywords_input()
        if not keywords:
            logger.info("No keywords provided. Exiting.")
            return 1
    
    # Get clear options if in interactive mode
    if interactive and not (clear_pdfs or clear_summaries or clear_library):
        clear_pdfs, clear_summaries, clear_library = get_clear_options_input()
    
    # Check for failed downloads and prompt for retry
    if interactive and not retry_failed and workflow.failed_tracker.has_failures():
        retriable_count = workflow.failed_tracker.count_retriable()
        total_failed = workflow.failed_tracker.count_failures()
        if retriable_count > 0:
            logger.info(f"\nFound {retriable_count} retriable failed downloads (out of {total_failed} total)")
            retry_choice = input("Retry previously failed downloads? [y/N]: ").strip().lower()
            retry_failed = retry_choice in ('y', 'yes')
    
    # Execute search and summarization
    log_header("Executing Literature Search")
    logger.info(f"Search keywords: {', '.join(keywords)}")
    logger.info(f"Results per keyword: {limit}")
    logger.info(f"Sources: {', '.join(sources)}")
    logger.info(f"Max parallel summaries: {max_parallel_summaries}")
    
    try:
        result = workflow.execute_search_and_summarize(
            keywords=keywords,
            limit_per_keyword=limit,
            max_parallel_summaries=max_parallel_summaries,
            resume_existing=True,
            interactive=True,
            sources=sources,
            retry_failed=retry_failed
        )
        
        # Display results
        stats = workflow.get_workflow_stats(result)
        
        # Get source information
        source_health = workflow.literature_search.get_source_health_status()
        
        logger.info(f"\n{'=' * 60}")
        logger.info("SEARCH COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Keywords searched: {', '.join(keywords)}")
        logger.info(f"Sources used: {', '.join(sources)}")
        logger.info(f"Papers found: {stats['search']['papers_found']}")
        logger.info(f"Papers already downloaded: {result.papers_already_existed}")
        logger.info(f"Papers newly downloaded: {result.papers_newly_downloaded}")
        logger.info(f"Download failures: {result.papers_failed_download}")
        logger.info(f"Papers summarized: {stats['summarization']['successful']}")
        if result.summaries_skipped > 0:
            logger.info(f"Summaries skipped (already exist): {result.summaries_skipped}")
        logger.info(f"Summary failures: {result.summaries_failed}")
        logger.info(f"Success rate: {result.success_rate:.1f}%")
        
        # Display source health status
        unhealthy_sources = [name for name, status in source_health.items() 
                          if not status.get('healthy', True)]
        if unhealthy_sources:
            logger.warning(f"\n⚠️  Note: Some sources had issues: {', '.join(unhealthy_sources)}")
        
        log_success("Literature search complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

