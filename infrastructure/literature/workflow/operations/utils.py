"""Common utilities for workflow operations."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.workflow.workflow import LiteratureWorkflow

logger = get_logger(__name__)

# Source descriptions and metadata
SOURCE_DESCRIPTIONS = {
    "arxiv": {
        "name": "arXiv",
        "description": "Preprint repository for physics, mathematics, computer science, and related fields",
        "supports_search": True,
        "rate_limit": "3 seconds between requests"
    },
    "semanticscholar": {
        "name": "Semantic Scholar",
        "description": "AI-powered academic search engine with citation analysis",
        "supports_search": True,
        "rate_limit": "1.5 seconds between requests (optional API key for higher limits)"
    },
    "biorxiv": {
        "name": "bioRxiv/medRxiv",
        "description": "Biology and medicine preprint server",
        "supports_search": True,
        "rate_limit": "Standard API limits"
    },
    "pubmed": {
        "name": "PubMed",
        "description": "Medical and life sciences literature database (NCBI)",
        "supports_search": True,
        "rate_limit": "~3 requests/second"
    },
    "europepmc": {
        "name": "Europe PMC",
        "description": "European biomedical literature database with full-text access",
        "supports_search": True,
        "rate_limit": "Standard API limits"
    },
    "crossref": {
        "name": "CrossRef",
        "description": "DOI-based metadata and citation database",
        "supports_search": True,
        "rate_limit": "1 second between requests"
    },
    "openalex": {
        "name": "OpenAlex",
        "description": "Open access academic database with comprehensive metadata",
        "supports_search": True,
        "rate_limit": "Standard API limits"
    },
    "dblp": {
        "name": "DBLP",
        "description": "Computer science bibliography database",
        "supports_search": True,
        "rate_limit": "Standard API limits"
    },
    "unpaywall": {
        "name": "Unpaywall",
        "description": "Open access PDF resolution (lookup only, no search)",
        "supports_search": False,
        "rate_limit": "Requires email address"
    }
}


def get_source_descriptions() -> dict:
    """Get descriptions for all available sources.
    
    Returns:
        Dictionary mapping source names to their descriptions.
    """
    return SOURCE_DESCRIPTIONS


def display_sources_with_status(
    workflow: LiteratureWorkflow,
    sources_to_display: Optional[List[str]] = None
) -> None:
    """Display available sources with their descriptions and health status.
    
    Args:
        workflow: LiteratureWorkflow instance to check source health.
        sources_to_display: Optional list of source names to display.
                          If None, displays all available sources.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("AVAILABLE LITERATURE SOURCES")
    logger.info("=" * 70)
    
    # Get all sources from workflow
    all_sources = list(workflow.literature_search.sources.keys())
    
    # Filter to requested sources if provided
    sources_to_show = sources_to_display if sources_to_display else all_sources
    
    # Get health status for all sources
    source_health = workflow.literature_search.get_source_health_status()
    
    # Display each source
    for source_name in sources_to_show:
        if source_name not in SOURCE_DESCRIPTIONS:
            # Unknown source - show basic info
            health_info = source_health.get(source_name, {})
            is_healthy = health_info.get('healthy', True)
            health_indicator = "✓" if is_healthy else "✗"
            
            logger.info(f"\n{health_indicator} {source_name.upper()}")
            logger.info(f"  Status: {'Healthy' if is_healthy else 'Unhealthy'}")
            continue
        
        desc = SOURCE_DESCRIPTIONS[source_name]
        health_info = source_health.get(source_name, {})
        is_healthy = health_info.get('healthy', True)
        health_indicator = "✓" if is_healthy else "✗"
        
        # Check if source supports search
        source_obj = workflow.literature_search.sources.get(source_name)
        supports_search = hasattr(source_obj, 'search') if source_obj else desc.get('supports_search', False)
        
        logger.info(f"\n{health_indicator} {desc['name']} ({source_name})")
        logger.info(f"  {desc['description']}")
        logger.info(f"  Status: {'Healthy' if is_healthy else 'Unhealthy'}")
        if not supports_search:
            logger.info("  Note: Lookup only (no search support)")
        if desc.get('rate_limit'):
            logger.info(f"  Rate limit: {desc['rate_limit']}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("")


def get_source_selection_input(
    workflow: LiteratureWorkflow,
    default_enabled: bool = True
) -> List[str]:
    """Prompt user to select which sources to use for search.
    
    Displays all available sources with descriptions and health status,
    then prompts Y/n for each source (default: all enabled).
    
    Args:
        workflow: LiteratureWorkflow instance to check source health.
        default_enabled: If True, default answer is 'Y' (enabled).
                       If False, default answer is 'N' (disabled).
    
    Returns:
        List of enabled source names.
    """
    # Display sources with status
    display_sources_with_status(workflow)
    
    # Get all searchable sources (exclude sources that don't support search)
    all_sources = list(workflow.literature_search.sources.keys())
    searchable_sources = [
        s for s in all_sources
        if hasattr(workflow.literature_search.sources[s], 'search')
    ]
    
    if not searchable_sources:
        logger.warning("No searchable sources available!")
        return []
    
    # Get health status
    source_health = workflow.literature_search.get_source_health_status()
    
    logger.info("Select sources to use for search:")
    logger.info("  (Press Enter for default, 'y' to enable, 'n' to disable)")
    logger.info("")
    
    enabled_sources = []
    default_char = "Y" if default_enabled else "N"
    
    for source_name in searchable_sources:
        desc = SOURCE_DESCRIPTIONS.get(source_name, {})
        source_display_name = desc.get('name', source_name.upper())
        
        health_info = source_health.get(source_name, {})
        is_healthy = health_info.get('healthy', True)
        health_indicator = "✓" if is_healthy else "✗"
        
        # Prompt for each source
        try:
            prompt = f"  {health_indicator} {source_display_name} [{default_char}]: "
            response = input(prompt).strip().lower()
            
            # Determine if source should be enabled
            if not response:
                # Default behavior
                should_enable = default_enabled
            else:
                should_enable = response in ('y', 'yes')
            
            if should_enable:
                enabled_sources.append(source_name)
                logger.debug(f"Enabled source: {source_name}")
            else:
                logger.debug(f"Disabled source: {source_name}")
                
        except (EOFError, KeyboardInterrupt):
            logger.info("\nSource selection cancelled, using defaults")
            # Return default enabled sources
            if default_enabled:
                return searchable_sources
            else:
                return []
    
    logger.info("")
    if enabled_sources:
        logger.info(f"Selected {len(enabled_sources)} source(s): {', '.join(enabled_sources)}")
    else:
        logger.warning("No sources selected! Using all available sources as fallback.")
        enabled_sources = searchable_sources
    
    return enabled_sources


def display_file_locations() -> None:
    """Display file location summary with absolute paths."""
    import json
    
    base_dir = Path("literature").resolve()
    
    logger.info("\nOutput file locations (absolute paths):")
    logger.info(f"  Base directory: {base_dir}")
    
    # BibTeX references
    bib_path = base_dir / "references.bib"
    if bib_path.exists():
        size = bib_path.stat().st_size
        logger.info(f"  ✓ {bib_path} ({size:,} bytes)")
    else:
        logger.info(f"  ✗ {bib_path} (not found)")
    
    # Library index
    library_path = base_dir / "library.json"
    if library_path.exists():
        size = library_path.stat().st_size
        try:
            with open(library_path, "r") as f:
                library_data = json.load(f)
                entry_count = library_data.get("count", 0)
            logger.info(f"  ✓ {library_path} ({size:,} bytes, {entry_count} papers)")
        except Exception:
            logger.info(f"  ✓ {library_path} ({size:,} bytes)")
    else:
        logger.info(f"  ✗ {library_path} (not found)")
    
    # PDFs directory
    pdf_dir = base_dir / "pdfs"
    if pdf_dir.exists():
        try:
            pdfs = list(pdf_dir.glob("*.pdf"))
            total_size = sum(f.stat().st_size for f in pdfs if f.is_file())
            logger.info(f"  ✓ {pdf_dir} ({len(pdfs)} PDFs, {total_size:,} bytes)")
        except Exception:
            logger.info(f"  ✓ {pdf_dir} (exists)")
    else:
        logger.info(f"  ✗ {pdf_dir} (not found)")
    
    # Extracted text directory
    extracted_dir = base_dir / "extracted_text"
    if extracted_dir.exists():
        try:
            texts = list(extracted_dir.glob("*.txt"))
            total_size = sum(f.stat().st_size for f in texts if f.is_file())
            logger.info(f"  ✓ {extracted_dir} ({len(texts)} files, {total_size:,} bytes)")
        except Exception:
            logger.info(f"  ✓ {extracted_dir} (exists)")
    else:
        logger.info(f"  ✗ {extracted_dir} (not found)")
    
    # Summaries directory
    summaries_dir = base_dir / "summaries"
    if summaries_dir.exists():
        try:
            summaries = list(summaries_dir.glob("*.md"))
            total_size = sum(f.stat().st_size for f in summaries if f.is_file())
            logger.info(f"  ✓ {summaries_dir} ({len(summaries)} summaries, {total_size:,} bytes)")
        except Exception:
            logger.info(f"  ✓ {summaries_dir} (exists)")
    else:
        logger.info(f"  ✗ {summaries_dir} (not found)")
    
    # Meta-analysis outputs
    output_dir = base_dir / "output"
    if output_dir.exists():
        try:
            outputs = list(output_dir.glob("*"))
            file_outputs = [f for f in outputs if f.is_file()]
            total_size = sum(f.stat().st_size for f in file_outputs)
            logger.info(f"  ✓ {output_dir} ({len(file_outputs)} files, {total_size:,} bytes)")
        except Exception:
            logger.info(f"  ✓ {output_dir} (exists)")
    else:
        logger.info(f"  ✗ {output_dir} (not found)")
    
    # Progress tracking
    progress_path = base_dir / "summarization_progress.json"
    if progress_path.exists():
        size = progress_path.stat().st_size
        logger.info(f"  ✓ {progress_path} ({size:,} bytes)")
    else:
        logger.info(f"  ✗ {progress_path} (not found)")




