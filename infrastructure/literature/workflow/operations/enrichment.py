"""DOI enrichment operations for literature workflow."""
from __future__ import annotations

from infrastructure.core.logging_utils import get_logger, log_header
from infrastructure.literature.workflow.workflow import LiteratureWorkflow

logger = get_logger(__name__)


def run_enrich_dois(workflow: LiteratureWorkflow) -> int:
    """Enrich library entries with DOIs from multiple sources.

    Searches CrossRef, Semantic Scholar, and OpenAlex for published
    versions of papers that don't have DOIs, especially arXiv preprints.

    Args:
        workflow: Configured LiteratureWorkflow instance.

    Returns:
        Exit code (0=success, 1=failure).
    """
    log_header("DOI ENRICHMENT")

    # Get entries missing DOIs
    entries_missing_doi = workflow.literature_search.library_index.get_entries_missing_doi()
    
    if not entries_missing_doi:
        logger.info("All library entries already have DOIs. Nothing to enrich.")
        return 0
    
    logger.info(f"Found {len(entries_missing_doi)} entries missing DOIs")
    logger.info("Searching CrossRef, Semantic Scholar, and OpenAlex for published versions...")
    logger.info("This may take a while due to rate limiting...\n")
    
    # Run enrichment
    try:
        stats = workflow.literature_search.enrich_dois()
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("DOI ENRICHMENT RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total processed: {stats.get('total_processed', 0)}")
        logger.info(f"DOIs found: {stats.get('found', 0)}")
        logger.info(f"Entries updated: {stats.get('updated', 0)}")
        logger.info(f"Failed: {stats.get('failed', 0)}")
        
        if stats.get('found', 0) > 0:
            success_rate = (stats.get('found', 0) / stats.get('total_processed', 1)) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        if stats.get('errors'):
            logger.warning(f"\nErrors encountered ({len(stats['errors'])}):")
            for error in stats['errors'][:10]:  # Show first 10 errors
                logger.warning(f"  • {error}")
            if len(stats['errors']) > 10:
                logger.warning(f"  ... and {len(stats['errors']) - 10} more errors")
        
        logger.info("\n✓ DOI enrichment complete")
        logger.info("Updated entries are saved in library.json and references.bib")
        
        return 0
        
    except Exception as e:
        logger.error(f"DOI enrichment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


