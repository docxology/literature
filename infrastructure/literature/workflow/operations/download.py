"""Download operation functions for literature workflow."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from infrastructure.core.logging_utils import get_logger, log_header, log_success
from infrastructure.literature.workflow.workflow import LiteratureWorkflow
from infrastructure.literature.library.index import LibraryEntry
from infrastructure.literature.sources import SearchResult
from infrastructure.literature.pdf.failed_tracker import FailedDownloadTracker

logger = get_logger(__name__)


def library_entry_to_search_result(entry: LibraryEntry) -> SearchResult:
    """Convert library entry to search result for processing."""
    return SearchResult(
        title=entry.title,
        authors=entry.authors or [],
        year=entry.year,
        doi=entry.doi,
        url=entry.url,
        pdf_url=entry.metadata.get("pdf_url"),
        abstract=entry.abstract,
        source=entry.source or "library"
    )


def failed_download_to_search_result(failure_data: dict) -> SearchResult:
    """Convert failed download data to search result for retry.
    
    Args:
        failure_data: Dictionary from failed download tracker.
    
    Returns:
        SearchResult object for retry attempt.
    """
    # Try to get URL from attempted URLs (prefer first one)
    pdf_url = None
    if failure_data.get("attempted_urls"):
        pdf_url = failure_data["attempted_urls"][0]
    
    return SearchResult(
        title=failure_data.get("title", "Unknown"),
        authors=[],  # Not stored in failure data
        year=None,  # Not stored in failure data
        doi=None,  # Not stored in failure data
        url=pdf_url or "",
        pdf_url=pdf_url,
        abstract="",
        source=failure_data.get("source", "unknown")
    )


def get_pdf_path_for_entry(entry: LibraryEntry) -> Optional[Path]:
    """Get PDF path for a library entry."""
    if entry.pdf_path:
        pdf_path = Path(entry.pdf_path)
        if not pdf_path.is_absolute():
            pdf_path = Path("literature") / pdf_path
        if pdf_path.exists():
            return pdf_path

    # Check expected location
    expected_pdf = Path("data/pdfs") / f"{entry.citation_key}.pdf"
    if expected_pdf.exists():
        return expected_pdf

    return None


def find_papers_needing_pdf(
    library_entries: List[LibraryEntry],
    failed_tracker: Optional[FailedDownloadTracker] = None,
    retry_failed: bool = False
) -> List[LibraryEntry]:
    """Find library entries that need PDF downloads.
    
    Args:
        library_entries: List of library entries to check.
        failed_tracker: Optional failed download tracker to filter out previously failed downloads.
        retry_failed: If True, include previously failed downloads. If False and failed_tracker
                     is provided, filter out entries that have failed downloads.
    
    Returns:
        List of library entries that need PDF downloads.
    """
    papers_needing_pdf = []
    skipped_count = 0
    
    for entry in library_entries:
        pdf_path = get_pdf_path_for_entry(entry)
        if not pdf_path or not pdf_path.exists():
            # Check if this entry has a previously failed download
            if failed_tracker and not retry_failed and failed_tracker.is_failed(entry.citation_key):
                skipped_count += 1
                continue
            papers_needing_pdf.append(entry)
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} paper(s) with previously failed downloads (use --retry-failed to retry)")
    
    return papers_needing_pdf


def run_download_only(workflow: LiteratureWorkflow, retry_failed: bool = False) -> int:
    """Download PDFs for existing bibliography entries.

    Args:
        workflow: Configured LiteratureWorkflow instance.
        retry_failed: If True, include previously failed downloads in queue.

    Returns:
        Exit code (0=success, 1=failure).
    """
    log_header("DOWNLOAD PDFs (FOR BIBLIOGRAPHY ENTRIES)")

    # Load library entries
    library_entries = workflow.literature_search.library_index.list_entries()

    # Find entries needing PDFs (filter out previously failed downloads unless retry_failed=True)
    papers_needing_pdf = find_papers_needing_pdf(
        library_entries,
        failed_tracker=workflow.failed_tracker,
        retry_failed=retry_failed
    )
    
    # Add failed downloads if retry requested
    failed_results = []
    if retry_failed and workflow.failed_tracker.has_failures():
        failed_downloads = workflow.failed_tracker.get_retriable_failed()
        logger.info(f"Found {len(failed_downloads)} retriable failed downloads")
        
        for citation_key, failure_data in failed_downloads.items():
            # Check if PDF already exists (might have been downloaded manually)
            expected_pdf = Path("data/pdfs") / f"{citation_key}.pdf"
            if not expected_pdf.exists():
                search_result = failed_download_to_search_result(failure_data)
                failed_results.append((citation_key, search_result))
            else:
                # Remove from tracker if PDF now exists
                workflow.failed_tracker.remove_successful(citation_key)
        
        if failed_results:
            logger.info(f"Retrying {len(failed_results)} previously failed downloads")

    if not library_entries:
        if not failed_results:
            logger.warning("Library is empty and no failed downloads to retry. Use --search-only to find and add papers first.")
            return 0
        else:
            logger.info("Library is empty, but retrying failed downloads")

    total_to_download = len(papers_needing_pdf) + len(failed_results)
    
    logger.info(f"\nLibrary contains {len(library_entries)} papers")
    logger.info(f"Papers needing PDF download: {len(papers_needing_pdf)}")
    if failed_results:
        logger.info(f"Previously failed downloads to retry: {len(failed_results)}")

    if total_to_download == 0:
        logger.info("\nAll papers in bibliography already have PDFs. Nothing to download.")
        return 0

    # Download PDFs
    log_header("DOWNLOADING PDFs")
    logger.info(f"Attempting to download {total_to_download} PDFs...")

    downloaded_count = 0
    failed_count = 0

    # Process regular library entries
    for i, entry in enumerate(papers_needing_pdf, 1):
        logger.info(f"[{i}/{total_to_download}] Processing: {entry.title[:60]}...")

        search_result = library_entry_to_search_result(entry)
        download_result = workflow.literature_search.download_paper_with_result(search_result)

        if download_result.success and download_result.pdf_path:
            # Enhanced success logging with absolute path and source
            abs_path = download_result.pdf_path.resolve()
            file_size = abs_path.stat().st_size if abs_path.exists() else 0
            source = entry.source or "unknown"
            
            if download_result.already_existed:
                logger.info(f"✓ Already exists: {abs_path} ({file_size:,} bytes) [Source: {source}]")
            else:
                log_success(f"✓ Downloaded: {abs_path} ({file_size:,} bytes) [Source: {source}]")
            downloaded_count += 1
        else:
            # Enhanced failure logging with full context
            error_msg = download_result.failure_message or "Unknown error"
            expected_path = Path("data/pdfs") / f"{entry.citation_key}.pdf"
            urls_attempted = download_result.attempted_urls or []
            source = entry.source or "unknown"
            
            logger.error(f"✗ Failed: {entry.citation_key}")
            logger.error(f"  Title: {entry.title[:80]}...")
            logger.error(f"  Error: {download_result.failure_reason or 'unknown'}: {error_msg}")
            logger.error(f"  Expected path: {expected_path.resolve()}")
            logger.error(f"  Source: {source}")
            if urls_attempted:
                logger.error(f"  URLs attempted: {len(urls_attempted)}")
                for j, url in enumerate(urls_attempted[:3], 1):  # Show first 3
                    logger.error(f"    {j}. {url[:100]}...")
                if len(urls_attempted) > 3:
                    logger.error(f"    ... and {len(urls_attempted) - 3} more")
            failed_count += 1
    
    # Process failed downloads retry
    for i, (citation_key, search_result) in enumerate(failed_results, len(papers_needing_pdf) + 1):
        logger.info(f"[{i}/{total_to_download}] Retrying: {search_result.title[:60]}...")

        download_result = workflow.literature_search.download_paper_with_result(search_result)

        if download_result.success and download_result.pdf_path:
            abs_path = download_result.pdf_path.resolve()
            file_size = abs_path.stat().st_size if abs_path.exists() else 0
            source = search_result.source or "unknown"
            
            log_success(f"✓ Retry successful: {abs_path} ({file_size:,} bytes) [Source: {source}]")
            downloaded_count += 1
        else:
            error_msg = download_result.failure_message or "Unknown error"
            expected_path = Path("data/pdfs") / f"{citation_key}.pdf"
            source = search_result.source or "unknown"
            
            logger.error(f"✗ Retry failed: {citation_key}")
            logger.error(f"  Title: {search_result.title[:80]}...")
            logger.error(f"  Error: {download_result.failure_reason or 'unknown'}: {error_msg}")
            logger.error(f"  Expected path: {expected_path.resolve()}")
            logger.error(f"  Source: {source}")
            failed_count += 1

    # Display summary
    logger.info(f"\n{'=' * 60}")
    logger.info("PDF DOWNLOAD COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Papers processed: {total_to_download}")
    logger.info(f"PDFs downloaded: {downloaded_count}")
    if failed_count > 0:
        logger.warning(f"Download failures: {failed_count}")
    logger.info(f"Success rate: {(downloaded_count / total_to_download) * 100:.1f}%")

    if downloaded_count > 0:
        log_success("PDF download complete!")
    else:
        logger.warning("No PDFs were downloaded")

    return 0 if downloaded_count > 0 else 1

