"""Cleanup operation functions for literature workflow."""
from __future__ import annotations

from pathlib import Path
from typing import List

from infrastructure.core.logging_utils import get_logger, log_header, log_success
from infrastructure.literature.workflow.workflow import LiteratureWorkflow
from infrastructure.literature.library.index import LibraryEntry
from infrastructure.literature.workflow.operations.utils import display_file_locations

logger = get_logger(__name__)


def find_orphaned_pdfs(library_entries: List[LibraryEntry]) -> List[LibraryEntry]:
    """Find PDFs that exist in filesystem but are not in library index.
    
    Creates minimal LibraryEntry objects for orphaned PDFs so they can be
    included in meta-analysis. Attempts to extract basic metadata from
    extracted text files if available.
    
    Args:
        library_entries: List of existing library entries.
    
    Returns:
        List of LibraryEntry objects for orphaned PDFs.
    """
    pdfs_dir = Path("data/pdfs")
    extracted_text_dir = Path("data/extracted_text")
    
    if not pdfs_dir.exists():
        return []
    
    # Get all citation keys from library
    library_keys = {entry.citation_key for entry in library_entries}
    
    # Find orphaned PDFs
    orphaned_entries = []
    
    for pdf_file in pdfs_dir.glob("*.pdf"):
        citation_key = pdf_file.stem
        
        # Skip if already in library
        if citation_key in library_keys:
            continue
        
        # Try to extract basic metadata from extracted text if available
        text_file = extracted_text_dir / f"{citation_key}.txt"
        title = f"Paper: {citation_key}"  # Default title
        abstract = ""
        
        if text_file.exists():
            try:
                text_content = text_file.read_text(encoding='utf-8')
                # Try to extract title from first few lines (common pattern)
                lines = text_content.split('\n')[:20]
                for i, line in enumerate(lines):
                    line = line.strip()
                    if len(line) > 20 and len(line) < 200 and not line.startswith('http'):
                        # Likely a title
                        title = line
                        break
                # Use first 500 chars as abstract
                abstract = text_content[:500].strip()
            except Exception as e:
                logger.debug(f"Could not read extracted text for {citation_key}: {e}")
        
        # Create minimal library entry
        # PDF paths are stored relative to project root (e.g., "data/pdfs/paper.pdf")
        pdf_path_str = str(pdf_file)
        if not pdf_path_str.startswith("data/"):
            # Make relative to project root if absolute
            try:
                pdf_path_str = str(pdf_file.relative_to(Path.cwd()))
            except ValueError:
                # If not relative, use absolute path
                pdf_path_str = str(pdf_file)
        
        entry = LibraryEntry(
            citation_key=citation_key,
            title=title,
            authors=[],
            year=None,
            doi=None,
            source="orphaned",
            url="",
            pdf_path=pdf_path_str,
            added_date="",
            abstract=abstract,
            venue=None,
            citation_count=None,
            metadata={"orphaned": True}
        )
        
        orphaned_entries.append(entry)
    
    if orphaned_entries:
        logger.info(f"Found {len(orphaned_entries)} orphaned PDFs to include in meta-analysis")
    
    return orphaned_entries


def find_orphaned_files(library_entries: List[LibraryEntry]) -> dict:
    """Find all orphaned files (PDFs, summaries, extracted text) not in bibliography.
    
    Args:
        library_entries: List of library entries to check against.
    
    Returns:
        Dictionary with:
        - orphaned_pdfs: List of Path objects for orphaned PDF files
        - orphaned_summaries: List of Path objects for orphaned summary files
        - orphaned_extracted_texts: List of Path objects for orphaned extracted text files
        - pdf_total_size: Total size of orphaned PDFs in bytes
        - summary_total_size: Total size of orphaned summaries in bytes
        - extracted_text_total_size: Total size of orphaned extracted texts in bytes
    """
    pdfs_dir = Path("data/pdfs")
    extracted_text_dir = Path("data/extracted_text")
    summaries_dir = Path("data/summaries")
    
    # Get all citation keys from library
    library_keys = {entry.citation_key for entry in library_entries}
    
    # Find orphaned PDFs
    orphaned_pdfs = []
    pdf_total_size = 0
    if pdfs_dir.exists():
        for pdf_file in pdfs_dir.glob("*.pdf"):
            citation_key = pdf_file.stem
            if citation_key not in library_keys:
                if pdf_file.is_file():
                    orphaned_pdfs.append(pdf_file)
                    try:
                        pdf_total_size += pdf_file.stat().st_size
                    except OSError:
                        pass
    
    # Find orphaned summaries
    orphaned_summaries = []
    summary_total_size = 0
    if summaries_dir.exists():
        for summary_file in summaries_dir.glob("*_summary.md"):
            citation_key = summary_file.stem.replace("_summary", "")
            if citation_key not in library_keys:
                if summary_file.is_file():
                    orphaned_summaries.append(summary_file)
                    try:
                        summary_total_size += summary_file.stat().st_size
                    except OSError:
                        pass
    
    # Find orphaned extracted text files
    orphaned_extracted_texts = []
    extracted_text_total_size = 0
    if extracted_text_dir.exists():
        for text_file in extracted_text_dir.glob("*.txt"):
            citation_key = text_file.stem
            if citation_key not in library_keys:
                if text_file.is_file():
                    orphaned_extracted_texts.append(text_file)
                    try:
                        extracted_text_total_size += text_file.stat().st_size
                    except OSError:
                        pass
    
    return {
        "orphaned_pdfs": orphaned_pdfs,
        "orphaned_summaries": orphaned_summaries,
        "orphaned_extracted_texts": orphaned_extracted_texts,
        "pdf_total_size": pdf_total_size,
        "summary_total_size": summary_total_size,
        "extracted_text_total_size": extracted_text_total_size
    }


def delete_orphaned_files(orphaned_files: dict) -> dict:
    """Delete orphaned files with proper error handling and logging.
    
    Args:
        orphaned_files: Dictionary from find_orphaned_files() containing file lists and sizes.
    
    Returns:
        Dictionary with deletion statistics:
        - pdfs_deleted: Number of PDFs successfully deleted
        - pdfs_failed: Number of PDFs that failed to delete
        - pdfs_size_freed_mb: Size of deleted PDFs in MB
        - summaries_deleted: Number of summaries successfully deleted
        - summaries_failed: Number of summaries that failed to delete
        - summaries_size_freed_mb: Size of deleted summaries in MB
        - extracted_texts_deleted: Number of extracted text files successfully deleted
        - extracted_texts_failed: Number of extracted text files that failed to delete
        - extracted_texts_size_freed_mb: Size of deleted extracted text files in MB
        - total_size_freed_mb: Total size freed in MB
    """
    stats = {
        "pdfs_deleted": 0,
        "pdfs_failed": 0,
        "pdfs_size_freed_mb": 0.0,
        "summaries_deleted": 0,
        "summaries_failed": 0,
        "summaries_size_freed_mb": 0.0,
        "extracted_texts_deleted": 0,
        "extracted_texts_failed": 0,
        "extracted_texts_size_freed_mb": 0.0,
        "total_size_freed_mb": 0.0
    }
    
    # Delete orphaned PDFs
    for pdf_file in orphaned_files["orphaned_pdfs"]:
        try:
            if pdf_file.exists() and pdf_file.is_file():
                file_size = pdf_file.stat().st_size
                pdf_file.unlink()
                stats["pdfs_deleted"] += 1
                stats["pdfs_size_freed_mb"] += file_size / (1024 * 1024)
                logger.debug(f"  ✓ Deleted orphaned PDF: {pdf_file.name} ({file_size / (1024*1024):.2f} MB)")
            else:
                logger.warning(f"  ✗ Orphaned PDF not found or not a file: {pdf_file}")
                stats["pdfs_failed"] += 1
        except Exception as e:
            logger.warning(f"  ✗ Failed to delete orphaned PDF {pdf_file.name}: {e}")
            stats["pdfs_failed"] += 1
    
    # Delete orphaned summaries
    for summary_file in orphaned_files["orphaned_summaries"]:
        try:
            if summary_file.exists() and summary_file.is_file():
                file_size = summary_file.stat().st_size
                summary_file.unlink()
                stats["summaries_deleted"] += 1
                stats["summaries_size_freed_mb"] += file_size / (1024 * 1024)
                logger.debug(f"  ✓ Deleted orphaned summary: {summary_file.name} ({file_size / (1024*1024):.2f} MB)")
            else:
                logger.warning(f"  ✗ Orphaned summary not found or not a file: {summary_file}")
                stats["summaries_failed"] += 1
        except Exception as e:
            logger.warning(f"  ✗ Failed to delete orphaned summary {summary_file.name}: {e}")
            stats["summaries_failed"] += 1
    
    # Delete orphaned extracted text files
    for text_file in orphaned_files["orphaned_extracted_texts"]:
        try:
            if text_file.exists() and text_file.is_file():
                file_size = text_file.stat().st_size
                text_file.unlink()
                stats["extracted_texts_deleted"] += 1
                stats["extracted_texts_size_freed_mb"] += file_size / (1024 * 1024)
                logger.debug(f"  ✓ Deleted orphaned extracted text: {text_file.name} ({file_size / (1024*1024):.2f} MB)")
            else:
                logger.warning(f"  ✗ Orphaned extracted text not found or not a file: {text_file}")
                stats["extracted_texts_failed"] += 1
        except Exception as e:
            logger.warning(f"  ✗ Failed to delete orphaned extracted text {text_file.name}: {e}")
            stats["extracted_texts_failed"] += 1
    
    stats["total_size_freed_mb"] = (
        stats["pdfs_size_freed_mb"] +
        stats["summaries_size_freed_mb"] +
        stats["extracted_texts_size_freed_mb"]
    )
    
    return stats


def run_cleanup(workflow: LiteratureWorkflow) -> int:
    """Clean up library by removing papers without PDFs and deleting orphaned files.

    Args:
        workflow: Configured LiteratureWorkflow instance.

    Returns:
        Exit code (0=success, 1=failure).
    """
    log_header("CLEANUP LIBRARY (REMOVE PAPERS WITHOUT PDFs AND ORPHANED FILES)")

    # Get library entries
    library_entries = workflow.literature_search.library_index.list_entries()

    # Find entries without PDFs
    entries_without_pdf = workflow.literature_search.library_index.get_entries_without_pdf()
    
    # Find orphaned files
    orphaned_files = find_orphaned_files(library_entries)
    
    orphaned_pdf_count = len(orphaned_files["orphaned_pdfs"])
    orphaned_summary_count = len(orphaned_files["orphaned_summaries"])
    orphaned_extracted_text_count = len(orphaned_files["orphaned_extracted_texts"])
    
    # Check if there's anything to clean up
    if not library_entries:
        if orphaned_pdf_count == 0 and orphaned_summary_count == 0 and orphaned_extracted_text_count == 0:
            logger.warning("Library is empty and no orphaned files found. Nothing to clean up.")
            return 0
        else:
            logger.info("Library is empty, but orphaned files found.")
    
    # Display summary
    logger.info(f"\nLibrary contains {len(library_entries)} papers")
    logger.info(f"Papers with PDFs: {len(library_entries) - len(entries_without_pdf)}")
    logger.info(f"Papers without PDFs: {len(entries_without_pdf)}")
    logger.info(f"\nOrphaned files found:")
    logger.info(f"  • Orphaned PDFs: {orphaned_pdf_count} ({orphaned_files['pdf_total_size'] / (1024*1024):.2f} MB)")
    logger.info(f"  • Orphaned summaries: {orphaned_summary_count} ({orphaned_files['summary_total_size'] / (1024*1024):.2f} MB)")
    logger.info(f"  • Orphaned extracted texts: {orphaned_extracted_text_count} ({orphaned_files['extracted_text_total_size'] / (1024*1024):.2f} MB)")
    
    # Check if there's anything to clean up
    if not entries_without_pdf and orphaned_pdf_count == 0 and orphaned_summary_count == 0 and orphaned_extracted_text_count == 0:
        logger.info("\nAll papers in the library have PDFs and no orphaned files found. Nothing to clean up.")
        return 0

    # Show details of papers to be removed
    if entries_without_pdf:
        logger.info(f"\nPapers to be removed from bibliography ({len(entries_without_pdf)}):")
        for i, entry in enumerate(entries_without_pdf, 1):
            year = entry.year or "n/d"
            authors = entry.authors[0] if entry.authors else "Unknown"
            if len(entry.authors or []) > 1:
                authors += " et al."
            logger.info(f"  {i}. {entry.citation_key} - {authors} ({year}): {entry.title[:60]}...")
    
    # Show details of orphaned files to be deleted
    if orphaned_pdf_count > 0:
        logger.info(f"\nOrphaned PDFs to be deleted ({orphaned_pdf_count}):")
        for i, pdf_file in enumerate(orphaned_files["orphaned_pdfs"][:10], 1):  # Show first 10
            try:
                size_mb = pdf_file.stat().st_size / (1024 * 1024)
                logger.info(f"  {i}. {pdf_file.name} ({size_mb:.2f} MB)")
            except OSError:
                logger.info(f"  {i}. {pdf_file.name}")
        if orphaned_pdf_count > 10:
            logger.info(f"  ... and {orphaned_pdf_count - 10} more")
    
    if orphaned_summary_count > 0:
        logger.info(f"\nOrphaned summaries to be deleted ({orphaned_summary_count}):")
        for i, summary_file in enumerate(orphaned_files["orphaned_summaries"][:10], 1):  # Show first 10
            try:
                size_mb = summary_file.stat().st_size / (1024 * 1024)
                logger.info(f"  {i}. {summary_file.name} ({size_mb:.2f} MB)")
            except OSError:
                logger.info(f"  {i}. {summary_file.name}")
        if orphaned_summary_count > 10:
            logger.info(f"  ... and {orphaned_summary_count - 10} more")
    
    if orphaned_extracted_text_count > 0:
        logger.info(f"\nOrphaned extracted text files to be deleted ({orphaned_extracted_text_count}):")
        for i, text_file in enumerate(orphaned_files["orphaned_extracted_texts"][:10], 1):  # Show first 10
            try:
                size_mb = text_file.stat().st_size / (1024 * 1024)
                logger.info(f"  {i}. {text_file.name} ({size_mb:.2f} MB)")
            except OSError:
                logger.info(f"  {i}. {text_file.name}")
        if orphaned_extracted_text_count > 10:
            logger.info(f"  ... and {orphaned_extracted_text_count - 10} more")

    # Calculate totals for confirmation
    total_items_to_remove = len(entries_without_pdf) + orphaned_pdf_count + orphaned_summary_count + orphaned_extracted_text_count
    total_size_mb = (
        orphaned_files["pdf_total_size"] +
        orphaned_files["summary_total_size"] +
        orphaned_files["extracted_text_total_size"]
    ) / (1024 * 1024)

    # Ask for confirmation
    logger.info(f"\n{'=' * 60}")
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Bibliography entries to remove: {len(entries_without_pdf)}")
    logger.info(f"Orphaned files to delete: {orphaned_pdf_count + orphaned_summary_count + orphaned_extracted_text_count}")
    logger.info(f"  • PDFs: {orphaned_pdf_count} ({orphaned_files['pdf_total_size'] / (1024*1024):.2f} MB)")
    logger.info(f"  • Summaries: {orphaned_summary_count} ({orphaned_files['summary_total_size'] / (1024*1024):.2f} MB)")
    logger.info(f"  • Extracted texts: {orphaned_extracted_text_count} ({orphaned_files['extracted_text_total_size'] / (1024*1024):.2f} MB)")
    logger.info(f"Total space to be freed: {total_size_mb:.2f} MB")
    logger.info(f"\nThis will permanently remove {total_items_to_remove} items.")
    logger.info("This action cannot be undone.")
    try:
        confirmation = input("\nProceed with cleanup? [y/N]: ").strip().lower()
    except KeyboardInterrupt:
        logger.info("\n\nCleanup cancelled by user.")
        return 1

    if confirmation not in ('y', 'yes'):
        logger.info("Cleanup cancelled.")
        return 0

    # Perform cleanup
    log_header("PERFORMING CLEANUP")
    
    # Remove bibliography entries without PDFs
    removed_count = 0
    if entries_without_pdf:
        logger.info(f"\nRemoving {len(entries_without_pdf)} papers from bibliography...")
        for entry in entries_without_pdf:
            try:
                if workflow.literature_search.remove_paper(entry.citation_key):
                    removed_count += 1
                    logger.info(f"  ✓ Removed from bibliography: {entry.citation_key}")
                else:
                    logger.warning(f"  ✗ Failed to remove from bibliography: {entry.citation_key}")
            except Exception as e:
                logger.error(f"  ✗ Error removing {entry.citation_key}: {e}")
                continue
    
    # Delete orphaned files
    orphaned_stats = {
        "pdfs_deleted": 0,
        "pdfs_failed": 0,
        "pdfs_size_freed_mb": 0.0,
        "summaries_deleted": 0,
        "summaries_failed": 0,
        "summaries_size_freed_mb": 0.0,
        "extracted_texts_deleted": 0,
        "extracted_texts_failed": 0,
        "extracted_texts_size_freed_mb": 0.0,
        "total_size_freed_mb": 0.0
    }
    
    if orphaned_pdf_count > 0 or orphaned_summary_count > 0 or orphaned_extracted_text_count > 0:
        logger.info(f"\nDeleting {orphaned_pdf_count + orphaned_summary_count + orphaned_extracted_text_count} orphaned files...")
        orphaned_stats = delete_orphaned_files(orphaned_files)
        
        if orphaned_stats["pdfs_deleted"] > 0:
            logger.info(f"  ✓ Deleted {orphaned_stats['pdfs_deleted']} orphaned PDFs ({orphaned_stats['pdfs_size_freed_mb']:.2f} MB)")
        if orphaned_stats["pdfs_failed"] > 0:
            logger.warning(f"  ✗ Failed to delete {orphaned_stats['pdfs_failed']} orphaned PDFs")
        
        if orphaned_stats["summaries_deleted"] > 0:
            logger.info(f"  ✓ Deleted {orphaned_stats['summaries_deleted']} orphaned summaries ({orphaned_stats['summaries_size_freed_mb']:.2f} MB)")
        if orphaned_stats["summaries_failed"] > 0:
            logger.warning(f"  ✗ Failed to delete {orphaned_stats['summaries_failed']} orphaned summaries")
        
        if orphaned_stats["extracted_texts_deleted"] > 0:
            logger.info(f"  ✓ Deleted {orphaned_stats['extracted_texts_deleted']} orphaned extracted text files ({orphaned_stats['extracted_texts_size_freed_mb']:.2f} MB)")
        if orphaned_stats["extracted_texts_failed"] > 0:
            logger.warning(f"  ✗ Failed to delete {orphaned_stats['extracted_texts_failed']} orphaned extracted text files")

    # Show comprehensive results
    remaining_count = len(library_entries) - removed_count
    logger.info(f"\n{'=' * 60}")
    logger.info("CLEANUP COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Bibliography entries removed: {removed_count}")
    if len(entries_without_pdf) > 0:
        logger.info(f"Bibliography removal success rate: {(removed_count / len(entries_without_pdf)) * 100:.1f}%")
    logger.info(f"Bibliography entries remaining: {remaining_count}")
    logger.info("")
    logger.info("Orphaned files deleted:")
    logger.info(f"  • PDFs: {orphaned_stats['pdfs_deleted']} deleted, {orphaned_stats['pdfs_failed']} failed ({orphaned_stats['pdfs_size_freed_mb']:.2f} MB freed)")
    logger.info(f"  • Summaries: {orphaned_stats['summaries_deleted']} deleted, {orphaned_stats['summaries_failed']} failed ({orphaned_stats['summaries_size_freed_mb']:.2f} MB freed)")
    logger.info(f"  • Extracted texts: {orphaned_stats['extracted_texts_deleted']} deleted, {orphaned_stats['extracted_texts_failed']} failed ({orphaned_stats['extracted_texts_size_freed_mb']:.2f} MB freed)")
    logger.info(f"Total space freed: {orphaned_stats['total_size_freed_mb']:.2f} MB")

    display_file_locations()

    log_success("Library cleanup complete!")
    return 0




