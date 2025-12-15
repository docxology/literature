"""Utility to check and re-extract incomplete HTML text files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.core.config import LiteratureConfig
from infrastructure.literature.pdf.handler import PDFHandler
from infrastructure.literature.pdf.html_extractor import HTMLTextExtractor
from infrastructure.literature.library.index import LibraryIndex

logger = get_logger(__name__)


def check_and_re_extract_incomplete_files(
    library_path: Path | None = None,
    pdf_dir: Path | None = None,
    min_words: int = 1000,
    config: LiteratureConfig | None = None
) -> List[Tuple[str, str]]:
    """Check extracted text files and re-extract if they're too short.
    
    Args:
        library_path: Optional path to library.json file. If None, uses config default.
        pdf_dir: Optional directory containing extracted text files. If None, uses config default.
        min_words: Minimum word count to consider a file complete (default: 1000).
        config: Optional LiteratureConfig. If None, loads from environment.
        
    Returns:
        List of tuples (citation_key, status) where status is 're-extracted', 'failed', or 'skipped'.
    """
    if config is None:
        config = LiteratureConfig.from_env()
    
    if pdf_dir is None:
        pdf_dir = Path(config.download_dir)
    
    # Load library index
    library_index = LibraryIndex(config)
    
    # Initialize handlers
    handler = PDFHandler(config, library_index)
    html_extractor = HTMLTextExtractor()
    
    results = []
    
    # Find all .txt files in pdf_dir
    txt_files = list(pdf_dir.glob("*.txt"))
    logger.info(f"Checking {len(txt_files)} extracted text files...")
    
    for txt_file in txt_files:
        citation_key = txt_file.stem
        
        try:
            # Read current content
            current_text = txt_file.read_text(encoding='utf-8')
            word_count = len(current_text.split())
            
            # Check if file is too short
            if word_count < min_words:
                logger.info(
                    f"File {txt_file.name} has only {word_count} words "
                    f"(minimum {min_words} required). Attempting re-extraction..."
                )
                
                # Get paper metadata from library
                entry = library_index.get_entry(citation_key)
                if not entry:
                    logger.warning(f"Entry not found in library for {citation_key}, skipping")
                    results.append((citation_key, 'skipped'))
                    continue
                
                # Try to re-extract from DOI or URL
                re_extracted = False
                
                # Try IEEE Xplore URL from DOI if available
                if entry.doi and entry.doi.startswith('10.1109/'):
                    from infrastructure.literature.pdf.fallbacks import doi_to_pdf_urls
                    ieee_urls = doi_to_pdf_urls(entry.doi)
                    # Filter for IEEE Xplore document URLs (for HTML extraction)
                    ieee_doc_urls = [u for u in ieee_urls if 'ieeexplore.ieee.org/document' in u]
                    
                    if ieee_doc_urls:
                        logger.info(f"Trying IEEE Xplore URL: {ieee_doc_urls[0]}")
                        try:
                            import requests
                            response = requests.get(
                                ieee_doc_urls[0],
                                timeout=config.pdf_download_timeout,
                                headers={"User-Agent": handler._downloader._get_user_agent()},
                                allow_redirects=True
                            )
                            
                            if response.status_code == 200:
                                content_type = response.headers.get("Content-Type", "").lower()
                                if "text/html" in content_type:
                                    extracted_text = html_extractor.extract_text(
                                        response.content, 
                                        ieee_doc_urls[0]
                                    )
                                    
                                    if extracted_text:
                                        # Validate extracted text
                                        is_valid, validation_reason = html_extractor.is_valid_paper_content(
                                            extracted_text,
                                            config.html_text_min_length
                                        )
                                        
                                        if is_valid:
                                            # Save new extraction
                                            html_extractor.save_extracted_text(extracted_text, txt_file)
                                            word_count_new = len(extracted_text.split())
                                            logger.info(
                                                f"✓ Re-extracted {txt_file.name}: "
                                                f"{word_count} → {word_count_new} words"
                                            )
                                            results.append((citation_key, 're-extracted'))
                                            re_extracted = True
                                        else:
                                            logger.warning(
                                                f"Re-extraction validation failed for {citation_key}: "
                                                f"{validation_reason}"
                                            )
                        except Exception as e:
                            logger.debug(f"Re-extraction from IEEE URL failed: {e}")
                
                # Try original URL if IEEE extraction didn't work
                if not re_extracted and entry.url:
                    logger.info(f"Trying original URL: {entry.url}")
                    try:
                        import requests
                        response = requests.get(
                            entry.url,
                            timeout=config.pdf_download_timeout,
                            headers={"User-Agent": handler._downloader._get_user_agent()},
                            allow_redirects=True
                        )
                        
                        if response.status_code == 200:
                            content_type = response.headers.get("Content-Type", "").lower()
                            if "text/html" in content_type:
                                extracted_text = html_extractor.extract_text(
                                    response.content,
                                    entry.url
                                )
                                
                                if extracted_text:
                                    is_valid, validation_reason = html_extractor.is_valid_paper_content(
                                        extracted_text,
                                        config.html_text_min_length
                                    )
                                    
                                    if is_valid:
                                        html_extractor.save_extracted_text(extracted_text, txt_file)
                                        word_count_new = len(extracted_text.split())
                                        logger.info(
                                            f"✓ Re-extracted {txt_file.name}: "
                                            f"{word_count} → {word_count_new} words"
                                        )
                                        results.append((citation_key, 're-extracted'))
                                        re_extracted = True
                                    else:
                                        logger.warning(
                                            f"Re-extraction validation failed for {citation_key}: "
                                            f"{validation_reason}"
                                        )
                    except Exception as e:
                        logger.debug(f"Re-extraction from original URL failed: {e}")
                
                if not re_extracted:
                    logger.warning(
                        f"Could not re-extract {citation_key}. "
                        f"File remains with {word_count} words."
                    )
                    results.append((citation_key, 'failed'))
            else:
                logger.debug(f"File {txt_file.name} has {word_count} words, skipping")
                results.append((citation_key, 'skipped'))
                
        except Exception as e:
            logger.error(f"Error processing {txt_file.name}: {e}")
            results.append((citation_key, 'error'))
    
    # Summary
    re_extracted_count = sum(1 for _, status in results if status == 're-extracted')
    failed_count = sum(1 for _, status in results if status == 'failed')
    skipped_count = sum(1 for _, status in results if status == 'skipped')
    
    logger.info(
        f"Re-extraction complete: {re_extracted_count} re-extracted, "
        f"{failed_count} failed, {skipped_count} skipped"
    )
    
    return results

