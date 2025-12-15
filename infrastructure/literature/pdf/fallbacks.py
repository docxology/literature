"""Fallback URL strategies for PDF downloads."""
from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urljoin

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.sources import SearchResult, UnpaywallSource, ArxivSource, BiorxivSource
from infrastructure.literature.core.config import LiteratureConfig

logger = get_logger(__name__)


def doi_to_pdf_urls(doi: str) -> List[str]:
    """Convert a DOI to potential PDF download URLs for common publishers.

    Args:
        doi: The DOI string (with or without doi.org prefix)

    Returns:
        List of potential PDF URLs for the DOI
    """
    # Clean the DOI
    doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '').replace('doi:', '')

    candidates = []

    # Elsevier/ScienceDirect
    if doi.startswith('10.1016/') or doi.startswith('10.1017/'):
        # Extract PII from DOI for ScienceDirect
        parts = doi.split('/')
        if len(parts) >= 2:
            pii = parts[-1]
            candidates.append(f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true")

    # Springer
    if 'springer' in doi.lower() or doi.startswith('10.1007/') or doi.startswith('10.1038/'):
        candidates.append(f"https://link.springer.com/content/pdf/{doi}.pdf")

    # Wiley
    if doi.startswith('10.1002/') or doi.startswith('10.1111/') or 'wiley' in doi.lower():
        candidates.append(f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}")

    # PLOS
    if doi.startswith('10.1371/'):
        candidates.append(f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable")

    # Frontiers
    if doi.startswith('10.3389/'):
        candidates.append(f"https://www.frontiersin.org/articles/{doi}/pdf")

    # MDPI (case-insensitive DOI handling)
    doi_lower = doi.lower()
    if doi_lower.startswith('10.3390/'):
        # Try both lowercase and original case
        doi_part = doi.split('/', 1)[1]
        candidates.append(f"https://www.mdpi.com/article/10.3390/{doi_part}/pdf")
        # Alternative pattern with uppercase journal code
        if doi_part != doi_part.upper():
            candidates.append(f"https://www.mdpi.com/article/10.3390/{doi_part.upper()}/pdf")
        # Try direct PDF URL pattern
        mdpi_parts = doi_part.split('/')
        if len(mdpi_parts) >= 3:
            # Pattern: journal/volume/issue/article
            candidates.append(f"https://www.mdpi.com/{mdpi_parts[0]}/{mdpi_parts[1]}/{mdpi_parts[2]}/{mdpi_parts[-1]}/pdf")

    # Nature
    if doi.startswith('10.1038/'):
        candidates.append(f"https://www.nature.com/articles/{doi}.pdf")

    # Oxford University Press
    if doi.startswith('10.1093/'):
        candidates.append(f"https://academic.oup.com/view-pdf/doi/{doi}")

    # IEEE Xplore
    if doi.startswith('10.1109/'):
        # IEEE DOIs format: 10.1109/JOURNAL.YEAR.ARTICLE_ID
        # Extract article ID (last part after final dot)
        doi_parts = doi.split('/')
        if len(doi_parts) >= 2:
            article_part = doi_parts[-1]
            # Article ID is typically the last segment after the final dot
            if '.' in article_part:
                article_id = article_part.split('.')[-1]
                # Try IEEE Xplore document page (for HTML extraction)
                candidates.insert(0, f"https://ieeexplore.ieee.org/document/{article_id}")
                # Try PDF download URLs
                candidates.append(f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?arnumber={article_id}")
                candidates.append(f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={article_id}")
            # Also try with full DOI path
            candidates.append(f"https://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber={article_part.split('.')[-1] if '.' in article_part else article_part}")

    # OSF.io (Open Science Framework)
    # Pattern: 10.31234/osf.io/XXXXX or 10.31219/osf.io/XXXXX
    osf_match = re.search(r'10\.312(?:34|19)/osf\.io/([a-z0-9_]+)', doi, re.IGNORECASE)
    if osf_match:
        osf_id = osf_match.group(1)
        # OSF.io direct download URL (preferred)
        candidates.insert(0, f"https://osf.io/{osf_id}/download")
        # Also try the DOI resolver as fallback
        candidates.append(f"https://doi.org/{doi}")
    else:
        # Generic DOI resolver fallback (only if not OSF.io)
        candidates.append(f"https://doi.org/{doi}")

    return candidates


def transform_pdf_url(url: str) -> List[str]:
    """Transform URL to multiple candidate PDF URLs for known sources.

    Generates a list of alternative URLs to try when the primary URL fails.
    Handles PMC, Europe PMC, and other common academic sources.

    Args:
        url: Original URL that might be an HTML landing page.

    Returns:
        List of candidate URLs to try, in priority order.
    """
    candidates: List[str] = []

    # Extract PMC ID from various URL patterns
    pmc_id = None

    # Pattern 1: www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/
    pmc_match = re.search(r'(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)', url)
    if pmc_match:
        pmc_id = pmc_match.group(1)

    # Pattern 2: pmc.ncbi.nlm.nih.gov/articles/PMC123456/
    if not pmc_id:
        pmc_match = re.search(r'pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d+)', url)
        if pmc_match:
            pmc_id = pmc_match.group(1)

    # Pattern 3: europepmc.org/article/PMC/123456 or europepmc.org/articles/PMC123456
    if not pmc_id:
        pmc_match = re.search(r'europepmc\.org/(?:article|articles)/(?:PMC/)?(\d+|PMC\d+)', url)
        if pmc_match:
            pmc_id_raw = pmc_match.group(1)
            pmc_id = pmc_id_raw if pmc_id_raw.startswith('PMC') else f"PMC{pmc_id_raw}"

    # Generate PMC URL candidates if we found a PMC ID
    if pmc_id:
        # Direct NCBI PDF endpoints (multiple patterns publishers use)
        candidates.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/")
        candidates.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/main.pdf")

        # PMC new domain
        candidates.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/")
        candidates.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/pdf/main.pdf")

        # Europe PMC (often more accessible)
        candidates.append(f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmc_id}&blobtype=pdf")
        candidates.append(f"https://europepmc.org/articles/{pmc_id}?pdf=render")

        # FTP-style direct access (older papers)
        pmc_num = pmc_id.replace('PMC', '')
        candidates.append(f"https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_pdf/{pmc_num}.pdf")

    # Handle Elsevier/ScienceDirect patterns
    sd_match = re.search(r'sciencedirect\.com/science/article/pii/([A-Z0-9]+)', url, re.IGNORECASE)
    if sd_match:
        pii = sd_match.group(1)
        # Elsevier open access PDF endpoint (works for some OA papers)
        candidates.append(f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true")

    # Handle MDPI patterns (often open access)
    # Pattern 1: mdpi.com/journal/volume/issue/article
    mdpi_match = re.search(r'mdpi\.com/(\d+-\d+)/(\d+)/(\d+)/(\d+)', url)
    if mdpi_match:
        journal, volume, issue, article = mdpi_match.groups()
        candidates.append(f"https://www.mdpi.com/{journal}/{volume}/{issue}/{article}/pdf")
    
    # Pattern 2: mdpi.com/article/10.3390/...
    mdpi_article_match = re.search(r'mdpi\.com/article/(10\.3390/[^/]+)', url, re.IGNORECASE)
    if mdpi_article_match:
        doi_part = mdpi_article_match.group(1)
        candidates.append(f"https://www.mdpi.com/article/{doi_part}/pdf")
        # Try with version parameter (sometimes needed)
        candidates.append(f"https://www.mdpi.com/article/{doi_part}/pdf?version={int(__import__('time').time())}")
    
    # Pattern 3: Direct PDF URL with version
    mdpi_pdf_match = re.search(r'mdpi\.com/(\d+-\d+)/(\d+)/(\d+)/(\d+)/pdf', url)
    if mdpi_pdf_match:
        journal, volume, issue, article = mdpi_pdf_match.groups()
        # Try with version parameter
        candidates.append(f"https://www.mdpi.com/{journal}/{volume}/{issue}/{article}/pdf?version={int(__import__('time').time())}")

    # Handle Frontiers patterns (open access)
    frontiers_match = re.search(r'frontiersin\.org/(?:articles|journals)/.+/full$', url)
    if frontiers_match:
        candidates.append(url.replace('/full', '/pdf'))

    # Handle arXiv patterns (ensure we have the PDF link)
    # arXiv IDs can be: YYMM.NNNNN (new format) or category/YYMMNNN (old format)
    # Match both abstract and PDF URLs, extract ID and normalize to PDF URL
    arxiv_match = re.search(r'arxiv\.org/(?:abs|pdf)/((?:\d{4}\.\d{4,5})|(?:\w+-\w+/\d{7}))(?:v\d+)?', url)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        # Remove version suffix if present in ID
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        # Primary arXiv PDF URL (preferred)
        candidates.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        # Fallback to export.arxiv.org (legacy, but sometimes more reliable)
        candidates.append(f"https://export.arxiv.org/pdf/{arxiv_id}.pdf")

    # Handle bioRxiv/medRxiv patterns
    biorxiv_match = re.search(r'(biorxiv|medrxiv)\.org/content/([\d.]+v?\d*)', url)
    if biorxiv_match:
        server, content_id = biorxiv_match.groups()
        candidates.append(f"https://www.{server}.org/content/{content_id}.full.pdf")

    # Handle OSF.io (Open Science Framework) patterns
    # Pattern 1: Direct OSF.io URLs (osf.io/XXXXX)
    osf_match = re.search(r'osf\.io/([a-z0-9_]+)', url, re.IGNORECASE)
    if osf_match:
        osf_id = osf_match.group(1)
        # OSF.io direct download URL
        candidates.append(f"https://osf.io/{osf_id}/download")
    
    # Pattern 2: DOI URLs containing osf.io (doi.org/10.31234/osf.io/XXXXX or 10.31219/osf.io/XXXXX)
    osf_doi_match = re.search(r'10\.312(?:34|19)/osf\.io/([a-z0-9_]+)', url, re.IGNORECASE)
    if osf_doi_match:
        osf_id = osf_doi_match.group(1)
        candidates.append(f"https://osf.io/{osf_id}/download")

    # Handle IEEE Xplore patterns
    # Pattern 1: ieee.org/document/XXXXX
    ieee_match = re.search(r'ieee\.org/document/(\d+)', url, re.IGNORECASE)
    if ieee_match:
        doc_id = ieee_match.group(1)
        # Try direct PDF access (works for some open access papers)
        candidates.append(f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={doc_id}")
        candidates.append(f"https://ieeexplore.ieee.org/ielx7/{doc_id[:2]}/{doc_id}/0{doc_id}.pdf")
    
    # Pattern 2: doi.org/10.1109/... (IEEE DOIs)
    ieee_doi_match = re.search(r'10\.1109/([^/]+)', url, re.IGNORECASE)
    if ieee_doi_match:
        ieee_path = ieee_doi_match.group(1)
        # Try IEEE Xplore document lookup
        candidates.append(f"https://ieeexplore.ieee.org/document/{ieee_path.split('.')[-1]}")

    # Handle Preprints.org patterns
    # Pattern 1: preprints.org/manuscript/XXXXX/download_pub
    preprints_match = re.search(r'preprints\.org/(?:frontend/)?manuscript/([a-z0-9]+)/download_pub', url, re.IGNORECASE)
    if preprints_match:
        manuscript_id = preprints_match.group(1)
        # Try alternative download patterns
        candidates.append(f"https://www.preprints.org/manuscript/{manuscript_id}/download")
        candidates.append(f"https://www.preprints.org/frontend/manuscript/{manuscript_id}/download_pub")
    
    # Pattern 2: doi.org/10.20944/preprints...
    preprints_doi_match = re.search(r'10\.20944/preprints(\d+)\.(\d+)\.v(\d+)', url, re.IGNORECASE)
    if preprints_doi_match:
        year, number, version = preprints_doi_match.groups()
        # Preprints.org doesn't have direct PDF URLs from DOI, but we can try the manuscript page
        candidates.append(f"https://www.preprints.org/manuscript/{number}/download")

    # Remove duplicates while preserving order, and exclude original URL
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen and c != url:
            seen.add(c)
            unique_candidates.append(c)

    return unique_candidates


class PDFFallbackStrategies:
    """Fallback strategies for finding PDFs when primary URLs fail."""
    
    def __init__(self, config: LiteratureConfig):
        """Initialize fallback sources.
        
        Args:
            config: Literature configuration.
        """
        self.config = config
        
        # Initialize fallback sources
        self._unpaywall: Optional[UnpaywallSource] = None
        if config.use_unpaywall and config.unpaywall_email:
            self._unpaywall = UnpaywallSource(config)
        
        # arXiv fallback for title-based search
        self._arxiv = ArxivSource(config)
        
        # bioRxiv/medRxiv fallback for DOI-based lookup
        self._biorxiv = BiorxivSource(config)
    
    def get_unpaywall_url(self, doi: str) -> Optional[str]:
        """Get PDF URL from Unpaywall if available.
        
        Args:
            doi: DOI string.
            
        Returns:
            PDF URL if found, else None.
        """
        if not self._unpaywall:
            return None
        
        return self._unpaywall.get_pdf_url(doi)
    
    def get_arxiv_fallback(self, result: SearchResult) -> Optional[str]:
        """Try to find an arXiv preprint version by title search.

        Args:
            result: SearchResult with title to search for.

        Returns:
            PDF URL if arXiv preprint found, else None.
        """
        if not result.title:
            return None

        try:
            logger.debug(f"Trying arXiv fallback for: {result.title[:50]}...")
            arxiv_match = self._arxiv.search_by_title(result.title)

            if arxiv_match:
                # Use pdf_url if available, otherwise transform the abstract URL
                if arxiv_match.pdf_url:
                    logger.info(f"Found arXiv preprint with PDF URL: {arxiv_match.url}")
                    return arxiv_match.pdf_url
                else:
                    # Transform abstract URL to PDF URL
                    transformed_urls = transform_pdf_url(arxiv_match.url)
                    if transformed_urls:
                        logger.info(f"Found arXiv preprint, transformed to PDF URL: {arxiv_match.url} -> {transformed_urls[0]}")
                        return transformed_urls[0]

            return None
        except Exception as e:
            logger.debug(f"arXiv fallback failed: {e}")
            return None
    
    def get_biorxiv_fallback(self, result: SearchResult) -> Optional[str]:
        """Try to find a bioRxiv/medRxiv preprint version by DOI.
        
        Args:
            result: SearchResult with DOI to look up.
            
        Returns:
            PDF URL if preprint found, else None.
        """
        if not result.doi:
            return None
        
        try:
            logger.debug(f"Trying bioRxiv/medRxiv fallback for DOI: {result.doi}")
            biorxiv_match = self._biorxiv.search_by_doi(result.doi)
            
            if biorxiv_match and biorxiv_match.pdf_url:
                logger.info(f"Found bioRxiv/medRxiv preprint: {biorxiv_match.url}")
                return biorxiv_match.pdf_url
            
            # Also try title search as fallback
            if result.title:
                biorxiv_match = self._biorxiv.search_by_title(result.title)
                if biorxiv_match and biorxiv_match.pdf_url:
                    logger.info(f"Found bioRxiv/medRxiv preprint by title: {biorxiv_match.url}")
                    return biorxiv_match.pdf_url
            
            return None
        except Exception as e:
            logger.debug(f"bioRxiv/medRxiv fallback failed: {e}")
            return None










