"""OSF.io (Open Science Framework) HTML parser.

Handles PDF URL extraction from OSF.io project pages.
OSF.io projects often have direct download links that can be extracted.
"""
from __future__ import annotations

import re
from typing import List
from urllib.parse import urljoin

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.html_parsers.base import BaseHTMLParser

logger = get_logger(__name__)


class OSFParser(BaseHTMLParser):
    """Parser for OSF.io (Open Science Framework) pages."""
    
    priority = 10  # Higher priority than generic parser
    
    def detect_publisher(self, url: str) -> bool:
        """Check if URL is from OSF.io.
        
        Args:
            url: URL to check.
            
        Returns:
            True if URL is from OSF.io.
        """
        return 'osf.io' in url.lower() or '10.3123' in url
    
    def extract_pdf_urls(self, html_content: bytes, base_url: str) -> List[str]:
        """Extract PDF URLs from OSF.io HTML content.
        
        OSF.io pages typically have:
        - Direct download links: /XXXXX/download
        - File browser links to PDF files
        - JavaScript-based download buttons
        
        Args:
            html_content: Raw HTML content as bytes.
            base_url: Base URL for resolving relative links.
            
        Returns:
            List of candidate PDF URLs found in HTML.
        """
        html_str = self._decode_html(html_content)
        candidates = []
        
        # Extract OSF.io project ID from base URL
        osf_id_match = re.search(r'osf\.io/([a-z0-9_]+)', base_url, re.IGNORECASE)
        osf_id = osf_id_match.group(1) if osf_id_match else None
        
        # Strategy 1: Direct download URL (most reliable for OSF.io)
        if osf_id:
            candidates.append(f"https://osf.io/{osf_id}/download")
        
        # Strategy 2: Look for download links in HTML
        # OSF.io often has links like: href="/XXXXX/download"
        download_link_patterns = [
            r'href=["\']([^"\']*download[^"\']*)["\']',
            r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'data-download-url=["\']([^"\']*)["\']',
            r'downloadUrl["\']?\s*[:=]\s*["\']([^"\']*)["\']',
        ]
        
        for pattern in download_link_patterns:
            matches = re.findall(pattern, html_str, re.IGNORECASE)
            for match in matches:
                if match:
                    full_url = urljoin(base_url, match)
                    # Filter for PDF or download URLs
                    if '.pdf' in full_url.lower() or 'download' in full_url.lower():
                        if full_url not in candidates:
                            candidates.append(full_url)
        
        # Strategy 3: Look for file browser links
        # OSF.io file browser often has links to files
        file_link_patterns = [
            r'<a[^>]*href=["\']([^"\']*files[^"\']*\.pdf[^"\']*)["\']',
            r'<a[^>]*data-file-name=["\'][^"\']*\.pdf[^"\']*["\'][^>]*href=["\']([^"\']*)["\']',
        ]
        
        for pattern in file_link_patterns:
            matches = re.findall(pattern, html_str, re.IGNORECASE)
            for match in matches:
                if match:
                    full_url = urljoin(base_url, match)
                    if full_url not in candidates:
                        candidates.append(full_url)
        
        # Strategy 4: Use generic PDF link finder as fallback
        candidates.extend(self._find_pdf_links(html_str, base_url))
        
        # Filter and return valid URLs
        return self._filter_valid_urls(candidates)


