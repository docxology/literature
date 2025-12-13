"""HTML text extraction utilities for fallback when PDFs are unavailable."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from infrastructure.core.logging_utils import get_logger

logger = get_logger(__name__)

# Try to import BeautifulSoup4 for better HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.debug("BeautifulSoup4 not available, using regex-based HTML extraction")


class HTMLTextExtractor:
    """Extract main text content from HTML pages.
    
    Designed for academic paper HTML pages when PDFs are not available.
    Removes navigation, headers, footers, scripts, and styles while preserving
    the main content structure.
    """
    
    def __init__(self):
        """Initialize HTML text extractor."""
        self.has_bs4 = HAS_BS4
    
    def extract_text(self, html_content: bytes, base_url: Optional[str] = None) -> str:
        """Extract main text content from HTML.
        
        Args:
            html_content: Raw HTML content as bytes.
            base_url: Optional base URL for context (not currently used but kept for API consistency).
            
        Returns:
            Extracted text content as a single string.
        """
        if self.has_bs4:
            return self._extract_with_bs4(html_content)
        else:
            return self._extract_with_regex(html_content)
    
    def _extract_with_bs4(self, html_content: bytes) -> str:
        """Extract text using BeautifulSoup4 for better parsing.
        
        Args:
            html_content: Raw HTML content as bytes.
            
        Returns:
            Extracted text content.
        """
        try:
            # Decode HTML content
            html_str = html_content.decode('utf-8', errors='ignore')
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_str, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find main content area
            # Common academic paper HTML structures
            main_content = None
            
            # Try common content selectors
            content_selectors = [
                'main',
                'article',
                '[role="main"]',
                '.content',
                '.main-content',
                '.paper-content',
                '.article-content',
                '#content',
                '#main-content',
                '.abstract',
                '.paper-body',
            ]
            
            for selector in content_selectors:
                try:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                except Exception:
                    continue
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.find('body')
            
            if not main_content:
                # Fallback to entire document
                main_content = soup
            
            # Extract text with structure preservation
            text_parts = []
            
            # Extract title if available
            title = soup.find('title')
            if title:
                text_parts.append(title.get_text().strip())
                text_parts.append("=" * len(title.get_text().strip()))
                text_parts.append("")
            
            # Extract headings and paragraphs
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
                element_text = element.get_text(separator=' ', strip=True)
                if element_text and len(element_text) > 10:  # Filter out very short fragments
                    # Add extra spacing for headings
                    if element.name.startswith('h'):
                        text_parts.append("")
                        text_parts.append(element_text)
                        text_parts.append("")
                    else:
                        text_parts.append(element_text)
            
            # Join and clean up
            text = '\n'.join(text_parts)
            text = self._clean_text(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"BeautifulSoup4 extraction failed: {e}, falling back to regex")
            return self._extract_with_regex(html_content)
    
    def _extract_with_regex(self, html_content: bytes) -> str:
        """Extract text using regex patterns (fallback when BeautifulSoup4 unavailable).
        
        Args:
            html_content: Raw HTML content as bytes.
            
        Returns:
            Extracted text content.
        """
        try:
            # Decode HTML content
            html_str = html_content.decode('utf-8', errors='ignore')
            
            # Remove script and style tags
            html_str = re.sub(r'<script[^>]*>.*?</script>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
            html_str = re.sub(r'<style[^>]*>.*?</style>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
            html_str = re.sub(r'<nav[^>]*>.*?</nav>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
            html_str = re.sub(r'<header[^>]*>.*?</header>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
            html_str = re.sub(r'<footer[^>]*>.*?</footer>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
            html_str = re.sub(r'<aside[^>]*>.*?</aside>', '', html_str, flags=re.DOTALL | re.IGNORECASE)
            
            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html_str, re.IGNORECASE | re.DOTALL)
            text_parts = []
            if title_match:
                title = self._strip_html_tags(title_match.group(1))
                if title:
                    text_parts.append(title)
                    text_parts.append("=" * len(title))
                    text_parts.append("")
            
            # Extract headings
            heading_patterns = [
                (r'<h1[^>]*>(.*?)</h1>', 'h1'),
                (r'<h2[^>]*>(.*?)</h2>', 'h2'),
                (r'<h3[^>]*>(.*?)</h3>', 'h3'),
            ]
            
            for pattern, tag in heading_patterns:
                matches = re.finditer(pattern, html_str, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    heading_text = self._strip_html_tags(match.group(1))
                    if heading_text and len(heading_text) > 3:
                        text_parts.append("")
                        text_parts.append(heading_text)
                        text_parts.append("")
            
            # Extract paragraphs
            para_matches = re.finditer(r'<p[^>]*>(.*?)</p>', html_str, re.IGNORECASE | re.DOTALL)
            for match in para_matches:
                para_text = self._strip_html_tags(match.group(1))
                if para_text and len(para_text) > 20:  # Filter out very short paragraphs
                    text_parts.append(para_text)
            
            # Extract div content (as fallback for paragraphs)
            div_matches = re.finditer(r'<div[^>]*class=["\'](?:content|main|article|paper|abstract)[^"\']*["\'][^>]*>(.*?)</div>', 
                                      html_str, re.IGNORECASE | re.DOTALL)
            for match in div_matches:
                div_text = self._strip_html_tags(match.group(1))
                if div_text and len(div_text) > 50:
                    text_parts.append(div_text)
            
            # Join and clean up
            text = '\n'.join(text_parts)
            text = self._clean_text(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Regex-based HTML extraction failed: {e}")
            # Last resort: strip all HTML tags
            try:
                html_str = html_content.decode('utf-8', errors='ignore')
                text = self._strip_html_tags(html_str)
                return self._clean_text(text)
            except Exception:
                return ""
    
    def _strip_html_tags(self, html: str) -> str:
        """Remove HTML tags from text.
        
        Args:
            html: HTML string.
            
        Returns:
            Text with HTML tags removed.
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Decode HTML entities
        try:
            import html as html_module
            text = html_module.unescape(text)
        except Exception:
            pass
        return text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text.
            
        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove common unwanted patterns
        text = re.sub(r'Skip to (main )?content', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Cookie (settings|preferences)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Accept (all )?cookies?', '', text, flags=re.IGNORECASE)
        
        # Remove very short lines (likely navigation fragments)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:
                cleaned_lines.append(line)
            elif not line:
                # Preserve paragraph breaks
                if cleaned_lines and cleaned_lines[-1]:
                    cleaned_lines.append("")
        
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup
        text = text.strip()
        
        return text
    
    def save_extracted_text(self, text: str, output_path: Path) -> None:
        """Save extracted text to file.
        
        Args:
            text: Extracted text content.
            output_path: Path to save text file.
            
        Raises:
            OSError: If file cannot be written.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding='utf-8')
            logger.info(f"Saved extracted HTML text to {output_path} ({len(text):,} characters)")
        except Exception as e:
            logger.error(f"Failed to save extracted text to {output_path}: {e}")
            raise

