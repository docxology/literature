"""HTML text extraction utilities for fallback when PDFs are unavailable."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from infrastructure.core.logging_utils import get_logger

logger = get_logger(__name__)

# Try to import BeautifulSoup4 for better HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
    if TYPE_CHECKING:
        from bs4 import BeautifulSoup as BeautifulSoupType
    else:
        BeautifulSoupType = BeautifulSoup
except ImportError:
    HAS_BS4 = False
    BeautifulSoupType = None  # type: ignore
    logger.debug("BeautifulSoup4 not available, using regex-based HTML extraction")


class HTMLTextExtractor:
    """Extract main text content from HTML pages.
    
    Designed for academic paper HTML pages when PDFs are not available.
    Removes navigation, headers, footers, scripts, and styles while preserving
    the main content structure.
    """
    
    def __init__(self, has_bs4: Optional[bool] = None):
        """Initialize HTML text extractor.
        
        Args:
            has_bs4: Override BeautifulSoup4 availability (for testing).
                     If None, uses global HAS_BS4 value.
        """
        self.has_bs4 = has_bs4 if has_bs4 is not None else HAS_BS4
    
    def extract_text(self, html_content: bytes, base_url: Optional[str] = None) -> str:
        """Extract main text content from HTML.
        
        Args:
            html_content: Raw HTML content as bytes.
            base_url: Optional base URL for publisher-specific extraction.
            
        Returns:
            Extracted text content as a single string.
        """
        if self.has_bs4:
            return self._extract_with_bs4(html_content, base_url)
        else:
            return self._extract_with_regex(html_content)
    
    def _extract_with_bs4(self, html_content: bytes, base_url: Optional[str] = None) -> str:
        """Extract text using BeautifulSoup4 for better parsing.
        
        Args:
            html_content: Raw HTML content as bytes.
            base_url: Optional base URL for publisher-specific extraction.
            
        Returns:
            Extracted text content.
        """
        try:
            # Decode HTML content
            html_str = html_content.decode('utf-8', errors='ignore')
            
            # Check if this is an IEEE Xplore page
            is_ieee = base_url and ('ieeexplore.ieee.org' in base_url or 'ieee.org' in base_url)
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_str, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # IEEE Xplore specific extraction
            if is_ieee:
                return self._extract_ieee_xplore(soup)
            
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
                if element_text:
                    # Preserve all headings (important for structure), filter short paragraphs/divs
                    if element.name.startswith('h'):
                        # Headings are always included regardless of length
                        text_parts.append("")
                        text_parts.append(element_text)
                        text_parts.append("")
                    elif len(element_text) > 10:  # Filter out very short fragments for paragraphs/divs
                        text_parts.append(element_text)
            
            # Join and clean up
            text = '\n'.join(text_parts)
            text = self._clean_text(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"BeautifulSoup4 extraction failed: {e}, falling back to regex")
            return self._extract_with_regex(html_content)
    
    def _extract_ieee_xplore(self, soup: "BeautifulSoupType") -> str:
        """Extract full paper content from IEEE Xplore HTML page.
        
        Args:
            soup: BeautifulSoup parsed HTML.
            
        Returns:
            Extracted text content.
        """
        text_parts = []
        
        # Extract title
        title_elem = soup.find('h1', class_='document-title') or soup.find('h1') or soup.find('title')
        if title_elem:
            title = title_elem.get_text(strip=True)
            if title and len(title) > 5:
                text_parts.append(title)
                text_parts.append("=" * len(title))
                text_parts.append("")
        
        # Extract authors
        authors_elem = soup.find('div', class_='authors') or soup.find('div', {'data-testid': 'authors'})
        if authors_elem:
            authors = authors_elem.get_text(strip=True)
            if authors:
                text_parts.append(f"Authors: {authors}")
                text_parts.append("")
        
        # Extract abstract - IEEE Xplore specific selectors
        abstract_selectors = [
            'div.abstract',
            'section.abstract',
            'div[data-testid="abstract"]',
            '.abstract-text',
            '#abstract',
            'div.u-mb-1',  # Common IEEE abstract container
        ]
        abstract_found = False
        for selector in abstract_selectors:
            abstract_elem = soup.select_one(selector)
            if abstract_elem:
                abstract_text = abstract_elem.get_text(separator=' ', strip=True)
                if abstract_text and len(abstract_text) > 50:
                    text_parts.append("Abstract")
                    text_parts.append("-" * 50)
                    text_parts.append(abstract_text)
                    text_parts.append("")
                    abstract_found = True
                    break
        
        # Extract main content - IEEE Xplore article body
        content_selectors = [
            'div.article-body',
            'div.article-content',
            'div[data-testid="article-body"]',
            'section.article-body',
            'div.xpl-article-content',
            'div.article-text',
            'div.u-pb-1',  # Common IEEE content container
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no specific content area, try to find sections
        if not main_content:
            # Look for sections with headings
            sections = soup.find_all(['section', 'div'], class_=re.compile(r'section|content|body', re.I))
            if sections:
                main_content = soup.new_tag('div')
                for section in sections:
                    main_content.append(section)
        
        # Extract all text from main content
        if main_content:
            # Extract all paragraphs and headings
            for elem in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section']):
                elem_text = elem.get_text(separator=' ', strip=True)
                if elem_text and len(elem_text) > 20:  # Filter short fragments
                    # Check if it's a heading
                    if elem.name and elem.name.startswith('h'):
                        text_parts.append("")
                        text_parts.append(elem_text)
                        text_parts.append("")
                    elif elem.name == 'p' or (elem.name == 'div' and len(elem_text) > 50):
                        text_parts.append(elem_text)
        
        # If still no content, try extracting from body but filter navigation
        if len(text_parts) < 5 or sum(len(p) for p in text_parts) < 1000:
            body = soup.find('body')
            if body:
                # Remove navigation and footer elements
                for nav in body.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style']):
                    nav.decompose()
                
                # Extract all meaningful text
                for elem in body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
                    elem_text = elem.get_text(separator=' ', strip=True)
                    # Filter out navigation text (short, repetitive)
                    if elem_text and len(elem_text) > 30 and not any(
                        nav_word in elem_text.lower() for nav_word in 
                        ['cookie', 'privacy', 'terms', 'sign in', 'register', 'account', 'purchase']
                    ):
                        if elem.name and elem.name.startswith('h'):
                            text_parts.append("")
                            text_parts.append(elem_text)
                            text_parts.append("")
                        else:
                            text_parts.append(elem_text)
        
        # Join and clean
        text = '\n'.join(text_parts)
        text = self._clean_text(text)
        
        return text
    
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
            
            # Extract elements in document order to preserve structure
            # Use a single pattern that matches all relevant elements and sorts by position
            element_patterns = [
                (r'<h1[^>]*>(.*?)</h1>', 'h1'),
                (r'<h2[^>]*>(.*?)</h2>', 'h2'),
                (r'<h3[^>]*>(.*?)</h3>', 'h3'),
                (r'<h4[^>]*>(.*?)</h4>', 'h4'),
                (r'<h5[^>]*>(.*?)</h5>', 'h5'),
                (r'<h6[^>]*>(.*?)</h6>', 'h6'),
                (r'<p[^>]*>(.*?)</p>', 'p'),
            ]
            
            # Find all elements with their positions
            elements = []
            for pattern, tag in element_patterns:
                for match in re.finditer(pattern, html_str, re.IGNORECASE | re.DOTALL):
                    text = self._strip_html_tags(match.group(1))
                    if text and len(text) > 3:
                        elements.append((match.start(), tag, text))
            
            # Sort by position to preserve document order
            elements.sort(key=lambda x: x[0])
            
            # Add elements in order
            for pos, tag, text in elements:
                if tag.startswith('h'):
                    # Heading - add spacing (preserve all headings for structure)
                    text_parts.append("")
                    text_parts.append(text)
                    text_parts.append("")
                elif tag == 'p' and len(text) > 3:  # Lower threshold for regex fallback to handle test cases
                    text_parts.append(text)
            
            # Extract div content (as fallback for paragraphs) - only if we didn't get much content
            if len(text_parts) < 5:
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
    
    def is_valid_paper_content(self, text: str, min_length: int = 2000) -> tuple[bool, str]:
        """Validate that extracted text appears to be a full academic paper.
        
        Checks both length and content quality to ensure the extracted text
        is not just a webpage header/footer but contains actual paper content.
        
        Args:
            text: Extracted text content to validate.
            min_length: Minimum character length required (default: 2000).
            
        Returns:
            Tuple of (is_valid: bool, reason: str). If valid, reason is empty string.
            If invalid, reason explains why (too short, missing sections, etc.).
        """
        if not text or not text.strip():
            return False, "extracted text is empty"
        
        text_length = len(text)
        
        # Check minimum length
        if text_length < min_length:
            return False, f"extracted {text_length:,} characters (minimum {min_length:,} required)"
        
        # Check for common academic paper section keywords (case-insensitive)
        text_lower = text.lower()
        paper_section_keywords = [
            "abstract",
            "introduction",
            "methods",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
            "background",
            "related work",
            "related works",
            "literature review",
            "experiments",
            "experimental",
            "evaluation",
            "analysis",
            "method",
            "approach",
        ]
        
        found_sections = []
        for keyword in paper_section_keywords:
            if keyword in text_lower:
                found_sections.append(keyword)
        
        if not found_sections:
            return False, f"extracted {text_length:,} characters but missing common paper sections (abstract, introduction, methods, etc.)"
        
        # Check for excessive repetition of short phrases (likely navigation/header text)
        # Split into words and check for high-frequency short phrases
        words = text.split()
        if len(words) < 100:  # Too few words to be a full paper
            return False, f"extracted {text_length:,} characters but only {len(words)} words (likely incomplete)"
        
        # Check for repetitive short phrases (common in navigation headers)
        # Count occurrences of 2-3 word phrases
        phrase_counts = {}
        for i in range(len(words) - 1):
            phrase = " ".join(words[i:i+2]).lower()
            if len(phrase) < 20:  # Only check short phrases
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # If any short phrase appears more than 10% of the time, it's likely navigation text
        max_repetition = max(phrase_counts.values()) if phrase_counts else 0
        if max_repetition > len(words) * 0.1:
            return False, f"extracted {text_length:,} characters but contains excessive repetition (likely navigation/header text)"
        
        # Validation passed
        return True, ""
    
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


