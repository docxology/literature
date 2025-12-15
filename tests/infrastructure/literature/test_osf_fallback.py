"""Tests for OSF.io URL transformation and HTML text extraction fallback."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from infrastructure.literature.pdf.fallbacks import transform_pdf_url, doi_to_pdf_urls
from infrastructure.literature.pdf.html_extractor import HTMLTextExtractor
from infrastructure.literature.html_parsers.osf import OSFParser


class TestOSFURLTransformation:
    """Test OSF.io URL transformation in fallbacks."""
    
    def test_osf_doi_to_pdf_urls(self):
        """Test OSF.io DOI conversion to direct download URLs."""
        doi = "10.31234/osf.io/tf4rq_v1"
        urls = doi_to_pdf_urls(doi)
        
        # Should include direct OSF.io download URL
        assert any("osf.io/tf4rq_v1/download" in url for url in urls)
        # Should be first in list (prioritized)
        assert "osf.io/tf4rq_v1/download" in urls[0]
    
    def test_osf_doi_alternative_prefix(self):
        """Test OSF.io DOI with alternative prefix (10.31219)."""
        doi = "10.31219/osf.io/abc123"
        urls = doi_to_pdf_urls(doi)
        
        assert any("osf.io/abc123/download" in url for url in urls)
    
    def test_osf_url_transformation(self):
        """Test OSF.io URL transformation in transform_pdf_url."""
        # Test direct OSF.io URL
        url = "https://osf.io/tf4rq_v1"
        candidates = transform_pdf_url(url)
        
        assert any("osf.io/tf4rq_v1/download" in c for c in candidates)
    
    def test_osf_doi_url_transformation(self):
        """Test OSF.io DOI URL transformation."""
        url = "https://doi.org/10.31234/osf.io/b5ykj_v2"
        candidates = transform_pdf_url(url)
        
        assert any("osf.io/b5ykj_v2/download" in c for c in candidates)
    
    def test_osf_case_insensitive(self):
        """Test OSF.io URL transformation is case insensitive."""
        url = "https://OSF.IO/TF4RQ_V1"
        candidates = transform_pdf_url(url)
        
        assert any("osf.io/tf4rq_v1/download" in c.lower() for c in candidates)


class TestHTMLTextExtractor:
    """Test HTML text extraction functionality."""
    
    def test_extract_simple_html(self):
        """Test extraction from simple HTML."""
        html = b"""
        <html>
        <head><title>Test Paper</title></head>
        <body>
            <h1>Introduction</h1>
            <p>This is a test paragraph with some content.</p>
            <p>Another paragraph with more text.</p>
        </body>
        </html>
        """
        
        extractor = HTMLTextExtractor()
        text = extractor.extract_text(html)
        
        assert "Test Paper" in text
        assert "Introduction" in text
        assert "test paragraph" in text
        assert "Another paragraph" in text
    
    def test_extract_removes_scripts(self):
        """Test that scripts and styles are removed."""
        html = b"""
        <html>
        <head>
            <script>alert('test');</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <p>Content here</p>
        </body>
        </html>
        """
        
        extractor = HTMLTextExtractor()
        text = extractor.extract_text(html)
        
        assert "alert" not in text
        assert "color: red" not in text
        assert "Content here" in text
    
    def test_extract_preserves_structure(self):
        """Test that heading structure is preserved."""
        html = b"""
        <html>
        <body>
            <h1>Main Title</h1>
            <h2>Section 1</h2>
            <p>Paragraph 1</p>
            <h2>Section 2</h2>
            <p>Paragraph 2</p>
        </body>
        </html>
        """
        
        extractor = HTMLTextExtractor()
        text = extractor.extract_text(html)
        
        # Verify all expected content is present
        assert "Main Title" in text, f"Main Title not found in extracted text: {text!r}"
        assert "Section 1" in text, f"Section 1 not found in extracted text: {text!r}"
        assert "Paragraph 1" in text, f"Paragraph 1 not found in extracted text: {text!r}"
        assert "Section 2" in text, f"Section 2 not found in extracted text: {text!r}"
        assert "Paragraph 2" in text, f"Paragraph 2 not found in extracted text: {text!r}"
        
        # Check that headings appear before their content (structure preservation)
        main_title_pos = text.find("Main Title")
        section1_pos = text.find("Section 1")
        para1_pos = text.find("Paragraph 1")
        
        assert main_title_pos >= 0, "Main Title not found in extracted text"
        assert section1_pos >= 0, "Section 1 not found in extracted text"
        assert para1_pos >= 0, "Paragraph 1 not found in extracted text"
        assert main_title_pos < section1_pos < para1_pos, (
            f"Structure not preserved: Main Title at {main_title_pos}, "
            f"Section 1 at {section1_pos}, Paragraph 1 at {para1_pos}"
        )
    
    def test_extract_cleans_whitespace(self):
        """Test that excessive whitespace is cleaned."""
        html = b"""
        <html>
        <body>
            <p>Paragraph    with    multiple    spaces</p>
        </body>
        </html>
        """
        
        extractor = HTMLTextExtractor()
        text = extractor.extract_text(html)
        
        # Should not have excessive spaces
        assert "   " not in text
        assert "Paragraph with multiple spaces" in text or "Paragraph" in text
    
    def test_save_extracted_text(self, tmp_path):
        """Test saving extracted text to file."""
        extractor = HTMLTextExtractor()
        text = "This is extracted text content."
        output_path = tmp_path / "test.txt"
        
        extractor.save_extracted_text(text, output_path)
        
        assert output_path.exists()
        assert output_path.read_text() == text
    
    def test_extract_empty_html(self):
        """Test extraction from empty HTML."""
        html = b"<html><body></body></html>"
        
        extractor = HTMLTextExtractor()
        text = extractor.extract_text(html)
        
        # Should return empty or minimal text
        assert isinstance(text, str)
    
    def test_extract_with_bs4_fallback(self):
        """Test that regex fallback works when BeautifulSoup4 unavailable."""
        # Create extractor with bs4 disabled (simulating unavailability)
        extractor = HTMLTextExtractor(has_bs4=False)
        html = b"<html><body><p>Test content</p></body></html>"
        text = extractor.extract_text(html)
        
        # Should still extract text using regex fallback
        # With lowered threshold (> 3 chars), "Test content" (12 chars) should be extracted
        assert len(text) > 0, f"Regex fallback returned empty text. Extracted: {text!r}"
        assert "Test content" in text, (
            f"Expected 'Test content' in extracted text, got: {text!r}"
        )


class TestOSFParser:
    """Test OSF.io HTML parser."""
    
    def test_detect_osf_url(self):
        """Test OSF.io URL detection."""
        parser = OSFParser()
        
        assert parser.detect_publisher("https://osf.io/tf4rq_v1")
        assert parser.detect_publisher("https://doi.org/10.31234/osf.io/abc123")
        assert not parser.detect_publisher("https://arxiv.org/abs/1234.5678")
    
    def test_extract_osf_download_url(self):
        """Test extraction of OSF.io download URLs from HTML."""
        html = b"""
        <html>
        <body>
            <a href="/tf4rq_v1/download">Download</a>
        </body>
        </html>
        """
        
        parser = OSFParser()
        base_url = "https://osf.io/tf4rq_v1"
        urls = parser.extract_pdf_urls(html, base_url)
        
        # Should include direct download URL
        assert any("osf.io/tf4rq_v1/download" in url for url in urls)
    
    def test_extract_osf_pdf_links(self):
        """Test extraction of PDF links from OSF.io HTML."""
        html = b"""
        <html>
        <body>
            <a href="/files/paper.pdf">PDF</a>
            <a href="/download">Download</a>
        </body>
        </html>
        """
        
        parser = OSFParser()
        base_url = "https://osf.io/project123"
        urls = parser.extract_pdf_urls(html, base_url)
        
        # Should find PDF and download links
        assert len(urls) > 0
        assert any(".pdf" in url.lower() or "download" in url.lower() for url in urls)


class TestOSFIntegration:
    """Integration tests for OSF.io fallback chain."""
    
    def test_osf_doi_to_download_url_chain(self):
        """Test complete chain from OSF.io DOI to download URL."""
        # Start with OSF.io DOI
        doi = "10.31234/osf.io/test123"
        
        # Convert DOI to URLs
        urls = doi_to_pdf_urls(doi)
        
        # Should generate direct download URL
        download_url = None
        for url in urls:
            if "osf.io/test123/download" in url:
                download_url = url
                break
        
        assert download_url is not None
        assert download_url.startswith("https://osf.io/test123/download")
    
    @pytest.mark.skip(reason="Requires network access - integration test")
    def test_osf_parser_with_real_url(self):
        """Test OSF parser with real OSF.io URL (requires network)."""
        # This would test with a real OSF.io page
        # Skipped by default to avoid network dependency in unit tests
        pass


