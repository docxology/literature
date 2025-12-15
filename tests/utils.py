"""Shared test utilities for the test suite.

Provides helper functions and utilities for testing across all modules.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_sample_pdf(pdf_path: Path, content: str = "Test PDF Content") -> Path:
    """Create a sample PDF file for testing.
    
    Args:
        pdf_path: Path where PDF should be created.
        content: Text content to include in PDF.
        
    Returns:
        Path to created PDF file.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 750, content)
        c.drawString(100, 730, "This is a test PDF file.")
        c.save()
        return pdf_path
    except ImportError:
        # Fallback: create minimal PDF
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n"
            b"<< /Type /Catalog >>\n"
            b"endobj\n"
            b"xref\n"
            b"0 1\n"
            b"trailer\n"
            b"<< /Root 1 0 R >>\n"
            b"%%EOF\n"
        )
        return pdf_path


def create_sample_library_entry(
    citation_key: str = "test2023paper",
    title: str = "Test Paper",
    authors: List[str] = None,
    year: int = 2023
) -> Dict[str, Any]:
    """Create a sample library entry dictionary.
    
    Args:
        citation_key: Citation key for the entry.
        title: Paper title.
        authors: List of author names.
        year: Publication year.
        
    Returns:
        Dictionary representing a library entry.
    """
    if authors is None:
        authors = ["Author One", "Author Two"]
    
    return {
        "citation_key": citation_key,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": f"10.1234/{citation_key}",
        "source": "test",
        "url": f"https://example.com/{citation_key}",
        "abstract": f"This is a test abstract for {title}.",
        "venue": "Test Journal",
        "citation_count": 0,
    }


def create_sample_search_result(
    title: str = "Test Paper",
    authors: List[str] = None,
    year: int = 2023
) -> Dict[str, Any]:
    """Create a sample search result dictionary.
    
    Args:
        title: Paper title.
        authors: List of author names.
        year: Publication year.
        
    Returns:
        Dictionary representing a search result.
    """
    if authors is None:
        authors = ["Author One", "Author Two"]
    
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": f"This is a test abstract for {title}.",
        "url": "https://example.com/paper",
        "doi": "10.1234/test",
        "source": "test",
        "pdf_url": "https://example.com/paper.pdf",
        "venue": "Test Journal",
        "citation_count": 0,
    }


def assert_file_exists(file_path: Path, description: str = "file"):
    """Assert that a file exists, with helpful error message.
    
    Args:
        file_path: Path to check.
        description: Description of the file for error messages.
    """
    assert file_path.exists(), f"{description} not found at {file_path}"


def assert_file_contains(file_path: Path, text: str, description: str = "file"):
    """Assert that a file contains specific text.
    
    Args:
        file_path: Path to file.
        text: Text to search for.
        description: Description of the file for error messages.
    """
    assert_file_exists(file_path, description)
    content = file_path.read_text()
    assert text in content, f"{description} does not contain expected text: {text}"


