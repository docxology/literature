"""
PDF validation module - Minimal version for literature repo.

This module provides PDF text extraction functionality needed by the
literature summarization system. It's a minimal subset of the full
validation module from the template repo.
"""

from pathlib import Path
import io
import contextlib
from infrastructure.core.logging_utils import get_logger


class PDFValidationError(Exception):
    """Raised when PDF validation encounters an error."""
    pass


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text content from a PDF file.

    Attempts to extract text using available PDF parsing libraries in order:
    pdfplumber → pypdf → PyPDF2. Suppresses harmless pypdf warnings.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a single string

    Raises:
        PDFValidationError: If file doesn't exist or text extraction fails
    """
    if not pdf_path.exists():
        raise PDFValidationError(f"PDF file not found: {pdf_path}")

    logger = get_logger(__name__)

    # Try pdfplumber first (best quality)
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return '\n'.join(text_parts)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}, trying alternatives...")

    # Try pypdf (newer PyPDF2) with warning suppression
    try:
        from pypdf import PdfReader

        with open(pdf_path, 'rb') as file:
            # Capture stderr to suppress pypdf warnings (e.g., "Ignoring wrong pointing object")
            # These are harmless and indicate pypdf is gracefully handling malformed PDF objects
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                pdf_reader = PdfReader(file)
                text_parts = []

                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            # Log suppressed warnings at DEBUG level for troubleshooting
            captured_warnings = stderr_capture.getvalue().strip()
            if captured_warnings:
                logger.debug(f"Suppressed pypdf warnings for {pdf_path.name}: {captured_warnings}")

            return '\n'.join(text_parts)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"pypdf extraction failed: {e}")

    # No PDF library available
    raise PDFValidationError(
        "No PDF parsing library available. Install one of: pdfplumber or pypdf.\n"
        "Installation commands:\n"
        "  pip install pdfplumber  # Recommended: best quality extraction\n"
        "  pip install pypdf        # Included in project dependencies\n"
        "\n"
        "Note: pypdf>=5.0 should be installed automatically with this project. "
        "If you see this error, try: pip install -e . or pip install pypdf"
    )

