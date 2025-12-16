"""Tests for output validation and quality control in summarization.

Tests methods/tools output validation, quote validation, summary quality checks,
and output size anomaly detection.
"""
import pytest
from unittest.mock import Mock, patch

from infrastructure.literature.summarization.methods_tools_validator import MethodsToolsValidator


class TestMethodsToolsValidator:
    """Test methods/tools output validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = MethodsToolsValidator()
        self.sample_pdf_text = """
        This paper presents a novel approach using PyTorch and TensorFlow for deep learning.
        We evaluated our method using ImageNet dataset and CIFAR-10 benchmarks.
        The experiments were conducted on local clusters with CUDA acceleration.
        """

    def test_no_repetition_bloat_fix(self):
        """Test that excessive 'Not specified' repetition is detected and fixed."""
        # Create bloated output with 10 repetitions
        bloated_output = "## Algorithms and Methodologies\nNot specified in paper\n" * 10
        bloated_output += "## Software Frameworks and Libraries\nNot specified in paper\n" * 5

        validated, warnings = self.validator.validate_output(
            bloated_output, self.sample_pdf_text, "test_key"
        )

        # Should consolidate all "Not specified" into single instances
        assert "Not specified in paper" in validated
        assert validated.count("Not specified in paper") <= 3  # Max allowed
        assert len(warnings) > 0
        assert any("repetition" in w.lower() for w in warnings)

    def test_hallucination_detection(self):
        """Test detection of hallucinated tools not in source."""
        # Output claiming to use tools not in PDF
        hallucinated_output = """## Software Frameworks and Libraries
* MATLAB (exact quote from paper) – "MATLAB"
* R (exact quote from paper) – "R programming language"
* SPSS (exact quote from paper) – "SPSS statistical software"
"""

        validated, warnings = self.validator.validate_output(
            hallucinated_output, self.sample_pdf_text, "test_key"
        )

        # Should remove hallucinated tools (MATLAB, R, SPSS not in PDF)
        assert "MATLAB" not in validated
        assert "SPSS" not in validated
        assert len(warnings) > 0

    def test_valid_tools_preserved(self):
        """Test that explicitly mentioned tools are preserved."""
        valid_output = """## Software Frameworks and Libraries
* PyTorch (exact quote from paper) – "PyTorch"
* TensorFlow (exact quote from paper) – "TensorFlow"
"""

        validated, warnings = self.validator.validate_output(
            valid_output, self.sample_pdf_text, "test_key"
        )

        # Should preserve valid tools that are actually in the PDF
        assert "PyTorch" in validated
        assert "TensorFlow" in validated
        assert len(warnings) == 0  # No warnings for valid content

    def test_output_size_limits(self):
        """Test detection of outputs exceeding size limits."""
        # Create output exceeding line and word limits
        large_output = "## Algorithms and Methodologies\n"
        large_output += "* Method description\n" * 250  # > 200 lines
        large_output += "\n" * 2000  # Add lots of words

        validated, warnings = self.validator.validate_output(
            large_output, self.sample_pdf_text, "test_key"
        )

        # Should detect size violations
        size_warnings = [w for w in warnings if "limit" in w.lower() or "exceeds" in w.lower()]
        assert len(size_warnings) > 0

    def test_formatting_fixes(self):
        """Test that formatting issues are corrected."""
        messy_output = """## Algorithms and Methodologies
*   Tool:PyTorch–description
*   Framework:TensorFlow–another description
"""

        validated, warnings = self.validator.validate_output(
            messy_output, self.sample_pdf_text, "test_key"
        )

        # Should fix spacing issues
        assert ": " in validated  # Colon followed by space
        assert " – " in validated  # Em dash with spaces

    def test_empty_sections_handled(self):
        """Test that empty sections are handled properly."""
        empty_output = """## Algorithms and Methodologies

## Software Frameworks and Libraries
Not specified in paper

## Datasets

## Evaluation Metrics
Not specified in paper
"""

        validated, warnings = self.validator.validate_output(
            empty_output, self.sample_pdf_text, "test_key"
        )

        # Should be valid output
        assert "## Algorithms and Methodologies" in validated
        assert "## Datasets" in validated
        assert len(warnings) == 0  # No issues with empty sections


class TestQuoteValidation:
    """Test quote validation enhancements."""

    def test_missing_quote_text_detection(self):
        """Test detection of quotes with missing text."""
        # This would be implemented when we enhance the quote validation
        # in _validate_quotes_against_source method
        pass

    def test_meta_commentary_removal(self):
        """Test removal of meta-commentary from quotes."""
        # This would be implemented when we enhance the quote validation
        # in _validate_quotes_against_source method
        pass

    def test_duplicate_quote_detection(self):
        """Test detection and removal of duplicate quotes."""
        # This would be implemented when we enhance the quote validation
        # in _validate_quotes_against_source method
        pass


class TestSummaryInformationDensity:
    """Test summary information density validation."""

    def test_vague_content_detection(self):
        """Test detection of vague, repetitive content."""
        # This would be implemented when we add information density checks
        # to the validator
        pass

    def test_specific_content_recognition(self):
        """Test recognition of specific, detailed content."""
        # This would be implemented when we add information density checks
        pass


class TestOutputAnomalyDetection:
    """Test comprehensive output anomaly detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_pdf_text = """
        This paper presents a novel approach using PyTorch and TensorFlow for deep learning.
        We evaluated our method using ImageNet dataset and CIFAR-10 benchmarks.
        The experiments were conducted on local clusters with CUDA acceleration.
        """

    def test_regression_bloat_detection(self):
        """Test detection of known bloat patterns from regression cases."""
        # Test case based on lapkovskis2025benchmarking_methods_tools.md
        # which had 932 repetitions of "Not specified in paper"

        # Simulate the bloated output pattern
        bloated = ""
        for i in range(100):  # Simulate 100 repetitions
            bloated += "Not specified in paper\n"

        validator = MethodsToolsValidator()
        validated, warnings = validator.validate_output(
            bloated, self.sample_pdf_text, "regression_test"
        )

        # Should detect and fix bloat
        assert validated.count("Not specified in paper") <= 3
        assert len(warnings) > 0
        assert any("repetition" in w.lower() for w in warnings)

    def test_regression_hallucination_detection(self):
        """Test detection of hallucinated tools from fujii2025realworld."""
        # Test case based on fujii2025realworld_methods_tools.md
        # which listed tools/datasets not actually mentioned in paper
        
        # Use a PDF text that doesn't contain the hallucinated items
        pdf_text_without_hallucinations = """
        This paper presents a novel approach using PyTorch for deep learning.
        We evaluated our method on standard benchmarks.
        The experiments were conducted on local clusters.
        """

        hallucinated = """## Datasets
* ImageNet (exact quote from paper) – "ImageNet"
* CIFAR-10 (exact quote from paper) – "CIFAR-10"

## Software Tools and Platforms
* Google Colab (exact quote from paper) – "Google Colab"
* AWS (exact quote from paper) – "AWS"
"""

        validator = MethodsToolsValidator()
        validated, warnings = validator.validate_output(
            hallucinated, pdf_text_without_hallucinations, "hallucination_test"
        )

        # Should remove hallucinated items not in PDF
        assert "ImageNet" not in validated
        assert "CIFAR-10" not in validated
        assert "Google Colab" not in validated
        assert "AWS" not in validated
        assert len(warnings) > 0

    @pytest.mark.parametrize("output_size,should_warn", [
        (50, False),      # Normal size (50 lines)
        (250, False),     # Normal size (250 words, 1 line)
        (2000, True),     # Exceeds word limit (2000 words > 1500)
        (60000, True),    # Exceeds word and character limit
    ])
    def test_output_size_limits_parametrized(self, output_size, should_warn):
        """Test output size limits with various sizes."""
        # Create output of specified size
        if output_size < 200:  # Line count
            output = "## Test\n" * output_size
        else:  # Word count
            output = "word " * output_size

        validator = MethodsToolsValidator()
        validated, warnings = validator.validate_output(
            output, self.sample_pdf_text, f"size_test_{output_size}"
        )

        size_warnings = [w for w in warnings if "limit" in w.lower() or "exceeds" in w.lower()]
        if should_warn:
            assert len(size_warnings) > 0, f"Expected size warning for {output_size} but got none"
        else:
            assert len(size_warnings) == 0, f"Unexpected size warning for {output_size}: {size_warnings}"
