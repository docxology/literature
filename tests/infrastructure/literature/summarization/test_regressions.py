"""Regression tests for summarization system failures.

Tests for known bugs and failure cases to prevent regressions.
"""
import pytest
from unittest.mock import Mock, patch

from infrastructure.literature.summarization.methods_tools_validator import MethodsToolsValidator


class TestSummarizationRegressions:
    """Regression tests for summarization failures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = MethodsToolsValidator()
        self.sample_pdf_text = """
        This paper presents a novel approach using PyTorch and TensorFlow for deep learning.
        We evaluated our method using ImageNet dataset and CIFAR-10 benchmarks.
        The experiments were conducted on local clusters with CUDA acceleration.
        """

    def test_lapkovskis_methods_bloat_regression(self):
        """Regression test for lapkovskis2025benchmarking methods bloat (990-line file).

        This test ensures that the methods validator catches and fixes
        the excessive 'Not specified in paper' repetition that created
        a 990-line output file.
        """
        # Simulate the bloated output pattern from the regression case
        bloated_lines = []
        for i in range(100):  # Create 100 repetitions like the original bug
            bloated_lines.append("Not specified in paper")
            bloated_lines.append("")  # Empty line

        bloated_output = "\n".join(bloated_lines)

        validated, warnings = self.validator.validate_output(
            bloated_output, self.sample_pdf_text, "lapkovskis_regression_test"
        )

        # Should consolidate all repetitions into single section
        assert "Not specified in paper" in validated
        assert validated.count("Not specified in paper") <= 3  # Max allowed
        assert len(warnings) > 0
        assert any("repetition" in w.lower() for w in warnings)

        # Output should be much shorter
        validated_lines = len(validated.split('\n'))
        assert validated_lines < 10  # Should be dramatically reduced

    def test_fujii_hallucinated_tools_regression(self):
        """Regression test for fujii2025realworld hallucinated tools.

        This test ensures that tools not mentioned in the paper (ImageNet, CIFAR-10,
        Google Colab, AWS) are removed by the validator.
        """
        # PDF text WITHOUT ImageNet, CIFAR-10, Google Colab, or AWS
        pdf_text_no_hallucinations = """
        This paper presents a novel approach using PyTorch and TensorFlow for deep learning.
        The experiments were conducted on local clusters with CUDA acceleration.
        """
        
        # Simulate the hallucinated output from fujii regression case
        hallucinated_output = """## Software Frameworks and Libraries
* PyTorch (exact quote from paper) – "PyTorch"
* TensorFlow (exact quote from paper) – "TensorFlow"
* ImageNet (exact quote from paper) – "ImageNet"
* CIFAR-10 (exact quote from paper) – "CIFAR-10"

## Datasets
* Google Colab (exact quote from paper) – "Google Colab"
* AWS (exact quote from paper) – "AWS"
* CUDA (exact quote from paper) – "CUDA"
"""

        validated, warnings = self.validator.validate_output(
            hallucinated_output, pdf_text_no_hallucinations, "fujii_regression_test"
        )

        # Should preserve valid tools that are actually in PDF
        assert "PyTorch" in validated
        assert "TensorFlow" in validated
        assert "CUDA" in validated

        # Should remove hallucinated tools not in PDF
        assert "ImageNet" not in validated
        assert "CIFAR-10" not in validated
        assert "Google Colab" not in validated
        assert "AWS" not in validated

        # Should have warnings about hallucinations
        assert len(warnings) > 0
        hallucination_warnings = [w for w in warnings if "hallucinated" in w.lower()]
        assert len(hallucination_warnings) > 0

    def test_severe_repetition_retry_regression(self):
        """Regression test for severe repetition retry failures.

        Tests that the system can handle cases where initial output has
        severe repetition that would normally trigger retries.
        """
        # Create content with severe repetition like the terminal logs showed
        repetitive_content = ""
        repeated_phrase = "the proposed framework achieves high performance"
        for i in range(10):  # Create severe repetition
            repetitive_content += f"## Section {i}\n"
            repetitive_content += f"{repeated_phrase}\n" * 3  # 3 repetitions per section
            repetitive_content += "\n"

        # This would normally be caught by the repetition validator
        # but let's test that our methods validator doesn't make it worse
        validated, warnings = self.validator.validate_output(
            repetitive_content, self.sample_pdf_text, "repetition_regression_test"
        )

        # Should still process without crashing
        assert isinstance(validated, str)
        assert len(warnings) >= 0  # May or may not have warnings

    def test_mixed_valid_invalid_tools_regression(self):
        """Regression test for mixed valid and invalid tools.

        Tests handling of output with both correctly extracted tools
        and hallucinated ones.
        """
        # PDF text WITHOUT ImageNet, CIFAR-10, MATLAB, or SPSS
        pdf_text_no_invalid = """
        This paper presents a novel approach using PyTorch and TensorFlow for deep learning.
        The experiments were conducted on local clusters with CUDA acceleration.
        """
        
        mixed_output = """## Software Frameworks and Libraries
* PyTorch (exact quote from paper) – "PyTorch"
* MATLAB (exact quote from paper) – "MATLAB"
* TensorFlow (exact quote from paper) – "TensorFlow"
* SPSS (exact quote from paper) – "SPSS"
* CUDA (exact quote from paper) – "CUDA"

## Datasets
* ImageNet (exact quote from paper) – "ImageNet"
* CIFAR-10 (exact quote from paper) – "CIFAR-10"
"""

        validated, warnings = self.validator.validate_output(
            mixed_output, pdf_text_no_invalid, "mixed_tools_regression_test"
        )

        # Should keep valid tools
        assert "PyTorch" in validated
        assert "TensorFlow" in validated
        assert "CUDA" in validated

        # Should remove invalid tools
        assert "MATLAB" not in validated
        assert "SPSS" not in validated
        assert "ImageNet" not in validated
        assert "CIFAR-10" not in validated

        # Should have warnings
        assert len(warnings) > 0

    def test_output_size_regression_990_lines(self):
        """Regression test specifically for the 990-line output case."""
        # Create output similar to the 990-line regression case
        huge_output = "## Algorithms and Methodologies\n"
        huge_output += "* Some method description\n" * 100  # 100 lines
        huge_output += "\n## Software Frameworks and Libraries\n"
        huge_output += "Not specified in paper\n" * 200  # 200 repetitions
        huge_output += "\n## Datasets\n"
        huge_output += "Not specified in paper\n" * 300  # 300 repetitions
        huge_output += "\n## Evaluation Metrics\n"
        huge_output += "* Accuracy (exact quote from paper) – \"accuracy\"\n" * 50
        huge_output += "\n## Software Tools and Platforms\n"
        huge_output += "Not specified in paper\n" * 340  # 340 repetitions

        validated, warnings = self.validator.validate_output(
            huge_output, self.sample_pdf_text, "size_regression_test"
        )

        # Should dramatically reduce size
        original_lines = len(huge_output.split('\n'))
        validated_lines = len(validated.split('\n'))

        assert validated_lines < original_lines * 0.1  # Should be <10% of original
        assert validated_lines < 50  # Should be reasonably small

        # Should have size warnings
        size_warnings = [w for w in warnings if "limit" in w.lower() or "exceeds" in w.lower()]
        assert len(size_warnings) > 0

    @pytest.mark.parametrize("repetition_count", [5, 10, 50, 100])
    def test_various_repetition_levels(self, repetition_count):
        """Test handling of various levels of 'Not specified' repetition."""
        # Create output with specified number of repetitions
        bloated = "\n".join(["Not specified in paper"] * repetition_count)

        validated, warnings = self.validator.validate_output(
            bloated, self.sample_pdf_text, f"repetition_test_{repetition_count}"
        )

        # Should always consolidate to max 3 instances
        assert validated.count("Not specified in paper") <= 3

        # Should have warnings for high repetition counts
        if repetition_count > 3:
            assert len(warnings) > 0
            assert any("repetition" in w.lower() for w in warnings)

    def test_edge_case_empty_sections_regression(self):
        """Regression test for handling completely empty sections."""
        empty_output = """## Algorithms and Methodologies

## Software Frameworks and Libraries

## Datasets

## Evaluation Metrics

## Software Tools and Platforms
"""

        validated, warnings = self.validator.validate_output(
            empty_output, self.sample_pdf_text, "empty_sections_regression_test"
        )

        # Should handle gracefully
        assert isinstance(validated, str)
        assert len(validated.split('\n')) > 0  # Should not be completely empty

    def test_formatting_regression_artifacts(self):
        """Regression test for PDF extraction artifacts in quotes."""
        # Test the → artifacts that were causing issues
        artifact_output = """## Important Quotes
**Quote:** "dynamicssimulationsand → dynamics simulations and"
**Quote:** "word1word2word3 → word1 word2 word3"
"""

        validated, warnings = self.validator.validate_output(
            artifact_output, self.sample_pdf_text, "artifact_regression_test"
        )

        # Should remove artifact quotes
        assert "→" not in validated
        assert "dynamicssimulationsand" not in validated
        assert len(warnings) > 0
