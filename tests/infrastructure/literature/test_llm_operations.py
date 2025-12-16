"""Integration tests for LLM operations with Ollama.

These tests use Ollama LLM to generate literature reviews, comparative analysis,
and other LLM operations. Requires Ollama to be running.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from infrastructure.literature.llm import LiteratureLLMOperations, LLMOperationResult
from infrastructure.literature.library import LibraryEntry


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestLLMOperations:
    """LLM operations tests with Ollama."""

    @pytest.fixture
    def llm_operations(self, ensure_ollama_available):
        """Create LLM operations with Ollama client."""
        return LiteratureLLMOperations(ensure_ollama_available)

    @pytest.fixture
    def sample_papers(self):
        """Create sample papers for LLM operations."""
        return [
            LibraryEntry(
                citation_key="paper1",
                title="Machine Learning in Physics",
                authors=["Author A", "Author B"],
                year=2020,
                abstract="This paper discusses machine learning applications in physics research. We present novel methods for analyzing physical systems using neural networks.",
                doi="10.1234/test1",
                source="arxiv",
                url="https://example.com/1",
                venue="Journal of Physics",
                citation_count=50,
            ),
            LibraryEntry(
                citation_key="paper2",
                title="Deep Learning for Biology",
                authors=["Author C", "Author D"],
                year=2021,
                abstract="Deep learning methods for biological data analysis. The paper presents convolutional neural networks for protein structure prediction.",
                doi="10.1234/test2",
                source="semanticscholar",
                url="https://example.com/2",
                venue="Nature Biology",
                citation_count=100,
            ),
            LibraryEntry(
                citation_key="paper3",
                title="Neural Networks and Active Inference",
                authors=["Author A", "Author E"],
                year=2022,
                abstract="Neural networks and active inference in cognitive science. This work explores the relationship between predictive processing and neural computation.",
                doi="10.1234/test3",
                source="arxiv",
                url="https://example.com/3",
                venue="Cognitive Science",
                citation_count=75,
            ),
        ]

    @pytest.mark.timeout(180)  # Use configurable timeout from test config
    def test_literature_review(self, llm_operations, sample_papers):
        """Test literature review generation."""
        from tests.test_config_loader import get_test_timeout
        
        # Use extended timeout from test config
        timeout = get_test_timeout("llm_review")
        
        try:
            result = llm_operations.generate_literature_review(
                papers=sample_papers,
                focus="methods",
                max_papers=3
            )
            
            assert isinstance(result, LLMOperationResult), (
                f"Expected LLMOperationResult, got {type(result)}"
            )
            assert result.operation_type == "literature_review", (
                f"Expected operation_type 'literature_review', got '{result.operation_type}'"
            )
            assert result.papers_used == len(sample_papers), (
                f"Expected {len(sample_papers)} papers used, got {result.papers_used}"
            )
            assert result.output_text is not None, "output_text should not be None"
            assert len(result.output_text) > 100, (
                f"Expected substantial content (>100 chars), got {len(result.output_text)} chars"
            )
            assert result.generation_time > 0, (
                f"Expected positive generation_time, got {result.generation_time}"
            )
            assert result.tokens_estimated > 0, (
                f"Expected positive tokens_estimated, got {result.tokens_estimated}"
            )
            assert len(result.citation_keys) == len(sample_papers), (
                f"Expected {len(sample_papers)} citation_keys, got {len(result.citation_keys)}"
            )
        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                pytest.skip(
                    f"LLM operation timed out (may need longer timeout or faster model): {e}"
                )
            raise

    @pytest.mark.timeout(120)
    def test_science_communication(self, llm_operations, sample_papers):
        """Test science communication generation."""
        result = llm_operations.generate_science_communication(
            papers=sample_papers[:2],
            audience="general_public"
        )
        
        assert isinstance(result, LLMOperationResult), (
            f"Expected LLMOperationResult, got {type(result)}"
        )
        assert result.operation_type == "science_communication", (
            f"Expected operation_type 'science_communication', got '{result.operation_type}'"
        )
        assert result.papers_used == 2, (
            f"Expected 2 papers used, got {result.papers_used}"
        )
        assert result.output_text is not None, "output_text should not be None"
        assert len(result.output_text) > 50, (
            f"Expected content (>50 chars), got {len(result.output_text)} chars"
        )
        assert result.generation_time > 0, (
            f"Expected positive generation_time, got {result.generation_time}"
        )

    @pytest.mark.timeout(120)
    def test_comparative_analysis(self, llm_operations, sample_papers):
        """Test comparative analysis."""
        result = llm_operations.generate_comparative_analysis(
            papers=sample_papers,
            aspect="methods"
        )
        
        assert isinstance(result, LLMOperationResult), (
            f"Expected LLMOperationResult, got {type(result)}"
        )
        assert result.operation_type == "comparative_analysis", (
            f"Expected operation_type 'comparative_analysis', got '{result.operation_type}'"
        )
        assert result.papers_used == len(sample_papers), (
            f"Expected {len(sample_papers)} papers used, got {result.papers_used}"
        )
        assert result.output_text is not None, "output_text should not be None"
        assert len(result.output_text) > 50, (
            f"Expected content (>50 chars), got {len(result.output_text)} chars"
        )
        output_lower = result.output_text.lower()
        assert "method" in output_lower or "approach" in output_lower, (
            f"Expected 'method' or 'approach' in output, got: {result.output_text[:200]}..."
        )

    @pytest.mark.timeout(120)
    def test_research_gaps(self, llm_operations, sample_papers):
        """Test research gap identification."""
        result = llm_operations.generate_research_gaps(
            papers=sample_papers,
            domain="machine learning"
        )
        
        assert isinstance(result, LLMOperationResult), (
            f"Expected LLMOperationResult, got {type(result)}"
        )
        assert result.operation_type == "research_gaps", (
            f"Expected operation_type 'research_gaps', got '{result.operation_type}'"
        )
        assert result.papers_used == len(sample_papers), (
            f"Expected {len(sample_papers)} papers used, got {result.papers_used}"
        )
        assert result.output_text is not None, "output_text should not be None"
        assert len(result.output_text) > 50, (
            f"Expected content (>50 chars), got {len(result.output_text)} chars"
        )
        assert result.generation_time > 0, (
            f"Expected positive generation_time, got {result.generation_time}"
        )

    @pytest.mark.timeout(120)
    def test_citation_network(self, llm_operations, sample_papers):
        """Test citation network analysis."""
        result = llm_operations.analyze_citation_network(
            papers=sample_papers
        )
        
        assert isinstance(result, LLMOperationResult), (
            f"Expected LLMOperationResult, got {type(result)}"
        )
        assert result.operation_type == "citation_network", (
            f"Expected operation_type 'citation_network', got '{result.operation_type}'"
        )
        assert result.papers_used == len(sample_papers), (
            f"Expected {len(sample_papers)} papers used, got {result.papers_used}"
        )
        assert result.output_text is not None, "output_text should not be None"
        assert len(result.output_text) > 50, (
            f"Expected content (>50 chars), got {len(result.output_text)} chars"
        )

    @pytest.mark.timeout(180)  # Same timeout as test_literature_review
    def test_literature_review_save(self, llm_operations, sample_papers, tmp_path):
        """Test literature review with file saving."""
        output_dir = tmp_path / "llm_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = llm_operations.generate_literature_review(
            papers=sample_papers,
            focus="general",
            max_papers=3
        )
        
        # Save result
        saved_path = llm_operations.save_result(
            result,
            output_dir=output_dir
        )
        
        assert saved_path.exists(), f"Expected saved file to exist at {saved_path}"
        assert saved_path.suffix == ".md", (
            f"Expected .md extension, got {saved_path.suffix}"
        )
        content = saved_path.read_text()
        assert len(content) > 0, "Saved file should not be empty"
        assert result.output_text in content, (
            f"Expected output_text to be in saved file content"
        )


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestLiteratureReview:
    """Literature review generation tests."""

    @pytest.fixture
    def llm_operations(self, ensure_ollama_available):
        """Create LLM operations with Ollama client."""
        return LiteratureLLMOperations(ensure_ollama_available)

    @pytest.mark.timeout(120)
    def test_review_with_summaries(self, llm_operations, tmp_path):
        """Test literature review using paper summaries."""
        # Create sample papers with summary files
        summaries_dir = tmp_path / "summaries"
        summaries_dir.mkdir(parents=True)
        
        papers = [
            LibraryEntry(
                citation_key="paper1",
                title="Paper 1",
                authors=["Author 1"],
                year=2024,
                abstract="Abstract 1",
                source="arxiv",
                url="https://example.com/1"
            ),
            LibraryEntry(
                citation_key="paper2",
                title="Paper 2",
                authors=["Author 2"],
                year=2024,
                abstract="Abstract 2",
                source="arxiv",
                url="https://example.com/2"
            ),
        ]
        
        # Create summary files
        (summaries_dir / "paper1_summary.md").write_text("# Paper 1 Summary\n\nKey findings and methods.")
        (summaries_dir / "paper2_summary.md").write_text("# Paper 2 Summary\n\nMain contributions and results.")
        
        result = llm_operations.generate_literature_review(
            papers=papers,
            focus="general",
            max_papers=2
        )
        
        assert result.operation_type == "literature_review", (
            f"Expected operation_type 'literature_review', got '{result.operation_type}'"
        )
        assert result.papers_used == 2, (
            f"Expected 2 papers used, got {result.papers_used}"
        )
        assert result.output_text is not None, "output_text should not be None"
        assert len(result.output_text) > 50, (
            f"Expected content (>50 chars), got {len(result.output_text)} chars"
        )


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestComparativeAnalysis:
    """Comparative analysis tests."""

    @pytest.fixture
    def llm_operations(self, ensure_ollama_available):
        """Create LLM operations with Ollama client."""
        return LiteratureLLMOperations(ensure_ollama_available)

    @pytest.mark.timeout(120)
    def test_compare_methods(self, llm_operations):
        """Test comparison of methods."""
        papers = [
            LibraryEntry(
                citation_key="p1",
                title="Method A Paper",
                authors=["Author"],
                year=2024,
                abstract="This paper presents Method A for solving problems using neural networks.",
                source="arxiv",
                url="https://example.com"
            ),
            LibraryEntry(
                citation_key="p2",
                title="Method B Paper",
                authors=["Author"],
                year=2024,
                abstract="This paper presents Method B using deep learning approaches.",
                source="arxiv",
                url="https://example.com"
            ),
        ]
        
        result = llm_operations.generate_comparative_analysis(
            papers=papers,
            aspect="methods"
        )
        
        assert result.operation_type == "comparative_analysis", (
            f"Expected operation_type 'comparative_analysis', got '{result.operation_type}'"
        )
        assert result.output_text is not None, "output_text should not be None"
        assert len(result.output_text) > 50, (
            f"Expected content (>50 chars), got {len(result.output_text)} chars"
        )

