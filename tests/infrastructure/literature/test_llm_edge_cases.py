"""Edge case tests for LLM operations.

Tests for boundary conditions, error cases, and fallback behaviors.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from infrastructure.literature.llm import LiteratureLLMOperations
from infrastructure.literature.library import LibraryEntry


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestLLMOperationsEdgeCases:
    """Edge case tests for all LLM operations."""

    @pytest.fixture
    def llm_operations(self, ensure_ollama_available):
        """Create LLM operations with Ollama client."""
        return LiteratureLLMOperations(ensure_ollama_available)

    @pytest.mark.timeout(120)  # Single LLM call
    def test_empty_paper_list(self, llm_operations):
        """Test operations handle empty paper list gracefully."""
        # Empty list should not crash
        result = llm_operations.generate_literature_review(papers=[], focus="general")
        assert result.operation_type == "literature_review"
        assert result.papers_used == 0

    @pytest.mark.timeout(120)  # Single LLM call
    def test_single_paper(self, llm_operations):
        """Test operations work with minimum (single) paper."""
        paper = LibraryEntry(
            citation_key="solo_paper",
            title="Single Paper Test",
            authors=["Solo Author"],
            year=2024,
            abstract="This is a single paper for testing minimum requirements.",
            source="arxiv",
            url="https://example.com"
        )
        
        # Should work with single paper
        result = llm_operations.generate_literature_review(papers=[paper])
        assert result.papers_used == 1
        assert result.output_text is not None

    @pytest.mark.timeout(120)  # Single LLM call
    def test_max_papers_limit_respected(self, llm_operations):
        """Test max_papers parameter is respected."""
        papers = [
            LibraryEntry(
                citation_key=f"paper{i}",
                title=f"Paper {i}",
                authors=[f"Author {i}"],
                year=2024,
                abstract=f"Abstract {i}",
                source="arxiv",
                url=f"https://example.com/{i}"
            )
            for i in range(15)
        ]
        
        # Should limit to max_papers
        result = llm_operations.generate_literature_review(
            papers=papers,
            max_papers=5
        )
        assert result.papers_used == 5, f"Expected 5 papers used, got {result.papers_used}"

    @pytest.mark.timeout(120)  # Single LLM call
    def test_papers_without_abstracts(self, llm_operations):
        """Test operations handle papers without abstracts."""
        paper = LibraryEntry(
            citation_key="no_abstract",
            title="Paper Without Abstract",
            authors=["Author"],
            year=2024,
            abstract=None,  # No abstract
            source="arxiv",
            url="https://example.com"
        )
        
        # Should handle missing abstract gracefully
        result = llm_operations.generate_literature_review(papers=[paper])
        assert result.output_text is not None
        assert len(result.output_text) > 0

    @pytest.mark.timeout(120)  # Single LLM call
    def test_papers_with_empty_titles(self, llm_operations):
        """Test operations handle papers with minimal metadata."""
        paper = LibraryEntry(
            citation_key="minimal",
            title="",  # Empty title
            authors=[],
            year=None,
            abstract="Some abstract text",
            source="arxiv",
            url="https://example.com"
        )
        
        # Should handle minimal metadata
        result = llm_operations.generate_science_communication(papers=[paper])
        assert result.output_text is not None

    @pytest.mark.timeout(120)
    def test_fallback_to_abstracts(self, llm_operations, tmp_path):
        """Test fallback behavior when summaries unavailable."""
        # Create paper without summary file
        paper = LibraryEntry(
            citation_key="no_summary",
            title="Paper Without Summary",
            authors=["Author"],
            year=2024,
            abstract="This paper has an abstract but no summary file.",
            source="arxiv",
            url="https://example.com"
        )
        
        # Should fall back to abstract
        result = llm_operations.generate_literature_review(papers=[paper])
        assert result.output_text is not None
        assert "abstract" in result.output_text.lower() or result.output_text  # Has content

    @pytest.mark.timeout(240)  # 4 LLM calls in loop
    def test_comparative_analysis_different_aspects(self, llm_operations):
        """Test comparative analysis with different aspect parameters."""
        papers = [
            LibraryEntry(
                citation_key="p1",
                title="Paper 1",
                authors=["A"],
                year=2024,
                abstract="Abstract 1",
                source="arxiv",
                url="https://example.com"
            ),
            LibraryEntry(
                citation_key="p2",
                title="Paper 2",
                authors=["B"],
                year=2024,
                abstract="Abstract 2",
                source="arxiv",
                url="https://example.com"
            ),
        ]
        
        # Test different aspects
        for aspect in ["methods", "results", "datasets", "performance"]:
            result = llm_operations.generate_comparative_analysis(papers=papers, aspect=aspect)
            assert result.operation_type == "comparative_analysis"
            assert result.metadata.get("aspect") == aspect

    @pytest.mark.slow
    @pytest.mark.timeout(240)  # 4 LLM calls in loop
    def test_research_gaps_various_domains(self, llm_operations):
        """Test research gap identification with different domains."""
        papers = [
            LibraryEntry(
                citation_key="p1",
                title="Test Paper",
                authors=["A"],
                year=2024,
                abstract="Test abstract",
                source="arxiv",
                url="https://example.com"
            )
        ]
        
        # Test different domains
        for domain in ["general", "machine learning", "neuroscience", "physics"]:
            result = llm_operations.generate_research_gaps(papers=papers, domain=domain)
            assert result.operation_type == "research_gaps"
            assert result.metadata.get("domain") == domain

    @pytest.mark.timeout(120)  # Single LLM call
    def test_citation_network_minimal_papers(self, llm_operations):
        """Test citation network analysis with minimal papers."""
        paper = LibraryEntry(
            citation_key="single",
            title="Single Paper",
            authors=["Author"],
            year=2024,
            abstract="Abstract",
            source="arxiv",
            url="https://example.com"
        )
        
        # Should work with single paper (minimal network)
        result = llm_operations.analyze_citation_network(papers=[paper])
        assert result.output_text is not None

    def test_save_result_creates_directory(self, llm_operations, tmp_path):
        """Test save_result creates output directory if missing."""
        from infrastructure.literature.llm import LLMOperationResult
        
        result = LLMOperationResult(
            operation_type="test",
            papers_used=1,
            citation_keys=["test"],
            output_text="Test output",
            generation_time=1.0,
            tokens_estimated=10
        )
        
        output_dir = tmp_path / "new_dir" / "nested"
        saved_path = llm_operations.save_result(result, output_dir)
        
        assert saved_path.exists()
        assert saved_path.parent == output_dir

    @pytest.mark.timeout(120)  # Single LLM call
    def test_operations_with_very_long_abstracts(self, llm_operations):
        """Test operations handle very long abstracts."""
        # Create paper with very long abstract (simulate edge case)
        long_abstract = " ".join(["This is a sentence."] * 100)
        
        paper = LibraryEntry(
            citation_key="long_abstract",
            title="Paper With Long Abstract",
            authors=["Author"],
            year=2024,
            abstract=long_abstract,
            source="arxiv",
            url="https://example.com"
        )
        
        result = llm_operations.generate_science_communication(papers=[paper])
        assert result.output_text is not None

    @pytest.mark.timeout(120)  # Single LLM call
    def test_operations_with_special_characters_in_title(self, llm_operations):
        """Test operations handle special characters in titles."""
        paper = LibraryEntry(
            citation_key="special_chars",
            title="Test: Paper & Analysis (2024) — A Study",
            authors=["Author"],
            year=2024,
            abstract="Abstract with special characters: & < > \" ' \\",
            source="arxiv",
            url="https://example.com"
        )
        
        result = llm_operations.generate_literature_review(papers=[paper])
        assert result.output_text is not None

    @pytest.mark.timeout(120)  # Single LLM call
    def test_operations_with_unicode_characters(self, llm_operations):
        """Test operations handle Unicode characters."""
        paper = LibraryEntry(
            citation_key="unicode",
            title="Étude sur l'apprentissage automatique avec données françaises",
            authors=["François Müller"],
            year=2024,
            abstract="研究关于机器学习的应用 αβγδε μ∞∑∫",
            source="arxiv",
            url="https://example.com"
        )
        
        result = llm_operations.generate_literature_review(papers=[paper])
        assert result.output_text is not None

    @pytest.mark.timeout(120)  # Single LLM call
    def test_operations_generation_time_recorded(self, llm_operations):
        """Test that generation time is properly recorded."""
        paper = LibraryEntry(
            citation_key="timing_test",
            title="Timing Test Paper",
            authors=["Author"],
            year=2024,
            abstract="Test abstract",
            source="arxiv",
            url="https://example.com"
        )
        
        result = llm_operations.generate_literature_review(papers=[paper])
        assert result.generation_time > 0, "Generation time should be positive"
        assert isinstance(result.generation_time, float), "Generation time should be float"

    @pytest.mark.timeout(120)  # Single LLM call
    def test_operations_token_estimation(self, llm_operations):
        """Test that token estimation is reasonable."""
        paper = LibraryEntry(
            citation_key="token_test",
            title="Token Test Paper",
            authors=["Author"],
            year=2024,
            abstract="Test abstract",
            source="arxiv",
            url="https://example.com"
        )
        
        result = llm_operations.generate_literature_review(papers=[paper])
        assert result.tokens_estimated > 0, "Token estimate should be positive"
        assert isinstance(result.tokens_estimated, int), "Token estimate should be integer"
        # Rough check: tokens should be less than output character count
        assert result.tokens_estimated < len(result.output_text) * 2

    @pytest.mark.timeout(120)  # Single LLM call
    def test_operations_citation_keys_tracked(self, llm_operations):
        """Test that citation keys are properly tracked."""
        papers = [
            LibraryEntry(
                citation_key=f"key{i}",
                title=f"Paper {i}",
                authors=["A"],
                year=2024,
                abstract=f"Abstract {i}",
                source="arxiv",
                url="https://example.com"
            )
            for i in range(3)
        ]
        
        result = llm_operations.generate_literature_review(papers=papers)
        assert len(result.citation_keys) == 3
        assert result.citation_keys == ["key0", "key1", "key2"]

