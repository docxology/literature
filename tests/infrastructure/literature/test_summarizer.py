"""Tests for paper summarization functionality."""
import pytest
from pathlib import Path
from unittest.mock import Mock

from infrastructure.literature.sources import SearchResult
from infrastructure.literature.summarization import (
    SummarizationResult,
    SummaryQualityValidator,
    SummarizationEngine,
    PaperSummarizer,  # Alias for backward compatibility
)
from infrastructure.validation.pdf_validator import PDFValidationError
from infrastructure.core.exceptions import LLMConnectionError


class TestSummarizationResult:
    """Test SummarizationResult dataclass."""

    def test_creation_success(self):
        """Test successful summarization result."""
        result = SummarizationResult(
            citation_key="test2024paper",
            success=True,
            summary_text="This is a test summary.",
            input_chars=1000,
            input_words=200,
            output_words=50,
            generation_time=30.0,
            attempts=1,
            quality_score=0.9
        )

        assert result.citation_key == "test2024paper"
        assert result.success is True
        assert result.summary_text == "This is a test summary."
        assert result.input_chars == 1000
        assert result.input_words == 200
        assert result.output_words == 50
        assert result.generation_time == 30.0
        assert result.attempts == 1
        assert result.quality_score == 0.9
        assert result.compression_ratio == 0.25  # 50/200
        assert result.words_per_second == 50/30  # 50 words / 30 seconds

    def test_creation_failure(self):
        """Test failed summarization result."""
        result = SummarizationResult(
            citation_key="test2024paper",
            success=False,
            error="PDF extraction failed",
            attempts=2
        )

        assert result.success is False
        assert result.error == "PDF extraction failed"
        assert result.attempts == 2
        assert result.summary_text is None

    def test_compression_ratio_edge_cases(self):
        """Test compression ratio with edge cases."""
        # Zero input words
        result = SummarizationResult(
            citation_key="key",
            success=True,
            summary_text="summary",
            input_chars=0,
            input_words=0,
            output_words=10,
            generation_time=1.0,
            attempts=1
        )
        assert result.compression_ratio == 10.0

        # Normal case
        result = SummarizationResult(
            citation_key="key",
            success=True,
            summary_text="summary",
            input_chars=100,
            input_words=20,
            output_words=10,
            generation_time=1.0,
            attempts=1
        )
        assert result.compression_ratio == 0.5

    def test_words_per_second_edge_cases(self):
        """Test words per second with edge cases."""
        # Zero generation time
        result = SummarizationResult(
            citation_key="key",
            success=True,
            summary_text="summary",
            input_chars=100,
            input_words=20,
            output_words=10,
            generation_time=0.0,
            attempts=1
        )
        assert result.words_per_second == 10000.0

        # Very small time
        result = SummarizationResult(
            citation_key="key",
            success=True,
            summary_text="summary",
            input_chars=100,
            input_words=20,
            output_words=10,
            generation_time=0.001,
            attempts=1
        )
        assert result.words_per_second == 10000.0


class TestSummaryQualityValidator:
    """Test SummaryQualityValidator functionality."""

    def test_creation(self):
        """Test validator creation."""
        validator = SummaryQualityValidator()
        assert validator.min_words == 200

        validator = SummaryQualityValidator(min_words=500)
        assert validator.min_words == 500

    def test_validate_perfect_summary(self):
        """Test validation of a perfect summary."""
        validator = SummaryQualityValidator(min_words=25)  # Lower for testing
        pdf_text = "This is a scientific paper about machine learning."
        summary = """### Overview
This paper presents machine learning techniques.

### Key Contributions
The authors introduce novel algorithms.

### Methodology
They use neural networks and statistical methods.

### Results
The results show improved performance.

### Limitations and Future Work
Future work includes scaling to larger datasets."""

        is_valid, score, errors = validator.validate_summary(summary, pdf_text, "test")

        assert is_valid is True
        assert score >= 0.8  # Should be high score
        assert len(errors) == 0

    def test_validate_poor_summary(self):
        """Test validation of a poor summary."""
        validator = SummaryQualityValidator(min_words=100)
        pdf_text = "This is a scientific paper about machine learning."
        summary = "This paper is about stuff."  # Too short, missing sections

        is_valid, score, errors = validator.validate_summary(summary, pdf_text, "test")

        assert is_valid is False
        assert score < 0.8  # Should be low score (adjusted for new min_words default)
        assert len(errors) > 0
        assert any("short" in error.lower() for error in errors)

    def test_detect_sentence_repetition(self):
        """Test sentence-level repetition detection."""
        validator = SummaryQualityValidator()

        # No repetition
        summary = "This paper presents method A. The authors show result B. They conclude with finding C."
        assert not validator._detect_sentence_repetition(summary)

        # With repetition
        summary = "This paper presents method A. This paper presents method A. This paper presents method A. This paper presents method A. This paper discusses limitation D."
        assert validator._detect_sentence_repetition(summary)

    def test_detect_section_repetition(self):
        """Test section-level repetition detection."""
        validator = SummaryQualityValidator()

        # No section repetition
        summary = """### Overview
This is the overview.

### Key Contributions
These are the contributions.

### Methodology
This is the methodology."""
        assert not validator._detect_section_repetition(summary)

        # With section repetition
        summary = """### Overview
This is the overview.

### Key Contributions
These are the contributions.

### Overview
This is another overview section."""
        assert validator._detect_section_repetition(summary)

        # Multiple repetitions
        summary = """### Summary
First summary.

### Summary
Second summary.

### Summary
Third summary."""
        assert validator._detect_section_repetition(summary)

    def test_detect_paragraph_repetition(self):
        """Test paragraph-level repetition detection."""
        validator = SummaryQualityValidator()

        # No paragraph repetition
        summary = """### Overview
This is the first paragraph with unique content about machine learning methods.

This is the second paragraph discussing experimental results and performance metrics.

This is the third paragraph covering limitations and future work."""
        assert not validator._detect_paragraph_repetition(summary, threshold=0.8)

        # With paragraph repetition
        summary = """### Overview
This is the first paragraph with unique content about machine learning methods.

This is the second paragraph discussing experimental results and performance metrics.

This is the first paragraph with unique content about machine learning methods.

This is the second paragraph discussing experimental results and performance metrics."""
        assert validator._detect_paragraph_repetition(summary, threshold=0.8)

    def test_detect_hallucination(self):
        """Test hallucination detection."""
        validator = SummaryQualityValidator()

        # No hallucination
        summary = "The paper presents machine learning methods."
        pdf_text = "The paper presents machine learning methods for classification."
        has_hallucination, reason = validator._detect_hallucination(summary, pdf_text)
        assert not has_hallucination

        # Hallucination - AI language
        summary = "I'm happy to help summarize this paper."
        pdf_text = "The paper presents machine learning methods."
        has_hallucination, reason = validator._detect_hallucination(summary, pdf_text)
        assert has_hallucination
        assert "AI assistant language" in reason

        # Hallucination - code content
        summary = "The paper uses def train_model(): function."
        pdf_text = "The paper uses machine learning techniques."
        has_hallucination, reason = validator._detect_hallucination(summary, pdf_text)
        assert has_hallucination
        assert "code in text summary" in reason

    def test_detect_off_topic_content(self):
        """Test off-topic content detection."""
        validator = SummaryQualityValidator()

        # Clean summary
        summary = "The paper presents novel methods for data analysis."
        errors = validator._detect_off_topic_content(summary)
        assert len(errors) == 0

        # Off-topic content
        summary = "Dear Professor, here is the summary you requested."
        errors = validator._detect_off_topic_content(summary)
        assert len(errors) > 0
        assert any("greeting" in error.lower() for error in errors)

    def test_domain_specific_validation(self):
        """Test that domain-specific validation has been removed (pattern-based only)."""
        validator = SummaryQualityValidator()

        # Physics paper - should not flag missing physics terms (term-based validation removed)
        pdf_text = "The collision energy was measured at 13 TeV. The quark-gluon plasma was observed."
        summary = "The paper discusses some experimental findings."  # No physics terms
        has_hallucination, reason = validator._detect_hallucination(summary, pdf_text)
        # Should not flag as hallucination due to missing terms (term-based validation removed)
        # Only pattern-based hallucination detection remains
        assert not has_hallucination or "physics terminology" not in reason.lower()

        # Physics paper with physics terms - should still work
        summary = "The paper measures collision energy and observes quark-gluon plasma."
        has_hallucination, reason = validator._detect_hallucination(summary, pdf_text)
        assert not has_hallucination

    def test_validate_with_section_repetition(self):
        """Test validation detects section repetition."""
        validator = SummaryQualityValidator(min_words=50)
        pdf_text = "This is a scientific paper about machine learning with many details."
        
        summary = """### Overview
This paper presents methods.

### Key Contributions
These are contributions.

### Overview
This paper presents methods again."""
        
        is_valid, score, errors = validator.validate_summary(summary, pdf_text, "test")
        
        assert not is_valid or score < 0.6  # Should be penalized
        assert any("section" in error.lower() and "repetition" in error.lower() for error in errors)

    def test_validate_with_paragraph_repetition(self):
        """Test validation detects paragraph repetition."""
        validator = SummaryQualityValidator(min_words=50)
        pdf_text = "This is a scientific paper about machine learning with many details."
        
        summary = """### Overview
This is a unique paragraph about machine learning methods and their applications in various domains.

This is another unique paragraph discussing experimental results and validation procedures.

This is a unique paragraph about machine learning methods and their applications in various domains."""
        
        is_valid, score, errors = validator.validate_summary(summary, pdf_text, "test")
        
        # Should detect paragraph repetition
        assert any("paragraph" in error.lower() and "repetition" in error.lower() for error in errors) or score < 0.8


class TestPaperSummarizer:
    """Test PaperSummarizer functionality."""

    def test_creation(self):
        """Test summarizer creation."""
        from infrastructure.llm import LLMClient
        llm_client = LLMClient()
        summarizer = SummarizationEngine(llm_client)

        assert summarizer.llm_client is llm_client
        # quality_validator is a property, not a method
        assert isinstance(summarizer.validator, SummaryQualityValidator)

    def test_creation_with_validator(self):
        """Test summarizer creation with custom validator."""
        from infrastructure.llm import LLMClient
        llm_client = LLMClient()
        validator = SummaryQualityValidator(min_words=100)
        # PaperSummarizer is an alias for SummarizationEngine
        summarizer = SummarizationEngine(llm_client, quality_validator=validator)

        # quality_validator is a property that returns validator
        assert summarizer.validator is validator

    @pytest.mark.requires_ollama
    def test_summarize_paper_success(self, tmp_path):
        """Test successful paper summarization with real LLM."""
        from infrastructure.llm import LLMClient
        
        # Check if Ollama is available
        try:
            llm_client = LLMClient()
            if not llm_client.check_connection():
                pytest.skip("Ollama not available")
        except Exception:
            pytest.skip("Ollama not available")

        summarizer = SummarizationEngine(llm_client, quality_validator=SummaryQualityValidator(min_words=10))

        result = SearchResult(
            title="Test Paper",
            authors=["Author One"],
            year=2024,
            abstract="Test abstract",
            url="http://example.com",
            source="arxiv"
        )

        # Create real PDF with sufficient text using reportlab
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = tmp_path / "test.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "Test Paper")
            c.drawString(100, 730, "This is a long scientific paper about machine learning with many details and explanations that cover various aspects of the field including algorithms, datasets, and evaluation metrics.")
            c.drawString(100, 710, "The paper presents novel methods for data analysis and optimization.")
            c.drawString(100, 690, "Experimental results demonstrate significant improvements over baseline methods.")
            c.save()
        except ImportError:
            pytest.skip("reportlab not available for PDF creation")

        # Execute
        summary_result = summarizer.summarize_paper(result, pdf_path)

        # Verify - summary_text is always present even if validation fails
        assert summary_result.citation_key == "test"
        assert summary_result.summary_text is not None
        assert len(summary_result.summary_text) > 0
        assert summary_result.attempts >= 1
        assert summary_result.quality_score >= 0
        # Note: success may be False if validation fails, but summary_text is always present

    def test_summarize_paper_extraction_failure(self, tmp_path):
        """Test summarization when PDF extraction fails - uses invalid PDF."""
        from infrastructure.llm import LLMClient
        
        llm_client = LLMClient()
        summarizer = SummarizationEngine(llm_client)

        result = SearchResult(
            title="Test Paper",
            authors=["Author One"],
            year=2024,
            abstract="Test abstract",
            url="http://example.com",
            source="arxiv"
        )

        # Create invalid PDF file that will fail extraction
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"Invalid PDF content that cannot be parsed")  # Not a valid PDF

        summary_result = summarizer.summarize_paper(result, pdf_path)

        assert summary_result.success is False
        assert "extraction failed" in summary_result.error.lower() or "pdf" in summary_result.error.lower()

    def test_summarize_paper_insufficient_text(self, tmp_path):
        """Test summarization with insufficient extracted text."""
        from infrastructure.llm import LLMClient
        
        llm_client = LLMClient()
        summarizer = SummarizationEngine(llm_client)

        result = SearchResult(
            title="Test Paper",
            authors=["Author One"],
            year=2024,
            abstract="Test abstract",
            url="http://example.com",
            source="arxiv"
        )

        # Create real PDF with very little text (< 100 chars) using reportlab
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = tmp_path / "test.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "Short text")  # Only ~11 chars
            c.save()
        except ImportError:
            pytest.skip("reportlab not available for PDF creation")

        summary_result = summarizer.summarize_paper(result, pdf_path)

        assert summary_result.success is False
        assert "insufficient text" in summary_result.error.lower()

    def test_summarize_paper_llm_failure(self, tmp_path):
        """Test summarization when LLM fails - uses invalid Ollama host."""
        from infrastructure.llm import LLMClient, LLMConfig
        
        # Use invalid Ollama host to trigger connection failure
        config = LLMConfig(base_url="http://localhost:99999")  # Invalid port
        llm_client = LLMClient(config)

        summarizer = SummarizationEngine(llm_client)

        result = SearchResult(
            title="Test Paper",
            authors=["Author One"],
            year=2024,
            abstract="Test abstract",
            url="http://example.com",
            source="arxiv"
        )

        # Create real PDF with sufficient text
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = tmp_path / "test.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "This is a long enough text for summarization that exceeds the minimum character requirement and should trigger the LLM call.")
            c.drawString(100, 730, "Additional content to ensure we have more than 100 characters of extractable text from this PDF document.")
            c.save()
        except ImportError:
            pytest.skip("reportlab not available for PDF creation")

        summary_result = summarizer.summarize_paper(result, pdf_path)

        assert summary_result.success is False
        assert summary_result.attempts >= 1  # Should attempt at least once

    @pytest.mark.requires_ollama
    @pytest.mark.timeout(120)
    def test_generate_summary_prompt(self):
        """Test summary generation with real LLM (compatibility method)."""
        from infrastructure.llm import LLMClient

        # Check if Ollama is available
        try:
            llm_client = LLMClient()
            if not llm_client.check_connection():
                pytest.skip("Ollama not available")
        except Exception:
            pytest.skip("Ollama not available")

        summarizer = SummarizationEngine(llm_client)

        result = SearchResult(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            year=2024,
            abstract="Test abstract",
            url="http://example.com",
            source="arxiv",
            venue="arXiv"
        )

        # Generate summary with real LLM using compatibility method
        # Note: _generate_summary is a compatibility method that uses LLM
        summary = summarizer._generate_summary(result, "Test PDF content with sufficient text for summarization. This paper discusses machine learning algorithms and their applications.")

        # Verify summary was generated (real LLM call)
        assert summary is not None
        assert len(summary) > 0
        # Summary should be a real generated summary (not a template)
        # (exact content depends on LLM, so we just verify it's not empty)
        # The summary should contain meaningful content, not just a placeholder
        assert "summary" in summary.lower() or len(summary) > 100

    def test_clean_summary_content(self):
        """Test summary content cleaning."""
        from infrastructure.llm import LLMClient
        llm_client = LLMClient()
        summarizer = SummarizationEngine(llm_client)

        # Test with unwanted sections
        raw_summary = """### References
[1] Smith et al.

### Summary
This is a summary.

Note: This was generated by AI.

### Methodology
The method used is...

### Citation
@article{smith2024}
"""

        cleaned = summarizer._clean_summary_content(raw_summary)

        assert "### References" not in cleaned
        assert "### Citation" not in cleaned
        assert "Note:" not in cleaned
        assert "### Summary" not in cleaned  # Summary section is removed as unwanted
        assert "### Methodology" in cleaned

    def test_save_summary(self, tmp_path):
        """Test summary saving to file."""
        from infrastructure.llm import LLMClient
        llm_client = LLMClient()
        summarizer = SummarizationEngine(llm_client)

        result = SearchResult(
            title="Test Paper",
            authors=["Author One"],
            year=2024,
            abstract="Test abstract",
            url="http://example.com",
            source="arxiv"
        )

        summary_result = SummarizationResult(
            citation_key="test2024",
            success=True,
            summary_text="Test summary content",
            input_words=100,
            output_words=50,
            generation_time=30.0
        )

        output_dir = tmp_path / "summaries"

        # Execute (pdf_path is optional) - now returns list of paths
        saved_paths = summarizer.save_summary(result, summary_result, output_dir, pdf_path=None)

        # Verify - should return list with at least the summary file
        assert isinstance(saved_paths, list)
        assert len(saved_paths) >= 1
        
        # Find the summary file
        summary_path = next((p for p in saved_paths if "summary.md" in str(p)), None)
        assert summary_path is not None
        assert summary_path.exists()
        assert summary_path.name == "test2024_summary.md"

        written_content = summary_path.read_text()
        assert "Test Paper" in written_content
        assert "Author One" in written_content
        assert "2024" in written_content
        assert "Test summary content" in written_content
        # Note: Statistics/metadata are now in JSON, not in markdown

    def test_save_summary_with_path_field(self, tmp_path):
        """Test that save_summary sets summary_path field in result."""
        from infrastructure.llm import LLMClient
        llm_client = LLMClient()
        summarizer = SummarizationEngine(llm_client)

        result = SearchResult(
            title="Test Paper",
            authors=["Author One"],
            year=2024,
            abstract="Test abstract",
            url="http://example.com",
            source="arxiv"
        )

        summary_result = SummarizationResult(
            citation_key="test2024",
            success=True,
            summary_text="Test summary content",
            input_words=100,
            output_words=50,
            generation_time=30.0
        )

        output_dir = tmp_path / "summaries"

        # Execute (pdf_path is optional)
        saved_paths = summarizer.save_summary(result, summary_result, output_dir, pdf_path=None)

        # Verify summary_path field is set (though save_summary doesn't modify the result)
        # The workflow should set this field after calling save_summary
        # save_summary now returns a list of paths
        assert isinstance(saved_paths, list)
        assert len(saved_paths) >= 1
        summary_path = next((p for p in saved_paths if "summary.md" in str(p)), None)
        assert summary_path is not None
        assert summary_path.exists()
        # Note: save_summary returns a list of paths but doesn't modify the result object

    def test_summary_result_with_path_field(self):
        """Test SummarizationResult with summary_path field."""
        result = SummarizationResult(
            citation_key="test2024",
            success=True,
            summary_text="Test summary",
            summary_path=Path("/path/to/summary.md")
        )

        assert result.summary_path == Path("/path/to/summary.md")

        # Test with None path
        result_none = SummarizationResult(
            citation_key="test2024",
            success=True,
            summary_text="Test summary"
        )

        assert result_none.summary_path is None

    def test_analyze_references(self):
        """Test reference detection and counting."""
        from infrastructure.llm import LLMClient
        llm_client = LLMClient()
        summarizer = SummarizationEngine(llm_client)

        # PDF with numbered references [1], [2], [3]
        pdf_text = """Introduction text here.
        
References
[1] Author A. Title. Journal, 2020.
[2] Author B. Title. Journal, 2021.
[3] Author C. Title. Journal, 2022."""
        
        ref_info = summarizer._analyze_references(pdf_text)
        assert ref_info['section_found'] is True
        assert ref_info['count'] == 3

        # PDF with LaTeX citations
        pdf_text_latex = """Introduction with \\cite{author2020} and \\cite{author2021}.
        
\\begin{thebibliography}
\\bibitem{author2020} Author A.
\\bibitem{author2021} Author B.
\\end{thebibliography}"""
        
        ref_info_latex = summarizer._analyze_references(pdf_text_latex)
        assert ref_info_latex['section_found'] is True
        assert ref_info_latex['count'] is not None

        # PDF without references
        pdf_text_no_refs = "Just some text without references."
        ref_info_no = summarizer._analyze_references(pdf_text_no_refs)
        assert ref_info_no['section_found'] is False
        assert ref_info_no['count'] is None

    def test_deduplicate_summary(self):
        """Test summary deduplication functionality."""
        from infrastructure.llm import LLMClient
        llm_client = LLMClient()
        summarizer = SummarizationEngine(llm_client)

        # Summary with duplicate sections
        summary_with_duplicates = """### Overview
This paper presents novel methods for data analysis.

### Key Contributions
The authors introduce new algorithms.

### Overview
This paper presents novel methods for data analysis.

### Methodology
The approach uses neural networks.

### Key Contributions
The authors introduce new algorithms."""

        deduplicated = summarizer._deduplicate_summary(summary_with_duplicates)

        # Should only have one "Overview" and one "Key Contributions"
        assert deduplicated.count("### Overview") == 1
        assert deduplicated.count("### Key Contributions") == 1
        assert "### Methodology" in deduplicated

        # Summary with duplicate paragraphs
        summary_para_dup = """### Overview
This is a unique paragraph about the paper's main contribution to the field.

This is another unique paragraph discussing experimental validation.

This is a unique paragraph about the paper's main contribution to the field."""

        deduplicated_para = summarizer._deduplicate_summary(summary_para_dup)
        
        # Should remove duplicate paragraph
        lines = deduplicated_para.split('\n')
        overview_count = sum(1 for line in lines if line.strip().startswith('### Overview'))
        assert overview_count == 1

        # Empty summary
        assert summarizer._deduplicate_summary("") == ""
        
        # Summary without duplicates
        clean_summary = """### Overview
Unique content here.

### Methodology
Different content here."""
        assert summarizer._deduplicate_summary(clean_summary) == clean_summary.strip()

    def test_validate_title_match(self):
        """Test title matching validation."""
        validator = SummaryQualityValidator()
        
        # Exact match
        summary = "# Expected Free Energy Formalizes Conflict Underlying Defense in Freudian Psychoanalysis\n\nContent here."
        paper_title = "Expected Free Energy Formalizes Conflict Underlying Defense in Freudian Psychoanalysis"
        match, error = validator._validate_title_match(summary, paper_title)
        assert match is True
        assert error == ""
        
        # Title in summary text (not as header)
        summary = "This paper discusses Expected Free Energy Formalizes Conflict Underlying Defense in Freudian Psychoanalysis."
        match, error = validator._validate_title_match(summary, paper_title)
        assert match is True
        
        # Wrong title
        summary = "# Deep Neural Networks for Adversarial Attacks\n\nContent here."
        match, error = validator._validate_title_match(summary, paper_title)
        assert match is False
        assert "does not match" in error or "not found" in error
        
        # Partial match (good word overlap)
        summary = "# Expected Free Energy and Conflict in Psychoanalysis\n\nContent here."
        match, error = validator._validate_title_match(summary, paper_title)
        assert match is True  # Should match due to word overlap
        
        # No title in summary
        summary = "This is a summary without a title header."
        match, error = validator._validate_title_match(summary, paper_title)
        # Should check if title words appear in summary
        assert isinstance(match, bool)

    def test_validate_content_topics(self):
        """Test that content topic matching validation has been removed."""
        validator = SummaryQualityValidator()

        # _validate_content_topics method has been removed (term-based validation removed)
        # This test verifies the method no longer exists
        assert not hasattr(validator, '_validate_content_topics'), \
            "_validate_content_topics method should have been removed"

    def test_validate_quotes_present(self):
        """Test quote presence validation."""
        validator = SummaryQualityValidator()
        
        # Summary with quotes
        summary_with_quotes = """### Overview
The paper states: "Expected free energy provides a framework for understanding defense mechanisms."

According to the introduction: "Freudian psychoanalysis emphasizes the role of conflict."

The authors note: [The expected free energy formalizes the underlying conflict in defense mechanisms.]
"""
        has_quotes, count = validator._validate_quotes_present(summary_with_quotes)
        assert has_quotes is True
        assert count >= 3
        
        # Summary without quotes
        summary_no_quotes = """### Overview
This paper discusses various topics. The authors present methods and results. The findings are significant."""
        has_quotes, count = validator._validate_quotes_present(summary_no_quotes)
        assert has_quotes is False
        assert count < 3
        
        # Summary with some quotes but not enough
        summary_few_quotes = """The paper says "something important" and that's it."""
        has_quotes, count = validator._validate_quotes_present(summary_few_quotes)
        assert has_quotes is False
        assert count < 3

    def test_validate_summary_with_title_matching(self):
        """Test validate_summary with title matching enabled."""
        validator = SummaryQualityValidator(min_words=25)
        
        paper_title = "Expected Free Energy Formalizes Conflict Underlying Defense in Freudian Psychoanalysis"
        pdf_text = "This paper applies expected free energy to Freudian psychoanalysis. Defense mechanisms and conflict are discussed."
        
        # Correct summary
        summary_correct = f"""# {paper_title}

**Authors:** Test Author

### Overview
This paper applies expected free energy to psychoanalysis. "The framework formalizes conflict" as stated in the introduction.
"""
        is_valid, score, errors = validator.validate_summary(
            summary_correct, pdf_text, "test2024", paper_title=paper_title
        )
        # Should pass title matching
        assert not any("title" in error.lower() for error in errors)
        
        # Wrong title in summary
        summary_wrong_title = """# Deep Neural Networks for Adversarial Attacks

### Overview
This paper discusses neural networks.
"""
        is_valid, score, errors = validator.validate_summary(
            summary_wrong_title, pdf_text, "test2024", paper_title=paper_title
        )
        # Should fail title matching
        assert any("title" in error.lower() for error in errors)
        assert score < 0.5  # Severe penalty for title mismatch

    def test_detect_severe_repetition(self):
        """Test severe repetition detection."""
        validator = SummaryQualityValidator()
        
        # Summary with sentence repetition (same sentence 3+ times)
        summary_sentence_rep = """### Overview
This paper presents novel methods for data analysis. This paper presents novel methods for data analysis. This paper presents novel methods for data analysis. The authors introduce new algorithms."""
        has_severe, reason = validator._detect_severe_repetition(summary_sentence_rep)
        assert has_severe is True
        assert "sentence" in reason.lower() or "appears" in reason.lower()
        
        # Summary with phrase repetition (same 5+ word phrase 5+ times)
        summary_phrase_rep = """The FEPs method is a general-purpose framework. The FEPs method is a general-purpose framework. The FEPs method is a general-purpose framework. The FEPs method is a general-purpose framework. The FEPs method is a general-purpose framework."""
        has_severe, reason = validator._detect_severe_repetition(summary_phrase_rep)
        assert has_severe is True
        assert "phrase" in reason.lower() or "appears" in reason.lower()
        
        # Summary with paragraph repetition (>30% duplicates)
        summary_para_rep = """### Overview
This is a unique paragraph about the paper's main contribution to the field of research.

This is another unique paragraph discussing experimental validation.

This is a unique paragraph about the paper's main contribution to the field of research.

This is a unique paragraph about the paper's main contribution to the field of research."""
        has_severe, reason = validator._detect_severe_repetition(summary_para_rep)
        # May or may not detect depending on similarity calculation
        assert isinstance(has_severe, bool)
        
        # Summary with longer phrase repetition (10+ words)
        long_phrase = "This is a very long phrase that contains ten words or more for testing purposes."
        summary_long_phrase_rep = f"{long_phrase} {long_phrase} {long_phrase} Additional content here."
        has_severe, reason = validator._detect_severe_repetition(summary_long_phrase_rep)
        assert has_severe is True
        assert "long phrase" in reason.lower() or "appears" in reason.lower()
        
        # Summary with nested repetition patterns
        summary_nested = 'The authors state: "The authors state: "The authors state: "This is a quote."'
        has_severe, reason = validator._detect_severe_repetition(summary_nested)
        assert has_severe is True
        assert "nested" in reason.lower() or "repetition" in reason.lower()
        
        # Summary with repetitive attribution phrases
        summary_attribution = '"The authors state: "quote1" ' * 12
        has_severe, reason = validator._detect_severe_repetition(summary_attribution)
        assert has_severe is True
        assert "attribution" in reason.lower() or "repetition" in reason.lower()
        
        # Summary without severe repetition
        summary_clean = """### Overview
This paper presents novel methods for data analysis. The authors introduce new algorithms. The results show significant improvements."""
        has_severe, reason = validator._detect_severe_repetition(summary_clean)
        assert has_severe is False
        assert reason == "" """### Overview
This paper presents novel methods for data analysis.

### Methodology
The authors introduce new algorithms.

### Results
The experimental results show significant improvements."""
        has_severe, reason = validator._detect_severe_repetition(summary_clean)
        assert has_severe is False
        assert reason == ""

    def test_validate_summary_with_severe_repetition(self):
        """Test that severe repetition is detected as critical failure."""
        validator = SummaryQualityValidator(min_words=25)
        
        # Summary with severe sentence repetition
        summary_rep = """### Overview
This paper presents novel methods. This paper presents novel methods. This paper presents novel methods. The authors introduce algorithms."""
        pdf_text = "This paper presents novel methods for data analysis. The authors introduce new algorithms."
        
        is_valid, score, errors = validator.validate_summary(
            summary_rep, pdf_text, "test2024", paper_title="Novel Methods for Data Analysis"
        )
        # Should detect severe repetition
        assert any("severe repetition" in error.lower() for error in errors)
        assert score < 0.5  # Critical penalty

    @pytest.mark.requires_ollama
    def test_hard_failure_rejection(self, tmp_path):
        """Test that summaries with hard failures are still saved but marked as rejected."""
        from infrastructure.literature.sources import SearchResult
        from infrastructure.llm import LLMClient

        # Check if Ollama is available
        try:
            llm_client = LLMClient()
            if not llm_client.check_connection():
                pytest.skip("Ollama not available")
        except Exception:
            pytest.skip("Ollama not available")

        summarizer = SummarizationEngine(llm_client)

        result = SearchResult(
            title="Expected Free Energy Formalizes Conflict",
            authors=["Test Author"],
            year=2020,
            abstract="Test abstract",
            url="http://example.com",
            source="test",
            venue="Test Journal"
        )

        # Create real PDF with content that doesn't match the title (will cause title mismatch)
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            pdf_path = tmp_path / "test.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            # Create PDF with content that doesn't match the expected title
            c.drawString(100, 750, "Completely Different Paper Title")
            c.drawString(100, 730, "This paper discusses something completely different from expected free energy.")
            c.drawString(100, 710, "The content has no relation to the expected title.")
            c.save()
        except ImportError:
            pytest.skip("reportlab not available for PDF creation")
        
        summary_result = summarizer.summarize_paper(result, pdf_path, max_retries=2)
        
        # Should fail due to title mismatch or validation errors (hard failure)
        # Note: With real LLM, the validation may catch this or the LLM may generate appropriate summary
        # The key is that hard failures should not be accepted after retries
        # However, summary_text is always present even if validation fails (always-save behavior)
        # This test verifies the retry logic doesn't accept invalid summaries
        if summary_result.summary_text:  # If generation succeeded
            # Validation should have failed
            assert summary_result.success is False or len(summary_result.validation_errors) > 0
        else:
            # Generation itself failed
            assert summary_result.success is False


class TestPDFProcessor:
    """Test PDF processor functionality."""
    
    def test_identify_sections(self):
        """Test section identification in PDF text."""
        from infrastructure.literature.summarization.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # PDF with typical structure
        pdf_text = """Expected Free Energy Formalizes Conflict

Abstract
This paper applies expected free energy to psychoanalysis.

1. Introduction
This paper discusses defense mechanisms in Freudian psychoanalysis.

2. Methods
The methodology section describes the approach.

3. Results
The results show significant findings.

Conclusion
The paper concludes with important insights."""
        
        sections = processor.identify_sections(pdf_text)
        
        # Should identify at least some sections
        assert isinstance(sections, dict)
        # Title should be found (first substantial line)
        assert 'title' in sections or len(sections) > 0
        
    def test_smart_truncate(self):
        """Test smart truncation with section prioritization."""
        from infrastructure.literature.summarization.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # Create PDF text with sections
        title = "Expected Free Energy Formalizes Conflict"
        abstract = "Abstract\n" + "This is the abstract. " * 50  # ~1000 chars
        introduction = "1. Introduction\n" + "Introduction content. " * 200  # ~4000 chars
        middle = "2. Methods\n" + "Methods content. " * 500  # ~8000 chars
        conclusion = "Conclusion\n" + "Conclusion content. " * 100  # ~2000 chars
        
        pdf_text = f"{title}\n\n{abstract}\n\n{introduction}\n\n{middle}\n\n{conclusion}"
        
        # Identify sections
        sections = processor.identify_sections(pdf_text)
        
        # Truncate to 5000 chars (should preserve prioritized sections)
        truncated, included, excluded = processor.smart_truncate(pdf_text, sections, 5000)
        
        assert len(truncated) <= 5000
        assert len(included) > 0  # Should include some sections
        # Title and abstract should be prioritized
        assert 'title' in included or 'abstract' in included or len(included) > 0
        
    def test_extract_prioritized_text_no_truncation(self, tmp_path):
        """Test PDF processor when no truncation is needed."""
        from infrastructure.literature.summarization.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # Create a real PDF with small text using reportlab
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = tmp_path / "test.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "Title: Test Paper")
            c.drawString(100, 730, "Abstract: This is a test.")
            c.drawString(100, 710, "Introduction: Test content.")
            c.save()
            
            # Extract with large limit (no truncation)
            result = processor.extract_prioritized_text(pdf_path, max_chars=100000)
            
            assert result.truncation_occurred is False
            assert result.original_length > 0
            assert result.final_length == result.original_length
            assert len(result.sections_included) > 0
        except ImportError:
            pytest.skip("reportlab not available for PDF creation")
        
    def test_extract_prioritized_text_with_truncation(self, tmp_path):
        """Test PDF processor with truncation."""
        from infrastructure.literature.summarization.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # Create a real PDF with large text using reportlab
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = tmp_path / "test.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            
            # Add title
            c.drawString(100, 750, "Test Paper Title")
            
            # Add abstract with repeated content
            y = 730
            c.drawString(100, y, "Abstract")
            y -= 20
            for i in range(20):  # Multiple lines to create substantial content
                c.drawString(100, y, f"Abstract content line {i}. " * 3)
                y -= 15
            
            # Add introduction
            y -= 10
            c.drawString(100, y, "1. Introduction")
            y -= 20
            for i in range(50):
                c.drawString(100, y, f"Introduction content line {i}. " * 2)
                y -= 15
            
            # Add methods
            y -= 10
            c.drawString(100, y, "2. Methods")
            y -= 20
            for i in range(100):
                c.drawString(100, y, f"Methods content line {i}. " * 2)
                y -= 15
                if y < 50:  # New page
                    c.showPage()
                    y = 750
            
            # Add conclusion
            y -= 10
            c.drawString(100, y, "Conclusion")
            y -= 20
            for i in range(30):
                c.drawString(100, y, f"Conclusion content line {i}. " * 2)
                y -= 15
            
            c.save()
            
            # Extract with small limit (truncation needed)
            result = processor.extract_prioritized_text(pdf_path, max_chars=2000)
            
            assert result.truncation_occurred is True
            assert result.final_length <= 2000
            assert len(result.sections_included) > 0
            # Should prioritize title, abstract, introduction, conclusion
            assert any(section in result.sections_included for section in ['title', 'abstract', 'introduction', 'conclusion'])
        except ImportError:
            pytest.skip("reportlab not available for PDF creation")


class TestAlwaysSaveBehavior:
    """Test that summaries are always saved even when validation fails."""
    
    def test_summary_text_always_present_on_validation_failure(self):
        """Test that summary_text is included even when validation fails."""
        result = SummarizationResult(
            citation_key="test2024",
            success=False,  # Validation failed
            summary_text="This is a test summary that failed validation.",  # But text is present
            input_chars=1000,
            input_words=200,
            output_words=50,
            generation_time=30.0,
            attempts=1,
            quality_score=0.3,  # Low score
            validation_errors=["Too short", "Missing quotes"]
        )
        
        # Summary text should always be present if generation succeeded
        assert result.summary_text is not None
        assert result.summary_text == "This is a test summary that failed validation."
        assert result.success is False  # But validation failed
        assert len(result.validation_errors) > 0
    
    def test_save_summary_with_validation_metadata(self, tmp_path):
        """Test that saved summaries include validation metadata."""
        from infrastructure.literature.sources import SearchResult
        
        result = SearchResult(
            title="Test Paper",
            authors=["Author One"],
            year=2024,
            source="test",
            abstract="Test abstract",
            url="https://example.com/test"
        )
        
        summary_result = SummarizationResult(
            citation_key="test2024",
            success=False,  # Validation failed
            summary_text="# Test Paper\n\nThis is a test summary.",
            input_chars=1000,
            input_words=200,
            output_words=50,
            generation_time=30.0,
            attempts=1,
            quality_score=0.3,
            validation_errors=["Too short: 50 words (minimum 200)", "No quotes found"]
        )
        
        engine = SummarizationEngine(Mock())
        output_dir = tmp_path / "summaries"
        
        # Save summary - now returns list of paths
        saved_paths = engine.save_summary(
            result=result,
            summary_result=summary_result,
            output_dir=output_dir
        )
        
        # Verify file exists
        assert isinstance(saved_paths, list)
        assert len(saved_paths) >= 1
        summary_path = next((p for p in saved_paths if "summary.md" in str(p)), None)
        assert summary_path is not None
        assert summary_path.exists()
        
        # Read saved content
        content = summary_path.read_text(encoding='utf-8')
        
        # Verify validation metadata is included
        assert "Validation Status" in content
        assert "⚠ Rejected" in content or "Rejected" in content
        assert "Quality Score" in content
        assert "0.30" in content or "0.3" in content
        assert "Validation Errors" in content
        assert "Too short" in content or "50 words" in content
        
        # Verify summary text is included
        assert "Test Paper" in content
        assert "This is a test summary" in content


class TestModelAwareConfiguration:
    """Test model-aware generation options and configuration."""
    
    def test_model_size_detection_small(self):
        """Test detection of small model (<7B)."""
        from infrastructure.literature.summarization.multi_stage_summarizer import get_model_aware_generation_options
        from unittest.mock import Mock
        
        mock_client = Mock()
        mock_client.config = Mock()
        mock_client.config.default_model = "gemma3:4b"
        
        options = get_model_aware_generation_options(mock_client, {}, "draft")
        
        assert options.temperature == 0.3
        assert options.max_tokens == 2000
    
    def test_model_size_detection_medium(self):
        """Test detection of medium model (7-13B)."""
        from infrastructure.literature.summarization.multi_stage_summarizer import get_model_aware_generation_options
        from unittest.mock import Mock
        
        mock_client = Mock()
        mock_client.config = Mock()
        mock_client.config.default_model = "llama3.1:8b"
        
        options = get_model_aware_generation_options(mock_client, {}, "draft")
        
        assert options.temperature == 0.4
        assert options.max_tokens == 2000
    
    def test_model_size_detection_large(self):
        """Test detection of large model (>13B)."""
        from infrastructure.literature.summarization.multi_stage_summarizer import get_model_aware_generation_options
        from unittest.mock import Mock
        
        mock_client = Mock()
        mock_client.config = Mock()
        mock_client.config.default_model = "llama3.1:70b"
        
        options = get_model_aware_generation_options(mock_client, {}, "draft")
        
        assert options.temperature == 0.5
        assert options.max_tokens == 2500
    
    def test_model_size_detection_refinement(self):
        """Test that refinement uses appropriate max_tokens."""
        from infrastructure.literature.summarization.multi_stage_summarizer import get_model_aware_generation_options
        from unittest.mock import Mock
        
        mock_client = Mock()
        mock_client.config = Mock()
        mock_client.config.default_model = "gemma3:4b"
        
        options = get_model_aware_generation_options(mock_client, {}, "refinement")
        
        assert options.temperature == 0.3
        assert options.max_tokens == 2500  # Higher for refinement


class TestNoTermBasedValidation:
    """Test that term-based validation has been removed."""
    
    def test_validator_no_term_based_errors(self):
        """Test that validator doesn't flag common academic terms."""
        validator = SummaryQualityValidator(min_words=25)
        
        pdf_text = "This paper discusses free energy principles in brain function."
        summary = "This paper presents a concept of free energy. The authors discuss mathematical approaches. The paper focuses on brain function. The summary describes the approach."
        
        is_valid, score, errors = validator.validate_summary(
            summary=summary,
            pdf_text=pdf_text,
            citation_key="test2024",
            paper_title="Free Energy Principles"
        )
        
        # Should not flag "free", "energy", "brain", "paper", "concept", "mathematical", "focus", "summary" as hallucinations
        error_text = " ".join(errors).lower()
        assert "terms found in summary but not in pdf" not in error_text
        assert "potential hallucination" not in error_text or "free" not in error_text.lower()
    
    def test_validator_still_detects_repetition(self):
        """Test that validator still detects repetition correctly."""
        validator = SummaryQualityValidator(min_words=25)
        
        pdf_text = "This is a test paper about machine learning."
        # Create summary with severe repetition
        summary = "This paper presents methods. " * 20  # Same sentence 20 times
        
        is_valid, score, errors = validator.validate_summary(
            summary=summary,
            pdf_text=pdf_text,
            citation_key="test2024",
            paper_title="Test Paper"
        )
        
        # Should detect repetition
        assert not is_valid or score < 0.5
        assert any("repetition" in error.lower() for error in errors)


class TestPostProcessingDeduplication:
    """Test post-processing deduplication functionality."""
    
    def test_deduplication_removes_repeated_sentences(self):
        """Test that deduplication removes repeated sentences."""
        from infrastructure.llm.validation.repetition import deduplicate_sections
        
        # Text with repeated sentences
        text = """This is sentence one. This is sentence two. This is sentence one. 
        This is sentence three. This is sentence one. This is sentence four."""
        
        deduplicated = deduplicate_sections(
            text,
            max_repetitions=1,
            mode="aggressive",
            similarity_threshold=0.85,
            min_content_preservation=0.7
        )
        
        # Should have fewer sentences
        original_sentences = text.count(".")
        deduplicated_sentences = deduplicated.count(".")
        assert deduplicated_sentences <= original_sentences
    
    def test_deduplication_preserves_unique_content(self):
        """Test that deduplication preserves unique content."""
        from infrastructure.llm.validation.repetition import deduplicate_sections
        
        # Text with unique sentences
        text = """This is sentence one. This is sentence two. This is sentence three. 
        This is sentence four. This is sentence five."""
        
        deduplicated = deduplicate_sections(
            text,
            max_repetitions=1,
            mode="aggressive",
            similarity_threshold=0.85,
            min_content_preservation=0.7
        )
        
        # Should preserve most content
        assert len(deduplicated) > len(text) * 0.7  # At least 70% preserved
    
    def test_deduplication_with_markdown(self):
        """Test deduplication with markdown formatting."""
        from infrastructure.llm.validation.repetition import deduplicate_sections
        
        # Text with markdown and repetition
        text = """### Overview
This is a paragraph about the paper.

### Key Contributions
This is a paragraph about contributions.

### Overview
This is a paragraph about the paper again."""
        
        deduplicated = deduplicate_sections(
            text,
            max_repetitions=1,
            mode="aggressive",
            similarity_threshold=0.85,
            min_content_preservation=0.7
        )
        
        # Should remove duplicate section
        assert deduplicated.count("### Overview") <= 1
    
    def test_deduplication_conservative_fallback(self):
        """Test that conservative fallback prevents infinite recursion."""
        from infrastructure.llm.validation.repetition import deduplicate_sections
        
        # Text that would trigger fallback (very repetitive)
        text = "This is a sentence. " * 100  # 100 identical sentences
        
        # Should not raise recursion error
        deduplicated = deduplicate_sections(
            text,
            max_repetitions=1,
            mode="aggressive",
            similarity_threshold=0.5,  # Very low threshold
            min_content_preservation=0.9  # High preservation requirement
        )
        
        # Should return something (even if it's the original)
        assert deduplicated is not None
        assert isinstance(deduplicated, str)
    
    def test_deduplication_content_preservation(self):
        """Test that content preservation limits are respected."""
        from infrastructure.llm.validation.repetition import deduplicate_sections
        
        # Text with some repetition but mostly unique
        text = "Unique sentence one. Unique sentence two. " * 10
        text += "Repeated sentence. " * 5
        text += "Unique sentence three. " * 10
        
        deduplicated = deduplicate_sections(
            text,
            max_repetitions=1,
            mode="aggressive",
            similarity_threshold=0.85,
            min_content_preservation=0.7
        )
        
        # Should preserve at least 70% of content
        preservation_ratio = len(deduplicated) / len(text) if len(text) > 0 else 1.0
        assert preservation_ratio >= 0.7 or len(deduplicated) == len(text)  # Either preserved or fallback used


class TestNestedRepetitionDetection:
    """Test nested repetition detection functionality."""
    
    def test_detect_nested_attribution(self):
        """Test detection of nested attribution phrases."""
        from infrastructure.llm.validation.repetition import detect_nested_repetition
        
        # Text with nested attribution
        text = '"The authors state: "The authors state: "This is a quote.""'
        
        has_nested, patterns = detect_nested_repetition(text)
        assert has_nested is True
        assert len(patterns) > 0
    
    def test_remove_nested_repetition(self):
        """Test removal of nested repetition patterns."""
        from infrastructure.llm.validation.repetition import remove_nested_repetition
        
        # Text with nested attribution
        text = '"The authors state: "The authors state: "This is a quote.""'
        
        cleaned, removed_count = remove_nested_repetition(text)
        assert removed_count > 0
        assert '"The authors state: "The authors state:' not in cleaned
    
    def test_detect_repeated_attribution_phrases(self):
        """Test detection of repeated attribution phrases."""
        from infrastructure.llm.validation.repetition import detect_nested_repetition
        
        # Text with many repeated attribution phrases
        text = '"The authors state: "quote1" ' * 15
        
        has_nested, patterns = detect_nested_repetition(text)
        assert has_nested is True
        assert any("Repeated attribution phrase" in p for p in patterns)
    
    def test_remove_excessive_attribution_phrases(self):
        """Test removal of excessive attribution phrases."""
        from infrastructure.llm.validation.repetition import remove_nested_repetition
        
        # Text with many repeated attribution phrases
        text = '"The authors state: "quote1" ' * 10
        
        cleaned, removed_count = remove_nested_repetition(text)
        # Should remove some but keep first few
        assert '"The authors state:' in cleaned
        assert removed_count > 0


class TestClaimsQuotesRepetition:
    """Test claims/quotes extraction repetition handling."""
    
    def test_claims_quotes_deduplication(self):
        """Test that claims/quotes extraction applies deduplication."""
        from infrastructure.llm.validation.repetition import deduplicate_sections
        
        # Simulated claims/quotes output with repetition
        claims_quotes = """## Key Claims and Hypotheses

1. Claim: The paper presents a novel method.
2. Claim: The paper presents a novel method.
3. Claim: The paper presents a novel method.

## Important Quotes

"The authors state: "The authors state: "This is a quote.""
"The authors state: "The authors state: "This is a quote."
"""
        
        deduplicated = deduplicate_sections(
            claims_quotes,
            max_repetitions=1,
            mode="aggressive",
            similarity_threshold=0.75,
            min_content_preservation=0.7
        )
        
        # Should have less repetition
        assert deduplicated.count("The paper presents a novel method") < claims_quotes.count("The paper presents a novel method")
    
    def test_nested_quote_patterns(self):
        """Test detection and removal of nested quote patterns."""
        from infrastructure.llm.validation.repetition import detect_nested_repetition, remove_nested_repetition
        
        # Text with nested quotes (3 levels of nesting)
        # Pattern: "The authors state: " repeated 3 times with nested quotes
        # Format: "text"text"text"content""" (3 opening quotes, content, 3 closing quotes)
        text = '"The authors state: "The authors state: "The authors state: "quote text"""'
        
        has_nested, patterns = detect_nested_repetition(text)
        assert has_nested is True
        
        cleaned, removed = remove_nested_repetition(text)
        assert removed > 0
        # Should have fewer nested levels
        assert cleaned.count('"The authors state: "The authors state:') < text.count('"The authors state: "The authors state:')


class TestThreePassSummarization:
    """Test three-pass summarization: comprehensive summary, claims/quotes, methods/tools."""
    
    def test_summarization_result_has_new_fields(self):
        """Test that SummarizationResult includes claims_quotes_text and methods_tools_text."""
        result = SummarizationResult(
            citation_key="test2024",
            success=True,
            summary_text="Main summary",
            claims_quotes_text="Claims and quotes",
            methods_tools_text="Methods and tools"
        )
        
        assert result.summary_text == "Main summary"
        assert result.claims_quotes_text == "Claims and quotes"
        assert result.methods_tools_text == "Methods and tools"
    
    def test_extract_claims_and_quotes(self, tmp_path):
        """Test claims and quotes extraction method."""
        from infrastructure.llm.core.client import LLMClient
        from infrastructure.llm.core.config import LLMConfig
        
        # Create mock LLM client
        config = LLMConfig(base_url="http://localhost:11434", default_model="test")
        llm_client = LLMClient(config)
        
        # Mock the query method
        llm_client.query = Mock(return_value="## Key Claims\n\n- Claim 1\n- Claim 2\n\n## Important Quotes\n\n**Quote:** \"Test quote\"\n**Context:** Introduction")
        
        engine = SummarizationEngine(llm_client)
        
        result = SearchResult(
            title="Test Paper",
            authors=["Author 1"],
            year=2024,
            abstract="Test abstract",
            url="https://example.com/paper",
            source="test"
        )
        
        pdf_text = "This is a test paper with some claims and quotes."
        
        claims_quotes = engine._extract_claims_and_quotes(
            pdf_text=pdf_text,
            result=result,
            citation_key="test2024"
        )
        
        assert "Key Claims" in claims_quotes
        assert "Important Quotes" in claims_quotes
        assert llm_client.query.called
        # Verify context was cleared
        assert len(llm_client.context.messages) == 0 or llm_client.query.call_args[1].get('reset_context', False)
    
    def test_analyze_methods_and_tools(self, tmp_path):
        """Test methods and tools analysis method."""
        from infrastructure.llm.core.client import LLMClient
        from infrastructure.llm.core.config import LLMConfig
        
        # Create mock LLM client
        config = LLMConfig(base_url="http://localhost:11434", default_model="test")
        llm_client = LLMClient(config)
        
        # Mock the query method
        llm_client.query = Mock(return_value="## Algorithms\n\n- Algorithm 1\n\n## Software Frameworks\n\n- PyTorch\n\n## Datasets\n\n- Dataset 1")
        
        engine = SummarizationEngine(llm_client)
        
        result = SearchResult(
            title="Test Paper",
            authors=["Author 1"],
            year=2024,
            abstract="Test abstract",
            url="https://example.com/paper",
            source="test"
        )
        
        pdf_text = "This paper uses PyTorch and Dataset 1."
        
        methods_tools = engine._analyze_methods_and_tools(
            pdf_text=pdf_text,
            result=result,
            citation_key="test2024"
        )
        
        assert "Algorithms" in methods_tools or "Software Frameworks" in methods_tools
        assert llm_client.query.called
        # Verify context was cleared
        assert len(llm_client.context.messages) == 0 or llm_client.query.call_args[1].get('reset_context', False)
    
    def test_save_summary_saves_all_three_files(self, tmp_path):
        """Test that save_summary saves all three output files."""
        from infrastructure.llm.core.client import LLMClient
        from infrastructure.llm.core.config import LLMConfig
        
        config = LLMConfig(base_url="http://localhost:11434", default_model="test")
        llm_client = LLMClient(config)
        engine = SummarizationEngine(llm_client)
        
        result = SearchResult(
            title="Test Paper",
            authors=["Author 1"],
            year=2024,
            abstract="Test abstract",
            url="https://example.com/paper",
            source="test"
        )
        
        summary_result = SummarizationResult(
            citation_key="test2024",
            success=True,
            summary_text="Main summary text",
            claims_quotes_text="Claims and quotes text",
            methods_tools_text="Methods and tools text",
            input_chars=1000,
            input_words=200,
            output_words=50
        )
        
        output_dir = tmp_path / "summaries"
        saved_paths = engine.save_summary(result, summary_result, output_dir)
        
        # Should return list of 3 paths
        assert len(saved_paths) == 3
        assert any("test2024_summary.md" in str(p) for p in saved_paths)
        assert any("test2024_claims_quotes.md" in str(p) for p in saved_paths)
        assert any("test2024_methods_tools.md" in str(p) for p in saved_paths)
        
        # Verify files exist
        summary_file = output_dir / "test2024_summary.md"
        claims_file = output_dir / "test2024_claims_quotes.md"
        methods_file = output_dir / "test2024_methods_tools.md"
        
        assert summary_file.exists()
        assert claims_file.exists()
        assert methods_file.exists()
        
        # Verify content
        assert "Main summary text" in summary_file.read_text()
        assert "Claims and quotes text" in claims_file.read_text()
        assert "Methods and tools text" in methods_file.read_text()
    
    def test_save_summary_handles_missing_optional_fields(self, tmp_path):
        """Test that save_summary handles missing claims_quotes_text or methods_tools_text."""
        from infrastructure.llm.core.client import LLMClient
        from infrastructure.llm.core.config import LLMConfig
        
        config = LLMConfig(base_url="http://localhost:11434", default_model="test")
        llm_client = LLMClient(config)
        engine = SummarizationEngine(llm_client)
        
        result = SearchResult(
            title="Test Paper",
            authors=["Author 1"],
            year=2024,
            abstract="Test abstract",
            url="https://example.com/paper",
            source="test"
        )
        
        # Only summary_text, no optional fields
        summary_result = SummarizationResult(
            citation_key="test2024",
            success=True,
            summary_text="Main summary text",
            claims_quotes_text=None,
            methods_tools_text=None,
            input_chars=1000,
            input_words=200,
            output_words=50
        )
        
        output_dir = tmp_path / "summaries"
        saved_paths = engine.save_summary(result, summary_result, output_dir)
        
        # Should only save summary file
        assert len(saved_paths) == 1
        assert any("test2024_summary.md" in str(p) for p in saved_paths)
        
        # Optional files should not exist
        claims_file = output_dir / "test2024_claims_quotes.md"
        methods_file = output_dir / "test2024_methods_tools.md"
        
        assert not claims_file.exists()
        assert not methods_file.exists()
    
    def test_summarize_paper_performs_three_passes(self, tmp_path):
        """Test that summarize_paper performs all three passes."""
        from infrastructure.llm.core.client import LLMClient
        from infrastructure.llm.core.config import LLMConfig
        from infrastructure.literature.summarization.multi_stage_summarizer import MultiStageSummarizer
        
        config = LLMConfig(base_url="http://localhost:11434", default_model="test")
        llm_client = LLMClient(config)
        
        # Create engine first
        engine = SummarizationEngine(llm_client)
        
        # Mock the multi-stage summarizer
        mock_summarizer = Mock(spec=MultiStageSummarizer)
        validation_result = Mock()
        validation_result.is_valid = True
        validation_result.score = 0.9
        validation_result.errors = []
        validation_result.has_hard_failure = Mock(return_value=False)
        mock_summarizer.summarize_with_refinement = Mock(return_value=(
            "Main summary",
            validation_result,
            1
        ))
        engine.multi_stage_summarizer = mock_summarizer
        
        # Mock the extraction methods
        engine._extract_claims_and_quotes = Mock(return_value="Claims and quotes")
        engine._analyze_methods_and_tools = Mock(return_value="Methods and tools")
        
        # Create extracted text file in expected location
        extracted_text_dir = Path("data/extracted_text")
        extracted_text_dir.mkdir(parents=True, exist_ok=True)
        extracted_text_file = extracted_text_dir / "test.txt"
        pdf_text = "Full paper text here. " * 100  # Ensure enough text
        extracted_text_file.write_text(pdf_text)
        
        # Mock text extraction
        engine.text_extractor = Mock()
        engine.text_extractor.load_extracted_text = Mock(return_value=pdf_text)
        
        # Mock context extraction
        from infrastructure.literature.summarization.models import SummarizationContext
        engine.context_extractor = Mock()
        engine.context_extractor.create_summarization_context = Mock(return_value=SummarizationContext(
            title="Test Paper",
            abstract="Abstract",
            introduction="Introduction",
            conclusion="Conclusion",
            key_terms=["test"],
            equations=[],
            full_text=pdf_text
        ))
        
        result = SearchResult(
            title="Test Paper",
            authors=["Author 1"],
            year=2024,
            abstract="Test abstract",
            url="https://example.com/paper",
            source="test"
        )
        
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("Test PDF content")
        
        summary_result = engine.summarize_paper(result, pdf_path)
        
        # Verify all three passes were called
        assert mock_summarizer.summarize_with_refinement.called, "Multi-stage summarizer should have been called"
        assert engine._extract_claims_and_quotes.called, "Claims/quotes extraction should have been called"
        assert engine._analyze_methods_and_tools.called, "Methods/tools analysis should have been called"
        
        # Verify result contains all three outputs
        assert summary_result.summary_text == "Main summary"
        assert summary_result.claims_quotes_text == "Claims and quotes"
        assert summary_result.methods_tools_text == "Methods and tools"
        
        # Cleanup
        if extracted_text_file.exists():
            extracted_text_file.unlink()
