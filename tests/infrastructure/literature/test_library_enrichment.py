"""Tests for DOI enrichment functionality."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from infrastructure.literature.library.enrichment import (
    DOIEnrichment,
    EnrichmentResult,
    EnrichmentStatistics,
)
from infrastructure.literature.library.index import LibraryEntry, LibraryIndex
from infrastructure.literature.core.config import LiteratureConfig
from infrastructure.literature.sources import SearchResult


class TestEnrichmentResult:
    """Test suite for EnrichmentResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = EnrichmentResult(
            citation_key="test2024paper",
            found=True,
            doi="10.1234/test.doi",
            source="crossref",
            match_score=0.95
        )
        assert result.citation_key == "test2024paper"
        assert result.found is True
        assert result.doi == "10.1234/test.doi"
        assert result.source == "crossref"
        assert result.match_score == 0.95

    def test_result_not_found(self):
        """Test result for not found case."""
        result = EnrichmentResult(
            citation_key="test2024paper",
            found=False,
            error="No DOI found"
        )
        assert result.found is False
        assert result.doi is None
        assert result.error == "No DOI found"


class TestEnrichmentStatistics:
    """Test suite for EnrichmentStatistics dataclass."""

    def test_stats_creation(self):
        """Test basic stats creation."""
        stats = EnrichmentStatistics(
            total_processed=10,
            found=5,
            updated=5,
            failed=0
        )
        assert stats.total_processed == 10
        assert stats.found == 5
        assert stats.updated == 5
        assert stats.failed == 0

    def test_stats_to_dict(self):
        """Test conversion to dictionary."""
        stats = EnrichmentStatistics(
            total_processed=10,
            found=5,
            updated=5,
            failed=0,
            errors=["Error 1", "Error 2"]
        )
        stats_dict = stats.to_dict()
        assert stats_dict["total_processed"] == 10
        assert stats_dict["found"] == 5
        assert stats_dict["updated"] == 5
        assert stats_dict["failed"] == 0
        assert len(stats_dict["errors"]) == 2


class TestDOIEnrichment:
    """Test suite for DOIEnrichment class."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a test configuration."""
        config = LiteratureConfig()
        config.library_index_file = str(tmp_path / "library.json")
        config.bibtex_file = str(tmp_path / "references.bib")
        return config

    @pytest.fixture
    def library_index(self, config):
        """Create a test library index."""
        return LibraryIndex(config)

    @pytest.fixture
    def enrichment(self, config, library_index):
        """Create a DOIEnrichment instance."""
        return DOIEnrichment(config, library_index)

    def test_enrich_entry_already_has_doi(self, enrichment, library_index):
        """Test that entries with DOIs are skipped."""
        entry = LibraryEntry(
            citation_key="test2024paper",
            title="Test Paper",
            authors=["John Smith"],
            year=2024,
            doi="10.1234/existing.doi"
        )
        library_index._entries[entry.citation_key] = entry
        
        result = enrichment.enrich_entry(entry)
        
        assert result.found is False
        assert result.error == "Entry already has DOI"

    def test_build_search_query(self, enrichment):
        """Test search query building."""
        entry = LibraryEntry(
            citation_key="test2024paper",
            title="Machine Learning Advances",
            authors=["John Smith", "Jane Doe"],
            year=2024
        )
        
        query = enrichment._build_search_query(entry)
        
        assert "Machine Learning Advances" in query
        assert "Smith" in query

    def test_calculate_author_overlap(self, enrichment):
        """Test author overlap calculation."""
        entry_authors = ["John Smith", "Jane Doe"]
        result_authors = ["John Smith", "Bob Johnson"]
        
        score = enrichment._calculate_author_overlap(entry_authors, result_authors)
        
        assert score > 0.0
        assert score <= 1.0

    def test_calculate_author_overlap_no_match(self, enrichment):
        """Test author overlap with no matches."""
        entry_authors = ["John Smith"]
        result_authors = ["Bob Johnson"]
        
        score = enrichment._calculate_author_overlap(entry_authors, result_authors)
        
        assert score == 0.0

    def test_calculate_year_match(self, enrichment):
        """Test year match calculation."""
        score = enrichment._calculate_year_match(2024, 2024)
        assert score == 1.0
        
        score = enrichment._calculate_year_match(2024, 2025)
        assert score > 0.0  # Within tolerance
        
        score = enrichment._calculate_year_match(2024, 2020)
        assert score < 1.0  # Outside tolerance

    def test_calculate_year_match_missing(self, enrichment):
        """Test year match with missing years."""
        score = enrichment._calculate_year_match(None, 2024)
        assert score == 0.5  # Neutral score
        
        score = enrichment._calculate_year_match(2024, None)
        assert score == 0.5  # Neutral score

    @patch('infrastructure.literature.library.enrichment.time.sleep')
    def test_enrich_entry_finds_doi(self, mock_sleep, enrichment, library_index):
        """Test enriching an entry that finds a DOI."""
        entry = LibraryEntry(
            citation_key="test2024paper",
            title="Machine Learning Advances",
            authors=["John Smith"],
            year=2024
        )
        library_index._entries[entry.citation_key] = entry
        
        # Mock search result
        mock_result = SearchResult(
            title="Machine Learning Advances",
            authors=["John Smith"],
            year=2024,
            abstract="",
            url="",
            doi="10.1234/test.doi",
            source="crossref"
        )
        
        # Mock source
        mock_source = Mock()
        mock_source.search.return_value = [mock_result]
        enrichment.sources = {"crossref": mock_source}
        
        result = enrichment.enrich_entry(entry)
        
        assert result.found is True
        assert result.doi == "10.1234/test.doi"
        assert result.source == "crossref"

    @patch('infrastructure.literature.library.enrichment.time.sleep')
    def test_enrich_entry_no_match(self, mock_sleep, enrichment, library_index):
        """Test enriching an entry with no match."""
        entry = LibraryEntry(
            citation_key="test2024paper",
            title="Unique Paper Title",
            authors=["John Smith"],
            year=2024
        )
        library_index._entries[entry.citation_key] = entry
        
        # Mock search result with low similarity
        mock_result = SearchResult(
            title="Completely Different Title",
            authors=["Bob Johnson"],
            year=2020,
            abstract="",
            url="",
            doi="10.1234/other.doi",
            source="crossref"
        )
        
        # Mock source
        mock_source = Mock()
        mock_source.search.return_value = [mock_result]
        enrichment.sources = {"crossref": mock_source}
        
        result = enrichment.enrich_entry(entry)
        
        assert result.found is False

    def test_find_best_match(self, enrichment):
        """Test finding best match from results."""
        entry = LibraryEntry(
            citation_key="test2024paper",
            title="Machine Learning Advances",
            authors=["John Smith"],
            year=2024
        )
        
        results = [
            SearchResult(
                title="Machine Learning Advances",
                authors=["John Smith"],
                year=2024,
                abstract="",
                url="",
                doi="10.1234/good.match",
                source="crossref"
            ),
            SearchResult(
                title="Different Title",
                authors=["Bob Johnson"],
                year=2020,
                abstract="",
                url="",
                doi="10.1234/bad.match",
                source="crossref"
            ),
        ]
        
        match_result = enrichment._find_best_match(entry, results)
        
        assert match_result is not None
        match, score = match_result
        assert match.doi == "10.1234/good.match"
        assert score > 0.8

    def test_find_best_match_no_doi(self, enrichment):
        """Test that results without DOIs are skipped."""
        entry = LibraryEntry(
            citation_key="test2024paper",
            title="Machine Learning Advances",
            authors=["John Smith"],
            year=2024
        )
        
        results = [
            SearchResult(
                title="Machine Learning Advances",
                authors=["John Smith"],
                year=2024,
                abstract="",
                url="",
                doi=None,  # No DOI
                source="crossref"
            ),
        ]
        
        match_result = enrichment._find_best_match(entry, results)
        
        assert match_result is None


