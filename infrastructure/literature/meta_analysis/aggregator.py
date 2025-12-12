"""Data aggregation for meta-analysis.

Collects and prepares data from library index and extracted texts
for temporal, keyword, metadata, and PCA analysis.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.library.index import LibraryIndex, LibraryEntry
from infrastructure.literature.core.config import LiteratureConfig

logger = get_logger(__name__)


@dataclass
class TemporalData:
    """Temporal aggregation of papers by year."""
    years: List[int]
    counts: List[int]
    papers_by_year: Dict[int, List[LibraryEntry]]
    total_papers: int
    year_range: Tuple[int, int]


@dataclass
class KeywordData:
    """Keyword extraction and aggregation."""
    keywords: List[str]
    keyword_counts: Dict[str, int]
    keywords_by_year: Dict[int, List[str]]
    keyword_frequency_over_time: Dict[str, List[Tuple[int, int]]]


@dataclass
class MetadataData:
    """Metadata aggregation for visualization."""
    venues: Dict[str, int]
    authors: Dict[str, int]
    sources: Dict[str, int]
    citation_counts: List[int]
    dois_available: int
    pdfs_available: int


@dataclass
class TextCorpus:
    """Text corpus for PCA and text analysis."""
    citation_keys: List[str]
    texts: List[str]
    titles: List[str]
    abstracts: List[str]
    years: List[Optional[int]]


class DataAggregator:
    """Aggregates library data for meta-analysis."""
    
    def __init__(self, config: Optional[LiteratureConfig] = None, default_entries: Optional[List[LibraryEntry]] = None):
        """Initialize aggregator.
        
        Args:
            config: Optional literature config (uses default if not provided).
            default_entries: Optional list of entries to use as default when entries=None.
        """
        if config is None:
            config = LiteratureConfig.from_env()
        self.config = config
        self.library_index = LibraryIndex(config)
        self._default_entries = default_entries
    
    def validate_data_quality(self, entries: Optional[List[LibraryEntry]] = None) -> Dict[str, Any]:
        """Validate data quality and return metrics.
        
        Args:
            entries: Optional list of entries (uses default or all if not provided).
            
        Returns:
            Dictionary with data quality metrics.
        """
        if entries is None:
            entries = self.aggregate_library_data()
        
        total = len(entries)
        if total == 0:
            return {
                'total': 0,
                'has_year': 0,
                'has_authors': 0,
                'has_abstract': 0,
                'has_doi': 0,
                'has_pdf': 0,
                'has_extracted_text': 0,
                'year_coverage': 0.0,
                'author_coverage': 0.0,
                'abstract_coverage': 0.0,
                'doi_coverage': 0.0,
                'pdf_coverage': 0.0,
                'extracted_text_coverage': 0.0,
                'sufficient_for_pca': False,
                'sufficient_for_keywords': False,
                'sufficient_for_temporal': False,
            }
        
        has_year = sum(1 for e in entries if e.year)
        has_authors = sum(1 for e in entries if e.authors)
        has_abstract = sum(1 for e in entries if hasattr(e, 'abstract') and e.abstract)
        has_doi = sum(1 for e in entries if e.doi)
        has_pdf = sum(1 for e in entries if e.pdf_path)
        
        # Check for extracted text
        extracted_text_dir = Path("data/extracted_text")
        has_extracted_text = sum(
            1 for e in entries 
            if (extracted_text_dir / f"{e.citation_key}.txt").exists()
        )
        
        return {
            'total': total,
            'has_year': has_year,
            'has_authors': has_authors,
            'has_abstract': has_abstract,
            'has_doi': has_doi,
            'has_pdf': has_pdf,
            'has_extracted_text': has_extracted_text,
            'year_coverage': (has_year / total * 100) if total > 0 else 0.0,
            'author_coverage': (has_authors / total * 100) if total > 0 else 0.0,
            'abstract_coverage': (has_abstract / total * 100) if total > 0 else 0.0,
            'doi_coverage': (has_doi / total * 100) if total > 0 else 0.0,
            'pdf_coverage': (has_pdf / total * 100) if total > 0 else 0.0,
            'extracted_text_coverage': (has_extracted_text / total * 100) if total > 0 else 0.0,
            'sufficient_for_pca': has_extracted_text >= 2,  # Need at least 2 papers for PCA
            'sufficient_for_keywords': has_abstract >= 1,  # Need at least 1 abstract
            'sufficient_for_temporal': has_year >= 1,  # Need at least 1 year
        }
    
    def aggregate_library_data(self) -> List[LibraryEntry]:
        """Collect all library entries.
        
        Returns:
            List of all library entries (or default entries if set).
        """
        if self._default_entries is not None:
            return self._default_entries
        return self.library_index.list_entries()
    
    def prepare_temporal_data(self, entries: Optional[List[LibraryEntry]] = None) -> TemporalData:
        """Prepare temporal data for year-based analysis.
        
        Args:
            entries: Optional list of entries (uses default or all if not provided).
            
        Returns:
            TemporalData with year-based aggregations.
        """
        if entries is None:
            entries = self.aggregate_library_data()
        
        papers_by_year: Dict[int, List[LibraryEntry]] = defaultdict(list)
        
        for entry in entries:
            if entry.year:
                papers_by_year[entry.year].append(entry)
        
        years = sorted(papers_by_year.keys())
        counts = [len(papers_by_year[year]) for year in years]
        
        year_range = (min(years), max(years)) if years else (0, 0)
        
        return TemporalData(
            years=years,
            counts=counts,
            papers_by_year=dict(papers_by_year),
            total_papers=len(entries),
            year_range=year_range,
        )
    
    def prepare_keyword_data(
        self,
        entries: Optional[List[LibraryEntry]] = None,
        extract_from_abstracts: bool = True
    ) -> KeywordData:
        """Prepare keyword data for analysis.
        
        Args:
            entries: Optional list of entries (uses default or all if not provided).
            extract_from_abstracts: Whether to extract keywords from abstracts.
            
        Returns:
            KeywordData with keyword information.
        """
        if entries is None:
            entries = self.aggregate_library_data()
        
        keyword_counts: Dict[str, int] = defaultdict(int)
        keywords_by_year: Dict[int, List[str]] = defaultdict(list)
        
        # Simple keyword extraction from titles and abstracts
        for entry in entries:
            # Extract from title
            title_words = self._extract_keywords(entry.title)
            for word in title_words:
                keyword_counts[word] += 1
                if entry.year:
                    keywords_by_year[entry.year].append(word)
            
            # Extract from abstract if available
            if extract_from_abstracts and hasattr(entry, 'abstract') and entry.abstract:
                abstract_words = self._extract_keywords(entry.abstract)
                for word in abstract_words:
                    keyword_counts[word] += 1
                    if entry.year:
                        keywords_by_year[entry.year].append(word)
        
        # Calculate frequency over time
        keyword_frequency_over_time: Dict[str, List[Tuple[int, int]]] = {}
        for keyword in keyword_counts.keys():
            frequency = []
            for year in sorted(keywords_by_year.keys()):
                count = keywords_by_year[year].count(keyword)
                if count > 0:
                    frequency.append((year, count))
            if frequency:
                keyword_frequency_over_time[keyword] = frequency
        
        return KeywordData(
            keywords=list(keyword_counts.keys()),
            keyword_counts=dict(keyword_counts),
            keywords_by_year=dict(keywords_by_year),
            keyword_frequency_over_time=keyword_frequency_over_time,
        )
    
    def _extract_keywords(self, text: str, min_length: int = 4) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Text to extract keywords from.
            min_length: Minimum keyword length.
            
        Returns:
            List of extracted keywords.
        """
        import re
        
        # Simple word extraction (can be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + r',}\b', text.lower())
        
        # Filter common stop words
        stop_words = {
            'this', 'that', 'these', 'those', 'with', 'from', 'into', 'during',
            'including', 'against', 'throughout', 'despite', 'towards', 'upon',
            'concerning', 'through', 'about', 'which', 'their', 'there', 'these',
            'would', 'could', 'should', 'might', 'may', 'will', 'shall', 'must',
            'have', 'has', 'had', 'been', 'being', 'were', 'was', 'are', 'is',
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
            'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        keywords = [w for w in words if w not in stop_words and len(w) >= min_length]
        return list(set(keywords))  # Remove duplicates
    
    def prepare_metadata_data(self, entries: Optional[List[LibraryEntry]] = None) -> MetadataData:
        """Prepare metadata for visualization.
        
        Args:
            entries: Optional list of entries (uses default or all if not provided).
            
        Returns:
            MetadataData with metadata aggregations.
        """
        if entries is None:
            entries = self.aggregate_library_data()
        
        venues: Dict[str, int] = defaultdict(int)
        authors: Dict[str, int] = defaultdict(int)
        sources: Dict[str, int] = defaultdict(int)
        citation_counts: List[int] = []
        dois_available = 0
        pdfs_available = 0
        
        for entry in entries:
            # Venues
            if entry.venue:
                venues[entry.venue] += 1
            
            # Authors
            for author in entry.authors:
                authors[author] += 1
            
            # Sources
            sources[entry.source] += 1
            
            # Citations
            if entry.citation_count is not None:
                citation_counts.append(entry.citation_count)
            
            # DOIs and PDFs
            if entry.doi:
                dois_available += 1
            if entry.pdf_path:
                pdfs_available += 1
        
        return MetadataData(
            venues=dict(venues),
            authors=dict(authors),
            sources=dict(sources),
            citation_counts=citation_counts,
            dois_available=dois_available,
            pdfs_available=pdfs_available,
        )
    
    def prepare_text_corpus(
        self,
        entries: Optional[List[LibraryEntry]] = None,
        extracted_text_dir: Optional[Path] = None
    ) -> TextCorpus:
        """Prepare text corpus for PCA analysis.
        
        Args:
            entries: Optional list of entries (uses default or all if not provided).
            extracted_text_dir: Directory containing extracted text files.
            
        Returns:
            TextCorpus with texts for analysis.
        """
        if entries is None:
            entries = self.aggregate_library_data()
        
        logger.debug(f"Preparing text corpus from {len(entries)} entries...")
        
        if extracted_text_dir is None:
            extracted_text_dir = Path("data/extracted_text")
        
        citation_keys: List[str] = []
        texts: List[str] = []
        titles: List[str] = []
        abstracts: List[str] = []
        years: List[Optional[int]] = []
        
        extracted_count = 0
        abstract_fallback_count = 0
        empty_count = 0
        
        for entry in entries:
            citation_keys.append(entry.citation_key)
            titles.append(entry.title)
            abstract = entry.abstract if hasattr(entry, 'abstract') else ""
            abstracts.append(abstract)
            years.append(entry.year)
            
            # Try to load extracted text
            text_file = extracted_text_dir / f"{entry.citation_key}.txt"
            if text_file.exists():
                try:
                    text_content = text_file.read_text(encoding='utf-8')
                    texts.append(text_content)
                    extracted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to read {text_file}: {e}")
                    texts.append(abstract)
                    abstract_fallback_count += 1
            else:
                # Fallback to abstract if no extracted text
                texts.append(abstract)
                if abstract:
                    abstract_fallback_count += 1
                else:
                    empty_count += 1
        
        logger.info(f"Text corpus prepared: {len(entries)} entries, "
                   f"{extracted_count} with extracted text, "
                   f"{abstract_fallback_count} using abstracts, "
                   f"{empty_count} empty")
        
        return TextCorpus(
            citation_keys=citation_keys,
            texts=texts,
            titles=titles,
            abstracts=abstracts,
            years=years,
        )


