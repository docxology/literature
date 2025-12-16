"""DOI enrichment for library entries.

Searches multiple sources (CrossRef, Semantic Scholar, OpenAlex) to find
DOIs for papers that don't have them, especially arXiv preprints that
may have been published later.
"""
from __future__ import annotations

import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.core.config import LiteratureConfig
from infrastructure.literature.library.index import LibraryIndex, LibraryEntry
from infrastructure.literature.sources import (
    CrossRefSource,
    SemanticScholarSource,
    OpenAlexSource,
    SearchResult,
)
from infrastructure.literature.sources.base import title_similarity

logger = get_logger(__name__)


@dataclass
class EnrichmentResult:
    """Result of DOI enrichment for a single entry.
    
    Attributes:
        citation_key: Citation key of the entry.
        found: Whether a DOI was found.
        doi: The DOI that was found (if any).
        source: Source that provided the DOI.
        match_score: Similarity score of the match.
        error: Error message if enrichment failed.
    """
    citation_key: str
    found: bool
    doi: Optional[str] = None
    source: Optional[str] = None
    match_score: float = 0.0
    error: Optional[str] = None


@dataclass
class EnrichmentStatistics:
    """Statistics for DOI enrichment operation.
    
    Attributes:
        total_processed: Total entries processed.
        found: Number of DOIs found.
        updated: Number of entries successfully updated.
        failed: Number of entries that failed.
        errors: List of error messages.
    """
    total_processed: int = 0
    found: int = 0
    updated: int = 0
    failed: int = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_processed": self.total_processed,
            "found": self.found,
            "updated": self.updated,
            "failed": self.failed,
            "errors": self.errors,
        }


class DOIEnrichment:
    """DOI enrichment coordinator.
    
    Searches multiple sources to find DOIs for papers missing them.
    Uses title/author/year matching to identify published versions of preprints.
    """
    
    # Matching thresholds
    TITLE_SIMILARITY_THRESHOLD = 0.85
    YEAR_TOLERANCE = 1  # ±1 year tolerance
    
    def __init__(
        self,
        config: LiteratureConfig,
        library_index: LibraryIndex,
        sources: Optional[Dict[str, Any]] = None
    ):
        """Initialize DOI enrichment.
        
        Args:
            config: Literature configuration.
            library_index: Library index to update.
            sources: Optional dict of source instances. If not provided,
                creates default sources (CrossRef, Semantic Scholar, OpenAlex).
        """
        self.config = config
        self.library_index = library_index
        
        # Initialize sources
        if sources:
            self.sources = sources
        else:
            self.sources = {
                "crossref": CrossRefSource(config),
                "semanticscholar": SemanticScholarSource(config),
                "openalex": OpenAlexSource(config),
            }
        
        # Source priority (order matters)
        self.source_priority = ["crossref", "semanticscholar", "openalex"]
    
    def enrich_entry(self, entry: LibraryEntry) -> EnrichmentResult:
        """Enrich a single library entry with DOI.
        
        Args:
            entry: Library entry to enrich.
            
        Returns:
            EnrichmentResult with enrichment status.
        """
        if entry.doi:
            logger.debug(f"Entry {entry.citation_key} already has DOI: {entry.doi}")
            return EnrichmentResult(
                citation_key=entry.citation_key,
                found=False,
                error="Entry already has DOI"
            )
        
        logger.info(f"Enriching DOI for: {entry.title[:60]}...")
        
        # Build search query from title and first author
        query = self._build_search_query(entry)
        
        # Search sources in priority order
        for source_name in self.source_priority:
            if source_name not in self.sources:
                continue
            
            source = self.sources[source_name]
            try:
                # Search with rate limiting
                results = source.search(query, limit=10)
                
                # Find best match
                match_result = self._find_best_match(entry, results)
                
                if match_result:
                    match, score = match_result
                    if match and match.doi:
                        logger.info(
                            f"Found DOI for {entry.citation_key}: {match.doi} "
                            f"(source: {source_name}, score: {score:.2f})"
                        )
                        return EnrichmentResult(
                            citation_key=entry.citation_key,
                            found=True,
                            doi=match.doi,
                            source=source_name,
                            match_score=score
                        )
                
                # Rate limiting delay between sources
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Error searching {source_name} for {entry.citation_key}: {e}")
                continue
        
        logger.debug(f"No DOI found for {entry.citation_key}")
        return EnrichmentResult(
            citation_key=entry.citation_key,
            found=False,
            error="No DOI found in any source"
        )
    
    def enrich_all(self, entries: Optional[List[LibraryEntry]] = None) -> EnrichmentStatistics:
        """Enrich multiple entries with DOIs.
        
        Args:
            entries: List of entries to enrich. If None, enriches all entries
                missing DOIs.
            
        Returns:
            EnrichmentStatistics with operation results.
        """
        if entries is None:
            entries = self.library_index.get_entries_missing_doi()
        
        stats = EnrichmentStatistics(total_processed=len(entries))
        
        logger.info(f"Starting DOI enrichment for {len(entries)} entries...")
        
        for i, entry in enumerate(entries, 1):
            logger.info(f"Processing {i}/{len(entries)}: {entry.citation_key}")
            
            try:
                result = self.enrich_entry(entry)
                
                if result.found and result.doi:
                    stats.found += 1
                    
                    # Update library entry
                    entry.doi = result.doi
                    self.library_index.update_entry(entry)
                    stats.updated += 1
                    
                    logger.info(f"✓ Updated {entry.citation_key} with DOI: {result.doi}")
                else:
                    stats.failed += 1
                    if result.error:
                        stats.errors.append(f"{entry.citation_key}: {result.error}")
                
            except Exception as e:
                stats.failed += 1
                error_msg = f"{entry.citation_key}: {str(e)}"
                stats.errors.append(error_msg)
                logger.error(f"Error enriching {entry.citation_key}: {e}")
            
            # Rate limiting between entries
            if i < len(entries):
                time.sleep(1.0)
        
        logger.info(
            f"DOI enrichment complete: {stats.found} found, "
            f"{stats.updated} updated, {stats.failed} failed"
        )
        
        return stats
    
    def _build_search_query(self, entry: LibraryEntry) -> str:
        """Build search query from entry metadata.
        
        Args:
            entry: Library entry.
            
        Returns:
            Search query string.
        """
        # Use title as primary query
        query = entry.title
        
        # Add first author if available
        if entry.authors:
            first_author = entry.authors[0]
            # Extract last name (handle "Last, First" or "First Last" formats)
            parts = first_author.replace(",", " ").split()
            if parts:
                last_name = parts[-1]
                query = f"{query} {last_name}"
        
        return query
    
    def _find_best_match(
        self,
        entry: LibraryEntry,
        results: List[SearchResult]
    ) -> Optional[Tuple[SearchResult, float]]:
        """Find best matching result for an entry.
        
        Args:
            entry: Library entry to match.
            results: Search results to match against.
            
        Returns:
            Tuple of (best_match, score) if found, None otherwise.
        """
        best_match = None
        best_score = 0.0
        
        for result in results:
            if not result.doi:
                continue
            
            # Calculate title similarity
            title_score = title_similarity(entry.title, result.title)
            
            if title_score < self.TITLE_SIMILARITY_THRESHOLD:
                continue
            
            # Check author overlap
            author_score = self._calculate_author_overlap(entry.authors, result.authors)
            
            # Check year match
            year_score = self._calculate_year_match(entry.year, result.year)
            
            # Combined score (weighted)
            combined_score = (
                title_score * 0.6 +  # Title is most important
                author_score * 0.3 +  # Authors are important
                year_score * 0.1      # Year is less important
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = result
        
        if best_match and best_score >= self.TITLE_SIMILARITY_THRESHOLD:
            return (best_match, best_score)
        
        return None
    
    def _calculate_author_overlap(
        self,
        entry_authors: List[str],
        result_authors: List[str]
    ) -> float:
        """Calculate author overlap score.
        
        Args:
            entry_authors: Authors from library entry.
            result_authors: Authors from search result.
            
        Returns:
            Score between 0.0 and 1.0.
        """
        if not entry_authors or not result_authors:
            return 0.0
        
        # Normalize author names (extract last names)
        def get_last_name(author: str) -> str:
            parts = author.replace(",", " ").split()
            return parts[-1].lower() if parts else ""
        
        entry_last_names = {get_last_name(a) for a in entry_authors}
        result_last_names = {get_last_name(a) for a in result_authors}
        
        if not entry_last_names or not result_last_names:
            return 0.0
        
        # At least one author must match
        overlap = len(entry_last_names & result_last_names)
        if overlap == 0:
            return 0.0
        
        # Score based on overlap ratio
        return min(overlap / len(entry_last_names), 1.0)
    
    def _calculate_year_match(
        self,
        entry_year: Optional[int],
        result_year: Optional[int]
    ) -> float:
        """Calculate year match score.
        
        Args:
            entry_year: Year from library entry.
            result_year: Year from search result.
            
        Returns:
            Score between 0.0 and 1.0.
        """
        if not entry_year or not result_year:
            return 0.5  # Neutral score if year missing
        
        year_diff = abs(entry_year - result_year)
        
        if year_diff == 0:
            return 1.0
        elif year_diff <= self.YEAR_TOLERANCE:
            return 0.8
        else:
            return max(0.0, 1.0 - (year_diff - self.YEAR_TOLERANCE) * 0.2)

