"""Reference management for literature search."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from infrastructure.core.exceptions import FileOperationError
from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.core.config import LiteratureConfig
from infrastructure.literature.sources import SearchResult

if TYPE_CHECKING:
    from infrastructure.literature.library_index import LibraryIndex

logger = get_logger(__name__)


class ReferenceManager:
    """Manages references and BibTeX generation.
    
    Coordinates with LibraryIndex for consistent citation key generation
    and comprehensive paper tracking.
    """

    def __init__(self, config: LiteratureConfig, library_index: Optional["LibraryIndex"] = None):
        """Initialize reference manager.
        
        Args:
            config: Literature configuration.
            library_index: Optional LibraryIndex for coordinated key generation.
        """
        self.config = config
        self._library_index = library_index

    def set_library_index(self, library_index: "LibraryIndex") -> None:
        """Set the library index for coordinated operations.
        
        Args:
            library_index: LibraryIndex instance to use.
        """
        self._library_index = library_index

    def add_reference(self, result: SearchResult) -> str:
        """Add paper to references.
        
        Uses LibraryIndex for consistent key generation if available,
        otherwise generates key locally.
        
        Args:
            result: Search result to add.
            
        Returns:
            BibTeX citation key.
        """
        # Use library index for consistent key generation if available
        if self._library_index:
            key = self._library_index.add_entry(
                title=result.title,
                authors=result.authors,
                year=result.year,
                doi=result.doi,
                source=result.source,
                url=result.url,
                abstract=result.abstract,
                venue=result.venue,
                citation_count=result.citation_count,
                pdf_url=result.pdf_url
            )
        else:
            key = self._generate_key(result)
        
        bib_entry = self._format_bibtex(result, key)
        self._append_to_bibtex(bib_entry)
        return key

    def _generate_key(self, result: SearchResult) -> str:
        """Generate BibTeX key (authorYYYYword).
        
        Uses same logic as LibraryIndex.generate_citation_key for consistency.
        
        Args:
            result: Search result.
            
        Returns:
            Citation key string.
        """
        # Get first author's last name
        if result.authors:
            first_author = result.authors[0]
            parts = first_author.replace(",", " ").split()
            author = parts[-1].lower() if parts else "anonymous"
        else:
            author = "anonymous"
        
        # Sanitize
        author = "".join(c for c in author if c.isalnum())
        
        year = str(result.year) if result.year else "nodate"
        
        # First significant word from title
        title_words = result.title.lower().split()
        skip_words = {"a", "an", "the", "on", "in", "of", "for", "to", "and", "with"}
        title_word = "paper"
        for word in title_words:
            clean_word = "".join(c for c in word if c.isalnum())
            if clean_word and clean_word not in skip_words:
                title_word = clean_word
                break
        
        return f"{author}{year}{title_word}"

    def _format_bibtex(self, result: SearchResult, key: str) -> str:
        """Format result as BibTeX entry.
        
        Args:
            result: Search result.
            key: Citation key.
            
        Returns:
            BibTeX entry string.
        """
        entry_type = "article"
        
        fields = [
            f"  title={{{result.title}}}",
            f"  author={{{' and '.join(result.authors)}}}",
            f"  year={{{result.year}}}" if result.year else None,
            f"  url={{{result.url}}}",
            f"  abstract={{{result.abstract}}}" if result.abstract else None,
            f"  doi={{{result.doi}}}" if result.doi else None,
            f"  journal={{{result.venue}}}" if result.venue else None
        ]
        
        fields_str = ",\n".join(f for f in fields if f)
        return f"@{entry_type}{{{key},\n{fields_str}\n}}\n"

    def _append_to_bibtex(self, entry: str) -> None:
        """Append entry to BibTeX file.
        
        Args:
            entry: BibTeX entry string.
        """
        path = Path(self.config.bibtex_file)
        
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if duplicate (by key)
            if path.exists():
                content = path.read_text()
                # Extract key from entry
                key = entry.split(',')[0].split('{')[1]
                if key in content:
                    logger.debug(f"Reference {key} already exists in BibTeX (skipping)")
                    return

            with open(path, 'a') as f:
                f.write(entry + "\n")
            logger.info(f"Added reference to {path}")
                
        except OSError as e:
            raise FileOperationError(
                f"Failed to update BibTeX file: {e}",
                context={"path": str(path)}
            )

    def export_library(self, path: Optional[Path] = None) -> Path:
        """Export the library to JSON format.
        
        Delegates to LibraryIndex if available.
        
        Args:
            path: Output path. Uses library index default if not specified.
            
        Returns:
            Path to exported file.
            
        Raises:
            RuntimeError: If no library index is configured.
        """
        if not self._library_index:
            raise RuntimeError("No library index configured for export")
        
        return self._library_index.export_json(path)

    def update_reference_doi(self, citation_key: str, doi: str) -> bool:
        """Update DOI field for an existing BibTeX entry.
        
        Args:
            citation_key: Citation key of the entry to update.
            doi: DOI to add or update.
            
        Returns:
            True if entry was found and updated, False otherwise.
        """
        path = Path(self.config.bibtex_file)
        
        if not path.exists():
            logger.warning(f"BibTeX file not found: {path}")
            return False
        
        try:
            content = path.read_text(encoding='utf-8')
            
            # Find the entry by citation key
            # Pattern: @type{citation_key,
            pattern = rf'@\w+\{{{re.escape(citation_key)}\s*,'
            match = re.search(pattern, content)
            
            if not match:
                logger.debug(f"BibTeX entry not found for citation key: {citation_key}")
                return False
            
            # Find the end of this entry (next @ or end of file)
            start_pos = match.start()
            # Find the closing brace for this entry
            brace_count = 0
            in_entry = False
            end_pos = start_pos
            
            for i in range(start_pos, len(content)):
                if content[i] == '{':
                    brace_count += 1
                    in_entry = True
                elif content[i] == '}':
                    brace_count -= 1
                    if in_entry and brace_count == 0:
                        end_pos = i + 1
                        break
            
            entry_text = content[start_pos:end_pos]
            
            # Check if DOI already exists
            doi_pattern = r'\s*doi\s*=\s*\{[^}]+\},?\s*\n?'
            if re.search(doi_pattern, entry_text, re.IGNORECASE):
                # Replace existing DOI
                new_entry_text = re.sub(
                    doi_pattern,
                    f'  doi={{{doi}}},\n',
                    entry_text,
                    flags=re.IGNORECASE
                )
            else:
                # Add DOI before the closing brace
                # Find the last field before closing brace
                last_field_match = re.search(r'(\s+[^,\s]+=\{[^}]+\},?\s*\n?)(?=\s*\})', entry_text)
                if last_field_match:
                    insert_pos = last_field_match.end()
                    new_entry_text = (
                        entry_text[:insert_pos] +
                        f'  doi={{{doi}}},\n' +
                        entry_text[insert_pos:]
                    )
                else:
                    # Fallback: insert before closing brace
                    new_entry_text = entry_text.rstrip()[:-1] + f',\n  doi={{{doi}}}\n}}\n'
            
            # Replace entry in content
            new_content = content[:start_pos] + new_entry_text + content[end_pos:]
            
            # Write back to file
            path.write_text(new_content, encoding='utf-8')
            logger.info(f"Updated DOI for {citation_key} in BibTeX file")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update BibTeX DOI for {citation_key}: {e}")
            raise FileOperationError(
                f"Failed to update BibTeX DOI: {e}",
                context={"path": str(path), "citation_key": citation_key}
            )
