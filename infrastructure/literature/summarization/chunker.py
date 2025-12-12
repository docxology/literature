"""PDF text chunking for two-stage summarization.

This module provides intelligent chunking of PDF text that preserves
section boundaries and prioritizes key sections for efficient two-stage
summarization.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TextChunk:
    """A chunk of text with metadata.
    
    Attributes:
        text: The chunk text content.
        start_pos: Starting character position in original text.
        end_pos: Ending character position in original text.
        section_name: Name of section this chunk belongs to (if known).
        chunk_index: Index of this chunk (0-based).
        is_prioritized: Whether this chunk contains prioritized sections.
    """
    text: str
    start_pos: int
    end_pos: int
    section_name: Optional[str] = None
    chunk_index: int = 0
    is_prioritized: bool = False


@dataclass
class ChunkingResult:
    """Result of PDF text chunking.
    
    Attributes:
        chunks: List of text chunks.
        total_chunks: Total number of chunks created.
        original_length: Original text length in characters.
        average_chunk_size: Average chunk size in characters.
        prioritized_chunks: Number of chunks containing prioritized sections.
    """
    chunks: List[TextChunk]
    total_chunks: int
    original_length: int
    average_chunk_size: float
    prioritized_chunks: int = 0


class PDFChunker:
    """Intelligent PDF text chunking with section awareness.
    
    Chunks PDF text while preserving section boundaries and prioritizing
    key sections (abstract, introduction, conclusion) for efficient
    two-stage summarization.
    """
    
    # Section header patterns (case-insensitive, flexible matching)
    SECTION_PATTERNS = {
        'abstract': [
            r'^\s*(?:Abstract|Summary|Synopsis)\s*$',
            r'^\s*ABSTRACT\s*$',
        ],
        'introduction': [
            r'^\s*(?:1\.?\s+)?Introduction\s*$',
            r'^\s*INTRODUCTION\s*$',
            r'^\s*(?:1\.?\s+)?Background\s*$',
        ],
        'conclusion': [
            r'^\s*(?:Conclusion|Conclusions|Discussion and Conclusion)\s*$',
            r'^\s*CONCLUSION\s*$',
            r'^\s*(?:Discussion|Summary and Discussion)\s*$',
        ],
        'methods': [
            r'^\s*(?:2\.?\s+)?(?:Methods?|Methodology|Experimental\s+Setup)\s*$',
            r'^\s*METHODS?\s*$',
        ],
        'results': [
            r'^\s*(?:3\.?\s+)?(?:Results?|Findings|Experimental\s+Results)\s*$',
            r'^\s*RESULTS?\s*$',
        ],
        'discussion': [
            r'^\s*(?:4\.?\s+)?(?:Discussion|Analysis)\s*$',
            r'^\s*DISCUSSION\s*$',
        ],
    }
    
    # Priority order for sections (higher priority preserved first)
    SECTION_PRIORITY = ['abstract', 'introduction', 'conclusion', 'methods', 'results', 'discussion']
    
    def __init__(
        self,
        target_chunk_size: int = 15000,
        chunk_overlap: int = 500,
        min_chunk_size: int = 1000
    ):
        """Initialize PDF chunker.
        
        Args:
            target_chunk_size: Target size for each chunk in characters (default: 15000).
            chunk_overlap: Overlap between chunks in characters (default: 500).
            min_chunk_size: Minimum chunk size to avoid tiny chunks (default: 1000).
        """
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def _search_section_pattern(self, pattern: str, text: str) -> Optional[re.Match]:
        """Search for section pattern in both original and normalized text.
        
        Args:
            pattern: Regex pattern to search for.
            text: Text to search in.
            
        Returns:
            Match object if found, None otherwise.
        """
        # Try original text first
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match
        
        # Try normalized text (strip leading whitespace from each line)
        normalized_lines = [line.strip() for line in text.split('\n')]
        normalized_text = '\n'.join(normalized_lines)
        match = re.search(pattern, normalized_text, re.IGNORECASE | re.MULTILINE)
        
        if match:
            # Map normalized position back to original text
            norm_pos = match.start()
            norm_line_idx = normalized_text[:norm_pos].count('\n')
            orig_lines = text.split('\n')
            
            if norm_line_idx < len(normalized_lines) and norm_line_idx < len(orig_lines):
                # Calculate position in original text
                # Sum lengths of all lines before the matched line, plus newlines
                orig_pos = sum(len(line) + 1 for line in orig_lines[:norm_line_idx])
                if norm_line_idx > 0:
                    orig_pos -= 1  # Adjust for last newline
                
                # Find the matched line in original and normalized
                matched_line_norm = normalized_lines[norm_line_idx]
                matched_line_orig = orig_lines[norm_line_idx]
                
                # Find offset within the line
                line_start_in_norm = sum(len(nl) + 1 for nl in normalized_lines[:norm_line_idx])
                if norm_line_idx > 0:
                    line_start_in_norm -= 1
                offset_in_line = norm_pos - line_start_in_norm
                
                # Find where the normalized content starts in the original line
                orig_line_stripped = matched_line_orig.lstrip()
                leading_whitespace = len(matched_line_orig) - len(orig_line_stripped)
                orig_line_start = orig_pos + leading_whitespace
                
                # Calculate final position
                final_pos = orig_line_start + offset_in_line
                
                # Create a match-like object with correct positions
                class MappedMatch:
                    def __init__(self, start_pos, end_pos, matched_text):
                        self._start = start_pos
                        self._end = end_pos
                        self._text = matched_text
                    
                    def start(self):
                        return self._start
                    
                    def end(self):
                        return self._end
                    
                    def group(self):
                        return self._text
                
                return MappedMatch(final_pos, final_pos + len(match.group()), match.group())
        
        return None
    
    def identify_sections(self, text: str) -> Dict[str, Tuple[int, int]]:
        """Identify key sections in PDF text.
        
        Args:
            text: Full PDF text content.
            
        Returns:
            Dictionary mapping section names to (start_char, end_char) positions.
        """
        sections: dict[str, Tuple[int, int]] = {}
        
        # Find abstract
        for pattern in self.SECTION_PATTERNS['abstract']:
            match = self._search_section_pattern(pattern, text)
            if match:
                start = match.end()
                # Abstract typically ends before Introduction or next major section
                next_section = re.search(
                    r'^\s*(?:1\.?\s+)?(?:Introduction|Background|Keywords|1\.)\s*$',
                    text[start:],
                    re.IGNORECASE | re.MULTILINE
                )
                if next_section:
                    end = start + next_section.start()
                else:
                    end = min(start + 2000, len(text))
                sections['abstract'] = (start, end)
                break
        
        # Find introduction
        for pattern in self.SECTION_PATTERNS['introduction']:
            match = self._search_section_pattern(pattern, text)
            if match:
                start = match.end()
                # Introduction typically 2000-5000 words, take first 5000 chars
                end = min(start + 5000, len(text))
                sections['introduction'] = (start, end)
                break
        
        # Find conclusion (search from end)
        for pattern in self.SECTION_PATTERNS['conclusion']:
            # Try original text first
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if not matches:
                # Try normalized text
                normalized_lines = [line.strip() for line in text.split('\n')]
                normalized_text = '\n'.join(normalized_lines)
                norm_matches = list(re.finditer(pattern, normalized_text, re.IGNORECASE | re.MULTILINE))
                if norm_matches:
                    # Use the last match and map back to original
                    norm_match = norm_matches[-1]
                    norm_pos = norm_match.start()
                    norm_line_idx = normalized_text[:norm_pos].count('\n')
                    orig_lines = text.split('\n')
                    if norm_line_idx < len(orig_lines):
                        orig_pos = sum(len(line) + 1 for line in orig_lines[:norm_line_idx]) - 1
                        if orig_pos < 0:
                            orig_pos = 0
                        # Create a match-like object
                        class MappedMatch:
                            def __init__(self, start_pos, end_pos):
                                self._start = start_pos
                                self._end = end_pos
                            def start(self):
                                return self._start
                            def end(self):
                                return self._end
                        matches = [MappedMatch(orig_pos, orig_pos + len(norm_match.group()))]
            
            if matches:
                match = matches[-1]  # Take the last match
                start = match.end()
                end = len(text)
                # Limit to reasonable size
                if end - start > 3000:
                    start = end - 3000
                sections['conclusion'] = (start, end)
                break
        
        # Find methods
        for pattern in self.SECTION_PATTERNS['methods']:
            match = self._search_section_pattern(pattern, text)
            if match:
                start = match.end()
                # Methods section typically ends at Results
                next_section = re.search(
                    r'^\s*(?:3\.?\s+)?(?:Results?|Findings)\s*$',
                    text[start:],
                    re.IGNORECASE | re.MULTILINE
                )
                if next_section:
                    end = start + next_section.start()
                else:
                    end = min(start + 10000, len(text))
                sections['methods'] = (start, end)
                break
        
        # Find results
        for pattern in self.SECTION_PATTERNS['results']:
            match = self._search_section_pattern(pattern, text)
            if match:
                start = match.end()
                # Results typically ends at Discussion
                next_section = re.search(
                    r'^\s*(?:4\.?\s+)?(?:Discussion|Conclusion)\s*$',
                    text[start:],
                    re.IGNORECASE | re.MULTILINE
                )
                if next_section:
                    end = start + next_section.start()
                else:
                    end = min(start + 10000, len(text))
                sections['results'] = (start, end)
                break
        
        # Find discussion
        for pattern in self.SECTION_PATTERNS['discussion']:
            match = self._search_section_pattern(pattern, text)
            if match:
                start = match.end()
                # Discussion typically ends at Conclusion
                next_section = re.search(
                    r'^\s*(?:Conclusion|Conclusions)\s*$',
                    text[start:],
                    re.IGNORECASE | re.MULTILINE
                )
                if next_section:
                    end = start + next_section.start()
                else:
                    end = min(start + 10000, len(text))
                sections['discussion'] = (start, end)
                break
        
        return sections
    
    def chunk_text(
        self,
        text: str,
        preserve_sections: bool = True
    ) -> ChunkingResult:
        """Chunk PDF text with section awareness.
        
        Args:
            text: Full PDF text to chunk.
            preserve_sections: Whether to preserve section boundaries (default: True).
            
        Returns:
            ChunkingResult with chunks and metadata.
        """
        original_length = len(text)
        
        # If text is small enough, return single chunk
        if original_length <= self.target_chunk_size:
            return ChunkingResult(
                chunks=[
                    TextChunk(
                        text=text,
                        start_pos=0,
                        end_pos=original_length,
                        chunk_index=0,
                        is_prioritized=True
                    )
                ],
                total_chunks=1,
                original_length=original_length,
                average_chunk_size=float(original_length),
                prioritized_chunks=1
            )
        
        chunks: List[TextChunk] = []
        
        if preserve_sections:
            # Identify sections
            sections = self.identify_sections(text)
            
            # If sections found, use section-aware chunking
            # Otherwise fall back to simple chunking
            if sections:
                chunks = self._chunk_with_sections(text, sections)
            else:
                # No sections found - fall back to simple chunking
                chunks = self._chunk_simple(text)
        else:
            # Simple chunking without section awareness
            chunks = self._chunk_simple(text)
        
        # Calculate statistics
        total_chunks = len(chunks)
        avg_size = sum(len(c.text) for c in chunks) / max(total_chunks, 1)
        prioritized = sum(1 for c in chunks if c.is_prioritized)
        
        return ChunkingResult(
            chunks=chunks,
            total_chunks=total_chunks,
            original_length=original_length,
            average_chunk_size=avg_size,
            prioritized_chunks=prioritized
        )
    
    def _chunk_with_sections(
        self,
        text: str,
        sections: Dict[str, Tuple[int, int]]
    ) -> List[TextChunk]:
        """Chunk text while preserving section boundaries.
        
        Args:
            text: Full text to chunk.
            sections: Dictionary of identified sections.
            
        Returns:
            List of TextChunk objects.
        """
        chunks: List[TextChunk] = []
        chunk_index = 0
        
        # Prioritized sections (abstract, intro, conclusion)
        prioritized_sections = ['abstract', 'introduction', 'conclusion']
        
        # Sort sections by position
        sorted_sections = sorted(sections.items(), key=lambda x: x[1][0])
        
        # First chunk: prioritize abstract, intro, conclusion
        first_chunk_parts = []
        first_chunk_start = 0
        first_chunk_size = 0
        
        for section_name in prioritized_sections:
            if section_name in sections:
                start, end = sections[section_name]
                section_text = text[start:end]
                if first_chunk_size + len(section_text) <= self.target_chunk_size:
                    first_chunk_parts.append((start, end, section_name))
                    first_chunk_size += len(section_text)
        
        # Build first chunk
        if first_chunk_parts:
            chunk_text_parts = []
            last_end = 0
            for start, end, section_name in first_chunk_parts:
                if start > last_end:
                    # Add gap marker
                    gap_size = start - last_end
                    if gap_size > 50:
                        chunk_text_parts.append(f"\n[... {gap_size} characters omitted ...]\n")
                chunk_text_parts.append(text[start:end])
                last_end = max(last_end, end)
            
            first_chunk_text = ''.join(chunk_text_parts)
            chunks.append(TextChunk(
                text=first_chunk_text,
                start_pos=first_chunk_parts[0][0] if first_chunk_parts else 0,
                end_pos=first_chunk_parts[-1][1] if first_chunk_parts else len(first_chunk_text),
                section_name=first_chunk_parts[0][2] if first_chunk_parts else None,
                chunk_index=chunk_index,
                is_prioritized=True
            ))
            chunk_index += 1
        
        # Safety: If no chunks created yet but sections exist, create at least one chunk
        if not chunks and sections:
            # Create chunk from first section found (sorted_sections already defined above)
            first_section_name, (first_start, first_end) = sorted_sections[0]
            section_text = text[first_start:first_end]
            if len(section_text.strip()) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    text=section_text,
                    start_pos=first_start,
                    end_pos=first_end,
                    section_name=first_section_name,
                    chunk_index=0,
                    is_prioritized=first_section_name in prioritized_sections
                ))
                chunk_index = 1
            else:
                # Section too small, fall back to simple chunking
                chunks = self._chunk_simple(text)
                return chunks
        
        # Remaining chunks: process rest of text
        if sorted_sections:
            last_processed = sorted_sections[-1][1][1]  # End of last section
        else:
            last_processed = 0
        
        # Determine starting position for remaining chunks
        if first_chunk_parts:
            # Start after the last prioritized section
            current_pos = max(last_processed, first_chunk_parts[-1][1])
        elif chunks:
            # We created a chunk from sections, start after it
            current_pos = chunks[-1].end_pos
        elif sorted_sections:
            # No prioritized sections found, but we have other sections - start from first section
            current_pos = sorted_sections[0][1][1]  # End of first section
        else:
            # No sections found at all - fall back to simple chunking from start
            current_pos = 0
        
        while current_pos < len(text):
            chunk_end = min(current_pos + self.target_chunk_size, len(text))
            
            # Try to end at a sentence boundary
            if chunk_end < len(text):
                # Look for sentence ending within last 200 chars
                sentence_end = max(
                    chunk_end - 200,
                    text.rfind('.', current_pos, chunk_end),
                    text.rfind('\n', current_pos, chunk_end)
                )
                if sentence_end > current_pos:
                    chunk_end = sentence_end + 1
            
            chunk_text = text[current_pos:chunk_end]
            
            # Add overlap from previous chunk
            if chunks and self.chunk_overlap > 0:
                overlap_start = max(0, current_pos - self.chunk_overlap)
                overlap_text = text[overlap_start:current_pos]
                chunk_text = overlap_text + chunk_text
            
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_pos=current_pos - self.chunk_overlap if chunks and self.chunk_overlap > 0 else current_pos,
                    end_pos=chunk_end,
                    chunk_index=chunk_index,
                    is_prioritized=False
                ))
                chunk_index += 1
            
            # Move to next chunk position (with overlap)
            next_pos = chunk_end - self.chunk_overlap if self.chunk_overlap > 0 else chunk_end
            
            # Avoid infinite loop: ensure we always advance by at least 1
            if next_pos <= current_pos:
                next_pos = current_pos + 1
            
            # Also ensure we don't go backwards beyond previous chunk
            if chunks and next_pos <= chunks[-1].start_pos:
                next_pos = chunk_end
            
            current_pos = next_pos
            
            # Final safety check: if we haven't advanced, force advancement
            if current_pos >= len(text):
                break
        
        # Final safety: ensure at least one chunk exists
        if not chunks:
            # Fall back to simple chunking
            chunks = self._chunk_simple(text)
        
        return chunks
    
    def _chunk_simple(self, text: str) -> List[TextChunk]:
        """Simple chunking without section awareness.
        
        Args:
            text: Full text to chunk.
            
        Returns:
            List of TextChunk objects.
        """
        chunks: List[TextChunk] = []
        chunk_index = 0
        current_pos = 0
        last_pos = -1  # Track last position to detect infinite loops
        
        while current_pos < len(text):
            # Safety check: if we haven't advanced, force advancement
            if current_pos <= last_pos:
                current_pos = last_pos + 1
                if current_pos >= len(text):
                    break
            
            last_pos = current_pos
            chunk_end = min(current_pos + self.target_chunk_size, len(text))
            
            # Try to end at sentence boundary
            if chunk_end < len(text):
                sentence_end = max(
                    chunk_end - 200,
                    text.rfind('.', current_pos, chunk_end),
                    text.rfind('\n', current_pos, chunk_end)
                )
                if sentence_end > current_pos:
                    chunk_end = sentence_end + 1
            
            chunk_text = text[current_pos:chunk_end]
            
            # Add overlap
            if chunks and self.chunk_overlap > 0:
                overlap_start = max(0, current_pos - self.chunk_overlap)
                overlap_text = text[overlap_start:current_pos]
                chunk_text = overlap_text + chunk_text
            
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_pos=current_pos - self.chunk_overlap if chunks and self.chunk_overlap > 0 else current_pos,
                    end_pos=chunk_end,
                    chunk_index=chunk_index,
                    is_prioritized=False
                ))
                chunk_index += 1
            
            # Move to next chunk position (with overlap)
            next_pos = chunk_end - self.chunk_overlap if self.chunk_overlap > 0 else chunk_end
            
            # Ensure we always advance by at least 1 character
            if next_pos <= current_pos:
                next_pos = current_pos + 1
            
            # Also ensure we don't go backwards beyond previous chunk
            if chunks and next_pos <= chunks[-1].start_pos:
                next_pos = chunk_end
            
            # Final safety: ensure we advance
            if next_pos <= current_pos:
                next_pos = current_pos + 1
            
            current_pos = next_pos
        
        return chunks
    
    def estimate_chunk_summary_size(self, num_chunks: int) -> int:
        """Estimate size of combined chunk summaries.
        
        Args:
            num_chunks: Number of chunks.
            
        Returns:
            Estimated combined summary size in characters.
        """
        # Assume each chunk summary is ~10% of chunk size
        avg_summary_per_chunk = self.target_chunk_size * 0.1
        return int(num_chunks * avg_summary_per_chunk)
    
    def combine_chunk_summaries(self, chunk_summaries: List[str]) -> str:
        """Combine chunk summaries into single text with deduplication.
        
        Args:
            chunk_summaries: List of summary texts for each chunk.
            
        Returns:
            Combined summary text with duplicates removed.
        """
        if not chunk_summaries:
            return ""
        
        # Apply deduplication to remove repeated content across chunks
        from infrastructure.llm.validation.repetition import deduplicate_sections
        
        # First combine all summaries
        combined_parts = []
        for i, summary in enumerate(chunk_summaries):
            combined_parts.append(f"=== Chunk {i + 1} Summary ===\n{summary}\n")
        combined_text = "\n".join(combined_parts)
        
        # Apply deduplication to remove repeated sentences/paragraphs
        deduplicated = deduplicate_sections(
            combined_text,
            max_repetitions=1,
            mode="balanced",
            similarity_threshold=0.80,  # Slightly lower threshold for chunk combination
            min_content_preservation=0.75
        )
        
        return deduplicated


