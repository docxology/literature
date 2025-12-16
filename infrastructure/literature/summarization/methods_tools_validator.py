"""Validation for methods and tools extraction output.

This module provides comprehensive validation and cleaning of methods/tools
extraction results to prevent bloat, hallucinations, and poor quality output.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from infrastructure.core.logging_utils import get_logger

logger = get_logger(__name__)


class MethodsToolsValidator:
    """
    Validates and cleans methods/tools extraction output for quality issues.

    Detects and fixes:
    - Excessive "Not specified in paper" repetition (bloat)
    - Hallucinated tools/frameworks not explicitly mentioned in source
    - Empty sections with individual placeholder items
    - Output size anomalies (too many lines/words)
    - Formatting inconsistencies
    """

    def __init__(self, max_output_lines: int = 200, max_output_words: int = 1500):
        """Initialize the validator.

        Args:
            max_output_lines: Maximum allowed lines in output
            max_output_words: Maximum allowed words in output
        """
        self.max_output_lines = max_output_lines
        self.max_output_words = max_output_words

    def validate_output(
        self,
        methods_text: str,
        pdf_text: str,
        citation_key: str
    ) -> Tuple[str, List[str]]:
        """Validate and clean methods/tools extraction output.

        Performs comprehensive validation and cleaning:
        1. Detects excessive "Not specified" repetition
        2. Validates extracted items exist in source PDF
        3. Removes hallucinated entries
        4. Consolidates empty sections
        5. Checks output size limits
        6. Fixes formatting issues

        Args:
            methods_text: Raw methods/tools extraction output
            pdf_text: Source PDF text for validation
            citation_key: Citation key for logging

        Returns:
            Tuple of (cleaned_text, warnings_list)
        """
        logger.info(f"[{citation_key}] Starting methods/tools output validation")

        warnings = []
        cleaned_text = methods_text

        # Phase 0: Check original size (before consolidation) to detect problems
        initial_size_warnings = self._check_output_size(cleaned_text, citation_key)
        warnings.extend(initial_size_warnings)

        # Phase 1: Detect and fix "Not specified" bloat
        cleaned_text, bloat_warnings = self._fix_not_specified_bloat(cleaned_text, citation_key)
        warnings.extend(bloat_warnings)

        # Phase 2: Consolidate duplicate bullet points
        cleaned_text, duplicate_warnings = self._consolidate_duplicate_bullets(cleaned_text, citation_key)
        warnings.extend(duplicate_warnings)

        # Phase 3: Validate items against source PDF
        cleaned_text, hallucination_warnings = self._validate_against_source(cleaned_text, pdf_text, citation_key)
        warnings.extend(hallucination_warnings)

        # Phase 4: Check output size limits
        size_warnings = self._check_output_size(cleaned_text, citation_key)
        warnings.extend(size_warnings)

        # Phase 5: Fix formatting issues
        cleaned_text = self._fix_formatting_issues(cleaned_text)

        # Log final metrics
        final_lines = len(cleaned_text.split('\n'))
        final_words = len(cleaned_text.split())

        logger.info(
            f"[{citation_key}] Methods/tools validation completed: "
            f"{final_lines} lines, {final_words} words, "
            f"{len(warnings)} warnings"
        )

        return cleaned_text, warnings

    def _fix_not_specified_bloat(self, text: str, citation_key: str) -> Tuple[str, List[str]]:
        """Fix excessive 'Not specified in paper' repetition."""
        warnings = []

        # Count occurrences of "Not specified in paper" (case insensitive)
        not_specified_pattern = re.compile(r'not specified in paper', re.IGNORECASE)
        matches = not_specified_pattern.findall(text)

        if len(matches) > 3:
            warnings.append(
                f"Excessive 'Not specified in paper' repetition detected "
                f"({len(matches)} instances, max allowed: 3)"
            )

            # Replace all occurrences with a single consolidated section
            lines = text.split('\n')
            cleaned_lines = []
            consolidated_added = False
            skip_until_non_header = False
            last_was_not_specified = False

            for i, line in enumerate(lines):
                # Check if this is a section header followed by "Not specified"
                if line.startswith('##'):
                    # Look ahead to see if next non-empty line is "Not specified"
                    next_idx = i + 1
                    while next_idx < len(lines) and lines[next_idx].strip() == '':
                        next_idx += 1
                    
                    if next_idx < len(lines) and not_specified_pattern.search(lines[next_idx]):
                        # This is a header followed by "Not specified"
                        if not consolidated_added:
                            # Keep this header
                            cleaned_lines.append(line)
                            last_was_not_specified = False
                        else:
                            # Skip this header and the following "Not specified"
                            skip_until_non_header = True
                            continue
                    else:
                        # Normal header, not followed by "Not specified"
                        cleaned_lines.append(line)
                        skip_until_non_header = False
                        last_was_not_specified = False
                elif not_specified_pattern.search(line):
                    if skip_until_non_header:
                        # Skip this "Not specified" line
                        skip_until_non_header = False
                        last_was_not_specified = False
                        continue
                    elif not consolidated_added:
                        # First "Not specified" - keep it
                        cleaned_lines.append(line)
                        consolidated_added = True
                        last_was_not_specified = True
                    else:
                        # Additional "Not specified" - skip it
                        last_was_not_specified = False
                        continue
                elif line.strip() == '':
                    # Empty line - skip if it follows a "Not specified" we're consolidating
                    if skip_until_non_header or (last_was_not_specified and consolidated_added):
                        continue
                    # Otherwise, keep empty lines but track them
                    cleaned_lines.append(line)
                    last_was_not_specified = False
                else:
                    # Normal line (reset skip flag)
                    skip_until_non_header = False
                    last_was_not_specified = False
                    cleaned_lines.append(line)

            text = '\n'.join(cleaned_lines)
            
            # Remove excessive empty lines after consolidation
            text = self._remove_excessive_empty_lines(text)
            
            logger.warning(
                f"[{citation_key}] Consolidated {len(matches)} 'Not specified' instances "
                f"into single section"
            )

        return text, warnings

    def _remove_excessive_empty_lines(self, text: str) -> str:
        """Remove excessive empty lines from text.
        
        Collapses multiple consecutive empty lines into at most one empty line.
        Removes leading and trailing empty lines.
        """
        # Collapse 3+ consecutive newlines into max 2 (one empty line)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading empty lines
        text = text.lstrip('\n')
        
        # Remove trailing empty lines
        text = text.rstrip('\n')
        
        return text

    def _consolidate_duplicate_bullets(self, text: str, citation_key: str) -> Tuple[str, List[str]]:
        """Consolidate duplicate bullet points within sections.
        
        Removes excessive repetition of identical bullet points, keeping
        at most 3 instances per unique bullet point.
        """
        warnings = []
        lines = text.split('\n')
        cleaned_lines = []
        current_section_bullets = {}  # Track bullets in current section
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Section headers reset the bullet tracking
            if line.startswith('##'):
                # Reset tracking for new section
                current_section_bullets = {}
                cleaned_lines.append(line)
                i += 1
                continue
            
            # Check if this is a bullet point
            if line.strip().startswith('*'):
                bullet_content = line.strip()
                
                # Count occurrences of this exact bullet in current section
                if bullet_content not in current_section_bullets:
                    current_section_bullets[bullet_content] = 0
                
                current_section_bullets[bullet_content] += 1
                
                # Keep first 3 occurrences, skip rest
                if current_section_bullets[bullet_content] <= 3:
                    cleaned_lines.append(line)
                else:
                    # Skip this duplicate
                    if current_section_bullets[bullet_content] == 4:
                        # First time we hit the limit, add warning
                        warnings.append(
                            f"Consolidated duplicate bullet points: '{bullet_content[:50]}...' "
                            f"(keeping max 3 instances)"
                        )
            else:
                # Non-bullet line - keep it
                cleaned_lines.append(line)
            
            i += 1
        
        if warnings:
            logger.warning(
                f"[{citation_key}] Consolidated duplicate bullet points in sections"
            )
        
        return '\n'.join(cleaned_lines), warnings

    def _validate_against_source(
        self,
        text: str,
        pdf_text: str,
        citation_key: str
    ) -> Tuple[str, List[str]]:
        """Validate that extracted items actually exist in the source PDF."""
        warnings = []

        # Normalize PDF text for comparison
        pdf_normalized = re.sub(r'\s+', ' ', pdf_text.lower())

        lines = text.split('\n')
        validated_lines = []
        hallucinations_removed = 0

        for line in lines:
            # Skip section headers and empty lines
            if line.startswith('##') or line.strip() == '' or 'not specified in paper' in line.lower():
                validated_lines.append(line)
                continue

            # Extract tool/framework names from bullet points
            # Pattern: "* ToolName (exact quote from paper) – description"
            # Also handle: "*   Tool:PyTorch–description" or "*   Framework:TensorFlow–another"
            tool_match = re.match(r'\*\s*(?:Tool:|Framework:)?\s*([^(:–\-]+)', line)
            if tool_match:
                tool_name = tool_match.group(1).strip()

                # Skip generic/common items that might be inferred
                common_tools = {
                    'python', 'matlab', 'r', 'c++', 'java', 'javascript',
                    'tensorflow', 'pytorch', 'scikit-learn', 'numpy', 'pandas',
                    'accuracy', 'precision', 'recall', 'f1-score', 'mse',
                    't-tests', 'anova', 'google colab', 'aws', 'local clusters'
                }

                if tool_name.lower() in common_tools:
                    # Check if this specific tool is actually mentioned
                    if tool_name.lower() not in pdf_normalized:
                        warnings.append(
                            f"Potentially hallucinated common tool removed: '{tool_name}'"
                        )
                        hallucinations_removed += 1
                        continue
                else:
                    # For non-common tools, require explicit mention
                    if tool_name.lower() not in pdf_normalized:
                        # Try fuzzy matching for minor variations
                        tool_words = set(re.findall(r'\b\w+\b', tool_name.lower()))
                        pdf_words = set(re.findall(r'\b\w+\b', pdf_normalized))

                        if not tool_words.intersection(pdf_words):
                            warnings.append(
                                f"Hallucinated tool/framework removed: '{tool_name}'"
                            )
                            hallucinations_removed += 1
                            continue

            validated_lines.append(line)

        if hallucinations_removed > 0:
            logger.warning(
                f"[{citation_key}] Removed {hallucinations_removed} hallucinated "
                f"tools/frameworks not found in source PDF"
            )

        return '\n'.join(validated_lines), warnings

    def _check_output_size(self, text: str, citation_key: str) -> List[str]:
        """Check if output exceeds reasonable size limits."""
        warnings = []

        lines = text.split('\n')
        words = text.split()
        chars = len(text)

        if len(lines) > self.max_output_lines:
            warnings.append(
                f"Output exceeds line limit: {len(lines)}/{self.max_output_lines} lines"
            )

        if len(words) > self.max_output_words:
            warnings.append(
                f"Output exceeds word limit: {len(words)}/{self.max_output_words} words"
            )

        if chars > 50000:  # 50KB character limit
            warnings.append(
                f"Output exceeds character limit: {chars:,}/50,000 characters"
            )

        if warnings:
            logger.warning(
                f"[{citation_key}] Output size warnings: {', '.join(warnings)}"
            )

        return warnings

    def _fix_formatting_issues(self, text: str) -> str:
        """Fix common formatting issues in the output."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove excessive whitespace
            line = line.strip()

            # Fix spacing around colons and dashes
            line = re.sub(r'\s*:\s*', ': ', line)
            line = re.sub(r'\s*–\s*', ' – ', line)
            line = re.sub(r'\s*\(\s*', ' (', line)
            line = re.sub(r'\s*\)\s*', ') ', line)

            # Remove trailing punctuation inconsistencies
            line = re.sub(r'[,.]\s*$', '', line)

            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)
        
        # Consolidate empty lines (max 1 consecutive empty line)
        text = self._remove_excessive_empty_lines(text)
        
        return text
