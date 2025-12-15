"""Prompt building for paper summarization.

This module builds improved multi-stage prompts with examples and
validation checklists to guide the LLM toward better summaries.
"""
from __future__ import annotations

import re
from typing import List, Optional

from infrastructure.literature.summarization.models import SummarizationContext
from infrastructure.literature.summarization.utils import detect_model_size


class SummarizationPromptBuilder:
    """Builds optimized prompts for paper summarization.
    
    Creates model-size-aware prompts that are ~50% shorter than original
    versions while maintaining clarity. Automatically detects model size
    and adjusts prompt complexity accordingly.
    
    Features:
    - Model-size awareness (shorter prompts for small models <7B)
    - Consolidated instructions (removed redundant sections)
    - Explicit anti-repetition guidance with enhanced rules
    - Professional tone requirements (prohibiting conversational openings)
    - Quote spacing fix instructions (for PDF extraction issues)
    - Quote formatting standard with varied attribution phrases
    - Structured output format requirements
    - Fallback simple prompts for refinement failures
    """
    
    def __init__(self, llm_client=None):
        """Initialize prompt builder.
        
        Args:
            llm_client: Optional LLM client for model size detection.
        """
        self.llm_client = llm_client
        self._model_size_cache: Optional[float] = None
    
    def _detect_model_size(self, metadata: dict) -> float:
        """Detect model size in billions of parameters (with caching).
        
        Args:
            metadata: Paper metadata (may contain model info).
            
        Returns:
            Model size in billions (e.g., 4.0 for gemma3:4b), or 7.0 as default.
        """
        if self._model_size_cache is not None:
            return self._model_size_cache
        
        # Use shared utility function
        self._model_size_cache = detect_model_size(self.llm_client, metadata)
        return self._model_size_cache
    
    def _is_small_model(self, metadata: dict) -> bool:
        """Check if using a small model (<7B parameters)."""
        return self._detect_model_size(metadata) < 7.0
    
    def _truncate_full_text(
        self,
        full_text: str,
        max_chars: int,
        context: SummarizationContext
    ) -> str:
        """Intelligently truncate full_text while preserving key sections.
        
        Preserves abstract, introduction, and conclusion sections, truncating
        middle sections while maintaining structure.
        
        Args:
            full_text: Full paper text to truncate.
            max_chars: Maximum characters allowed.
            context: SummarizationContext with structured sections.
            
        Returns:
            Truncated text that fits within max_chars.
        """
        if len(full_text) <= max_chars:
            return full_text
        
        # Calculate space needed for key sections
        key_sections_size = (
            len(context.abstract or "") +
            len(context.introduction or "") +
            len(context.conclusion or "")
        )
        
        # Reserve space for key sections plus buffer for headers/formatting
        buffer = 2000  # Space for headers, instructions, etc.
        remaining_space = max_chars - key_sections_size - buffer
        
        if remaining_space < 10000:  # Minimum space for middle content
            # If not enough space, prioritize abstract and intro
            remaining_space = max_chars - len(context.abstract or "") - len(context.introduction or "") - buffer
            if remaining_space < 5000:
                # Last resort: just use abstract and intro
                return (context.abstract or "") + "\n\n" + (context.introduction or "") + "\n\n[Content truncated due to size limits]"
        
        # Try to preserve structure by finding section boundaries
        # Look for common section headers
        import re
        section_pattern = re.compile(
            r'^\s*(?:##?\s+)?(?:Abstract|Introduction|Methods?|Results?|Discussion|Conclusion)',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Find where introduction ends (approximate)
        intro_end = 0
        if context.introduction:
            intro_pos = full_text.find(context.introduction)
            if intro_pos >= 0:
                intro_end = intro_pos + len(context.introduction)
        
        # Find where conclusion starts
        conclusion_start = len(full_text)
        if context.conclusion:
            conclusion_pos = full_text.find(context.conclusion)
            if conclusion_pos >= 0:
                conclusion_start = conclusion_pos
        
        # Extract middle section
        middle_start = intro_end
        middle_end = conclusion_start
        
        if middle_end > middle_start:
            middle_text = full_text[middle_start:middle_end]
            
            # Truncate middle section to fit
            if len(middle_text) > remaining_space:
                # Truncate from middle, preserving start and end
                half_space = remaining_space // 2
                truncated_middle = (
                    middle_text[:half_space] +
                    "\n\n[... content truncated ...]\n\n" +
                    middle_text[-half_space:]
                )
            else:
                truncated_middle = middle_text
        else:
            # Fallback: truncate from end
            truncated_middle = full_text[intro_end:middle_start + remaining_space]
            if len(truncated_middle) < len(full_text[intro_end:]):
                truncated_middle += "\n\n[... content truncated ...]"
        
        # Reconstruct with preserved sections
        result_parts = []
        if context.abstract:
            result_parts.append(context.abstract)
        if context.introduction:
            result_parts.append(context.introduction)
        if truncated_middle:
            result_parts.append(truncated_middle)
        if context.conclusion:
            result_parts.append(context.conclusion)
        
        result = "\n\n".join(result_parts)
        
        # Final safety check
        if len(result) > max_chars:
            # Last resort: truncate from end
            result = result[:max_chars - 100] + "\n\n[... truncated ...]"
        
        return result
    
    def build_draft_prompt(
        self,
        context: SummarizationContext,
        metadata: dict,
        max_chars: Optional[int] = None
    ) -> str:
        """Build prompt for initial draft generation.
        
        Args:
            context: Structured context from paper.
            metadata: Paper metadata (title, authors, year, source).
            max_chars: Optional maximum characters for prompt. If provided and full_text
                      exceeds limit, will truncate intelligently.
            
        Returns:
            Complete prompt string for draft generation.
        """
        title = metadata.get('title', context.title)
        authors = metadata.get('authors', [])
        year = metadata.get('year', '')
        source = metadata.get('source', '')
        citation_key = metadata.get('citation_key', 'unknown')
        is_small = self._is_small_model(metadata)
        
        # Build prompt sections (optimized, ~50% shorter)
        sections = []
        
        # CRITICAL: Paper isolation warnings at the start
        sections.append("=== IMPORTANT: ISOLATE THIS PAPER ===")
        sections.append("You are summarizing ONLY the paper below. Do NOT reference or use content from any other papers.")
        sections.append("Do NOT mix information from different papers. Only use information from THIS specific paper.")
        sections.append("")
        sections.append(f"Paper Title: {title}")
        sections.append(f"Citation Key: {citation_key}")
        if authors:
            sections.append(f"Authors: {', '.join(authors[:3])}")
        sections.append("")
        sections.append("REMEMBER: Extract quotes, claims, and findings ONLY from the paper text provided below.")
        sections.append("")
        
        # Add year if not already included
        if year:
            sections.append(f"Year: {year}")
            sections.append("")
        
        # Section 2: Key Context (only for small models, skip for large)
        if is_small and context.abstract:
            sections.append("Abstract: " + context.abstract[:500] + ("..." if len(context.abstract) > 500 else ""))
            sections.append("")
        
        if context.key_terms:
            sections.append(f"Key Terms: {', '.join(context.key_terms[:10])}")
            sections.append("")
        
        # Section 3: Full Paper Text (single mention, concise instructions)
        sections.append("=== FULL PAPER TEXT ===")
        if not is_small:
            sections.append("Use this as your PRIMARY source for all information.")
        sections.append("")
        
        # Truncate full_text if max_chars provided and needed
        full_text = context.full_text
        if max_chars is not None:
            # Estimate space for other sections
            other_sections_size = sum(len(s) for s in sections) + 2000  # Buffer for instructions
            available_for_text = max_chars - other_sections_size
            
            if len(full_text) > available_for_text:
                full_text = self._truncate_full_text(full_text, available_for_text, context)
                sections.append("[NOTE: Paper text has been truncated to fit context window. Key sections (abstract, introduction, conclusion) are preserved.]")
                sections.append("")
        
        sections.append(full_text)
        sections.append("")
        
        # Section 4: Instructions (consolidated, model-aware)
        sections.append(self._build_instructions_section(context, is_small))
        
        return "\n".join(sections)
    
    def build_refinement_prompt(
        self,
        draft: str,
        issues: List[str],
        context: SummarizationContext,
        metadata: Optional[dict] = None
    ) -> str:
        """Build prompt for refining a draft summary (optimized, ~50% shorter).
        
        Args:
            draft: Current draft summary.
            issues: List of specific issues found during validation.
            context: Structured context from paper.
            metadata: Optional metadata for model size detection.
            
        Returns:
            Refinement prompt string.
        """
        is_small = self._is_small_model(metadata or {})
        sections = []
        
        # CRITICAL: Paper isolation warnings
        title = context.title
        citation_key = metadata.get('citation_key', 'unknown') if metadata else 'unknown'
        sections.append("=== IMPORTANT: ISOLATE THIS PAPER ===")
        sections.append("You are revising a summary for ONLY the paper below. Do NOT reference or use content from any other papers.")
        sections.append(f"Paper Title: {title}")
        sections.append(f"Citation Key: {citation_key}")
        sections.append("REMEMBER: Extract quotes, claims, and findings ONLY from the paper text provided below.")
        sections.append("")
        
        # Issues (concise)
        sections.append("Issues to fix:")
        for i, issue in enumerate(issues[:5], 1):  # Limit to top 5 issues
            sections.append(f"{i}. {issue}")
        sections.append("")
        
        # Current draft (truncated for small models)
        if is_small and len(draft) > 2000:
            sections.append("Current draft (first 2000 chars):")
            sections.append(draft[:2000] + "...")
        else:
            sections.append("Current draft:")
            sections.append(draft)
        sections.append("")
        
        # Key terms (if needed)
        if context.key_terms:
            sections.append(f"Key terms: {', '.join(context.key_terms[:8])}")
            sections.append("")
        
        # Full text (single mention)
        sections.append("=== FULL PAPER TEXT ===")
        sections.append(context.full_text)
        sections.append("")
        
        # Refinement instructions (consolidated)
        sections.append("=== REVISE TO ===")
        sections.append("PROFESSIONAL TONE: Begin directly with content - NO conversational openings like 'Okay, here's...'")
        sections.append("")
        sections.append(f"1. Fix all issues above")
        sections.append(f"2. Title: \"{context.title}\"")
        sections.append("3. Include 10-15 quotes from paper text")
        sections.append("   - Extract quotes VERBATIM from the paper text - do NOT modify or \"correct\" them")
        sections.append("   - FIX SPACING ISSUES FROM PDF EXTRACTION:")
        sections.append("     * PDF extraction may remove spaces between words (e.g., 'dynamicssimulationsand' → 'dynamics simulations and')")
        sections.append("     * When extracting quotes, restore missing spaces between words if they appear concatenated")
        sections.append("     * Look for patterns like 'word1word2word3' and add spaces: 'word1 word2 word3'")
        sections.append("     * This is a common issue from PDF text extraction that needs correction")
        sections.append("   - Use consistent quote formatting: 'The authors state: \"quote\"' or vary attribution phrases")
        sections.append("   - Vary attribution phrases to avoid repetition")
        sections.append("   - CRITICAL: Only extract quotes that actually appear in the paper text")
        
        # Add specific guidance based on issues
        has_missing_quotes = any("quote" in issue.lower() or "quotes" in issue.lower() for issue in issues)
        has_missing_methodology = any("methodology" in issue.lower() or "method" in issue.lower() for issue in issues)
        
        if has_missing_quotes:
            sections.append("   - Search the full paper text to find relevant quotes")
            sections.append("   - Extract quotes that support your claims")
        
        if has_missing_methodology:
            sections.append("   - Include detailed methodology: experimental setup, algorithms, procedures")
            sections.append("   - Describe the approach and implementation details")
        
        sections.append("4. ELIMINATE ALL REPETITION - each sentence must be unique")
        sections.append("   - Check before each sentence: 'Have I already said this?' If yes, write something new")
        sections.append("   - Vary attribution phrases - do NOT repeat 'The authors state' multiple times")
        sections.append("5. Extract methodology, results with numbers, key quotes")
        sections.append("6. 1000-1500 words, structured with ### headers")
        sections.append("")
        sections.append("Generate COMPLETE revised summary.")
        
        return "\n".join(sections)
    
    def build_simple_refinement_prompt(
        self,
        draft: str,
        issues: List[str],
        context: SummarizationContext
    ) -> str:
        """Build simplified refinement prompt for fallback strategy.
        
        Args:
            draft: Current draft summary.
            issues: List of specific issues.
            context: Structured context.
            
        Returns:
            Simplified refinement prompt.
        """
        sections = []
        sections.append(f"Fix these issues in your summary:")
        for issue in issues[:3]:  # Top 3 issues only
            sections.append(f"- {issue}")
        sections.append("")
        sections.append("Current summary:")
        sections.append(draft[:1500] if len(draft) > 1500 else draft)
        sections.append("")
        sections.append("Paper text:")
        sections.append(context.full_text[:30000] if len(context.full_text) > 30000 else context.full_text)
        sections.append("")
        sections.append("Rewrite the summary fixing the issues. Use exact title: " + context.title)
        sections.append("PROFESSIONAL TONE: Begin directly with content - NO conversational openings.")
        sections.append("NO REPETITION. Each sentence must be unique. Vary attribution phrases.")
        sections.append("Extract quotes VERBATIM from paper text - do NOT modify or \"correct\" them.")
        
        return "\n".join(sections)
    
    def _build_instructions_section(self, context: SummarizationContext, is_small: bool = False) -> str:
        """Build strong, explicit instructions section requesting specific elements.
        
        Includes:
        - Professional tone requirements (prohibiting conversational openings)
        - Quote extraction with spacing fix instructions
        - Quote formatting standard with varied attribution phrases
        - Enhanced anti-repetition rules with check-before-writing guidance
        - Structured output format requirements
        
        Args:
            context: Structured context from paper.
            is_small: Whether using a small model (<7B parameters).
            
        Returns:
            Complete instructions section string.
        """
        sections = []
        
        sections.append("=== INSTRUCTIONS ===")
        sections.append("")
        
        # Professional tone requirements
        sections.append("0. PROFESSIONAL TONE REQUIREMENTS:")
        sections.append("   - Begin directly with the paper title or content - NO conversational openings")
        sections.append("   - Do NOT use phrases like: 'Okay, here's...', 'Here's a summary...',")
        sections.append("     'Let me summarize...', 'I'll extract...', or similar conversational language")
        sections.append("   - Start immediately with substantive content in formal academic tone")
        sections.append("   - Example BAD: 'Okay, here's a summary of the paper...'")
        sections.append("   - Example GOOD: 'This paper investigates [topic]...'")
        sections.append("")
        
        # Title (critical)
        sections.append(f"1. Start with exact title: \"{context.title}\"")
        sections.append("")
        
        # Extract Quotes - explicit request for verbatim extraction
        sections.append("2. EXTRACT QUOTES:")
        sections.append("   - Extract 10-15 direct quotes from the paper that support key claims")
        sections.append("   - QUOTE EXTRACTION AND FORMATTING:")
        sections.append("     * Extract quotes VERBATIM from the paper text - do NOT modify or \"correct\" them")
        sections.append("     * Extract quotes exactly as they appear in the source text")
        sections.append("     * Preserve all aspects of the quote exactly as written, including spacing")
        sections.append("     * FIX SPACING ISSUES FROM PDF EXTRACTION:")
        sections.append("       - PDF extraction may remove spaces between words (e.g., 'dynamicssimulationsand' → 'dynamics simulations and')")
        sections.append("       - When extracting quotes, restore missing spaces between words if they appear concatenated")
        sections.append("       - Look for patterns like 'word1word2word3' and add spaces: 'word1 word2 word3'")
        sections.append("       - This is a common issue from PDF text extraction that needs correction")
        sections.append("     * Use proper quotation marks: \"quote text\" (double quotes)")
        sections.append("     * CRITICAL: Only extract quotes that actually appear in the paper text")
        sections.append("     * Do NOT generate, invent, or \"fix\" quotes - extract them exactly as written (with spacing normalized)")
        sections.append("   - QUOTE FORMATTING STANDARD:")
        sections.append("     * Attribution format: 'The authors state: \"quote text\"' OR 'According to the paper: \"quote text\"'")
        sections.append("     * Vary attribution phrases to avoid repetition (use: 'The authors state', 'They note',")
        sections.append("       'The paper argues', 'According to the research', 'The study demonstrates')")
        sections.append("     * Include section context when available: 'In the Introduction, the authors state: \"quote text\"'")
        sections.append("     * Ensure proper spacing around quotes and punctuation")
        sections.append("   - Search the full paper text to find relevant quotes")
        sections.append("   - Each quote must be verbatim from the paper text (with spacing normalized)")
        sections.append("   - FIX SPACING ISSUES FROM PDF EXTRACTION:")
        sections.append("     * PDF extraction may remove spaces between words (e.g., 'dynamicssimulationsand' → 'dynamics simulations and')")
        sections.append("     * When extracting quotes, restore missing spaces between words if they appear concatenated")
        sections.append("     * Look for patterns like 'word1word2word3' and add spaces: 'word1 word2 word3'")
        sections.append("     * This is a common issue from PDF text extraction that needs correction")
        sections.append("")
        
        # Identify Claims - explicit request
        sections.append("3. IDENTIFY CLAIMS:")
        sections.append("   - Identify the main claims and arguments made by the authors")
        sections.append("   - State each claim clearly and support it with quotes from the paper")
        sections.append("   - Distinguish between primary claims and supporting arguments")
        sections.append("")
        
        # Summarize Key Findings - explicit request
        sections.append("4. SUMMARIZE KEY FINDINGS:")
        sections.append("   - Summarize the key findings with specific numbers, metrics, and results")
        sections.append("   - Include quantitative data: percentages, statistics, measurements")
        sections.append("   - Extract numerical results from the results section")
        sections.append("   - Present findings with supporting evidence from the paper")
        sections.append("")
        
        # Describe Methods - explicit request
        sections.append("5. DESCRIBE METHODS:")
        sections.append("   - Describe the methodology, experimental setup, and approach used")
        sections.append("   - Include details about: algorithms, procedures, experimental design")
        sections.append("   - Explain how the research was conducted")
        sections.append("   - Extract specific methodological details from the methods section")
        sections.append("")
        
        # Present Results - explicit request
        sections.append("6. PRESENT RESULTS:")
        sections.append("   - Present the results with quantitative data and statistical significance")
        sections.append("   - Include specific numbers, tables, figures mentioned in the paper")
        sections.append("   - Extract results from the results section with exact values")
        sections.append("   - Support results with quotes or data from the paper")
        sections.append("")
        
        # Anti-repetition (CRITICAL) - Enhanced with explicit checks
        sections.append("7. NO REPETITION - CRITICAL REQUIREMENT (ENHANCED):")
        sections.append("   - CRITICAL: Before writing EACH sentence, check: 'Have I already said this exact idea?'")
        sections.append("   - If you've already stated an idea, DO NOT repeat it - move to the next unique point")
        sections.append("   - Each sentence must be COMPLETELY UNIQUE - no duplicate ideas, even with different words")
        sections.append("   - Each claim appears EXACTLY ONCE - if you've stated it, move to the next unique point")
        sections.append("   - Each paragraph must be COMPLETELY UNIQUE - no duplicate paragraphs")
        sections.append("   - Do NOT repeat the same sentence, even with slight variations or word changes")
        sections.append("   - Do NOT repeat paragraphs or sections - each section must have unique content")
        sections.append("   - Each claim should appear only ONCE in the entire summary")
        sections.append("   - Vary attribution phrases: use 'The authors state', 'They note', 'The paper argues',")
        sections.append("     'According to the research', 'The study demonstrates' - do NOT repeat the same phrase")
        sections.append("   - If you find yourself writing similar content, STOP immediately and write something completely different")
        sections.append("   - Before each sentence, ask: 'Have I already said this?' If yes, write something new")
        sections.append("   - Vary your language: use synonyms, different sentence structures, different perspectives")
        sections.append("   - REPETITION CHECKLIST: After writing each sentence, verify it's not a duplicate of any previous sentence")
        sections.append("")
        sections.append("   EXAMPLES OF WHAT NOT TO DO:")
        sections.append("   ❌ BAD: 'The authors state: \"X\". The authors state: \"Y\". The authors state: \"Z\".'")
        sections.append("   ✅ GOOD: 'The authors state: \"X\". They further note: \"Y\". The paper argues: \"Z\".'")
        sections.append("")
        sections.append("   ❌ BAD: Repeating the same claim 3+ times with slight variations")
        sections.append("   ✅ GOOD: State each claim once, then move to the next unique point")
        sections.append("")
        
        # Structure
        sections.append("8. STRUCTURE:")
        sections.append("   - Use markdown headers: ### Overview, ### Methodology, ### Results, ### Discussion")
        sections.append("   - Target length: 1000-1500 words")
        sections.append("   - Ensure all requested elements (quotes, claims, findings, methods, results) are included")
        sections.append("")
        
        return "\n".join(sections)
    
    def _build_validation_checklist(self, context: SummarizationContext) -> str:
        """Build validation checklist for summary quality.
        
        Args:
            context: Structured context from paper.
            
        Returns:
            Validation checklist string.
        """
        sections = []
        
        sections.append("=== VALIDATION CHECKLIST ===")
        sections.append("")
        sections.append("Before submitting, verify:")
        sections.append("")
        sections.append("1. Quote Requirements:")
        sections.append("   - Contains 10-15 quotes from the paper text")
        sections.append("   - Each quote is properly formatted: 'The authors state: \"quote\"'")
        sections.append("   - Quotes support the claims made in the summary")
        sections.append("")
        sections.append("2. Length Requirements:")
        sections.append("   - Summary is 1000-1500 words")
        sections.append("   - All sections are adequately covered")
        sections.append("")
        sections.append("3. Content Requirements:")
        sections.append("   - Title matches exactly: \"" + context.title + "\"")
        sections.append("   - All key terms are discussed")
        sections.append("   - Methodology is clearly described")
        sections.append("   - Results include numerical data")
        sections.append("   - Discussion provides meaningful insights")
        sections.append("")
        sections.append("4. Quality Requirements:")
        sections.append("   - NO repetition of sentences or paragraphs")
        sections.append("   - Each claim is supported by evidence")
        sections.append("   - Structure follows: ### Overview, ### Methodology, ### Results, ### Discussion")
        sections.append("")
        
        return "\n".join(sections)
    
    def build_examples_section(self) -> str:
        """Build section with examples of good and bad summaries."""
        sections = []
        
        sections.append("=== EXAMPLES ===")
        sections.append("")
        
        sections.append("GOOD SUMMARY EXAMPLE:")
        sections.append("---")
        sections.append("# Active Inference: A Process Theory")
        sections.append("")
        sections.append("### Overview")
        sections.append("This paper presents active inference as a process theory. The authors state:")
        sections.append("\"Active inference provides a unified framework for understanding behavior.\"")
        sections.append("")
        sections.append("### Key Contributions")
        sections.append("The paper introduces three key concepts: (1) variational free energy, (2) expected free energy,")
        sections.append("and (3) active inference. The abstract notes: [paraphrase of key finding].")
        sections.append("")
        sections.append("### Methodology")
        sections.append("The approach uses Bayesian inference principles...")
        sections.append("---")
        sections.append("")
        
        sections.append("BAD SUMMARY EXAMPLE (AVOID):")
        sections.append("---")
        sections.append("# Overview/Summary")
        sections.append("")
        sections.append("This paper presents methods. The paper presents methods.")
        sections.append("The paper presents methods. The paper presents methods.")
        sections.append("[REPETITION - BAD]")
        sections.append("")
        sections.append("The paper discusses neural networks. [WRONG - paper is about active inference, not neural networks]")
        sections.append("[HALLUCINATION - BAD]")
        sections.append("---")
        sections.append("")
        
        return "\n".join(sections)
