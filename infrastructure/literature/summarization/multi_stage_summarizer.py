"""Multi-stage summarization with draft and refine approach.

This module implements a two-stage summarization process:
1. Generate initial draft using structured context
2. Validate and refine based on specific issues
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

from infrastructure.core.logging_utils import get_logger
from infrastructure.core.exceptions import ContextLimitError
from infrastructure.literature.summarization.models import (
    SummarizationContext,
    SummarizationProgressEvent,
    ValidationResult
)
from infrastructure.literature.summarization.validator import SummaryQualityValidator
from infrastructure.literature.summarization.prompt_builder import SummarizationPromptBuilder
from infrastructure.literature.summarization.streaming import stream_with_progress
from infrastructure.literature.summarization.chunker import PDFChunker, ChunkingResult
from infrastructure.literature.summarization.utils import detect_model_size

if TYPE_CHECKING:
    from infrastructure.llm.core.client import LLMClient
    from infrastructure.llm.core.config import GenerationOptions

logger = get_logger(__name__)


def get_model_aware_generation_options(
    llm_client: "LLMClient",
    metadata: dict,
    stage: str = "draft",
    has_repetition_issue: bool = False
) -> "GenerationOptions":
    """Get model-aware generation options.
    
    Args:
        llm_client: LLM client for model detection.
        metadata: Paper metadata.
        stage: Generation stage ("draft" or "refinement").
        has_repetition_issue: Whether repetition was detected (lowers temperature).
        
    Returns:
        GenerationOptions with model-appropriate settings.
    """
    from infrastructure.llm.core.config import GenerationOptions
    
    # Detect model size using shared utility
    model_size = detect_model_size(llm_client, metadata)
    
    # Model-aware temperature with repetition handling
    if model_size < 7:
        # Small models: lower temperature for more focused output
        # Even lower if repetition detected
        if has_repetition_issue:
            temperature = 0.1  # Very low for repetition issues
        else:
            temperature = 0.3
        max_tokens = 2000 if stage == "draft" else 2500
    elif model_size <= 13:
        # Medium models
        if has_repetition_issue:
            temperature = 0.2  # Lower for repetition issues
        else:
            temperature = 0.4
        max_tokens = 2000 if stage == "draft" else 2500
    else:
        # Large models: slightly higher temperature
        if has_repetition_issue:
            temperature = 0.3  # Lower for repetition issues
        else:
            temperature = 0.5
        max_tokens = 2500 if stage == "draft" else 3000
    
    return GenerationOptions(
        temperature=temperature,
        max_tokens=max_tokens
    )


class MultiStageSummarizer:
    """Multi-stage summarization with draft generation and refinement.
    
    Implements a two-stage approach:
    1. Draft generation using structured context and model-aware prompts
    2. Validation and optional refinement based on specific issues
    
    Features:
    - Model-aware generation options (temperature, max_tokens)
    - Post-processing deduplication before validation
    - Fallback refinement strategies (simpler prompts, lower temperature)
    - Two-stage mode for large texts (chunk → summarize → combine)
    - Automatic retry with different strategies on failure
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        validator: SummaryQualityValidator,
        prompt_builder: SummarizationPromptBuilder,
        max_refinement_attempts: int = 2,
        two_stage_enabled: Optional[bool] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        two_stage_threshold: Optional[int] = None
    ):
        """Initialize multi-stage summarizer.
        
        Args:
            llm_client: LLM client for generation.
            validator: Quality validator instance.
            prompt_builder: Prompt builder instance.
            max_refinement_attempts: Maximum refinement attempts.
            two_stage_enabled: Enable two-stage mode (auto-detect from env if None).
            chunk_size: Target chunk size in chars (default: 15000).
            chunk_overlap: Overlap between chunks in chars (default: 500).
            two_stage_threshold: Text size threshold to trigger two-stage mode (default: 200000).
        """
        import os
        
        self.llm_client = llm_client
        self.validator = validator
        self.prompt_builder = prompt_builder
        self.max_refinement_attempts = max_refinement_attempts
        
        # Two-stage configuration
        if two_stage_enabled is None:
            two_stage_str = os.environ.get('LITERATURE_TWO_STAGE_ENABLED', 'true').lower()
            self.two_stage_enabled = two_stage_str in ('true', '1', 'yes')
        else:
            self.two_stage_enabled = two_stage_enabled
        
        self.chunk_size = chunk_size or int(os.environ.get('LITERATURE_CHUNK_SIZE', '15000'))
        # Reduce overlap for small models to minimize repetition
        default_overlap = chunk_overlap if chunk_overlap is not None else int(os.environ.get('LITERATURE_CHUNK_OVERLAP', '500'))
        # Detect model size and adjust overlap using shared utility
        model_size = detect_model_size(llm_client, None)
        if model_size < 7:
            # Small models: reduce overlap to 200
            default_overlap = 200
            logger.debug(f"Detected small model ({model_size}B), reducing chunk overlap to {default_overlap}")
        
        self.chunk_overlap = default_overlap
        self.two_stage_threshold = two_stage_threshold or int(os.environ.get('LITERATURE_TWO_STAGE_THRESHOLD', '200000'))
        
        self.chunker = PDFChunker(
            target_chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Auto two-stage fallback configuration
        import os
        auto_two_stage_str = os.environ.get('LITERATURE_AUTO_TWO_STAGE', 'true').lower()
        self.auto_two_stage_fallback = auto_two_stage_str in ('true', '1', 'yes')
    
    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate token count for a prompt (1 token ≈ 4 chars).
        
        Args:
            prompt: Prompt text to estimate.
            
        Returns:
            Estimated token count.
        """
        return len(prompt) // 4
    
    def _check_context_fits(self, prompt: str, reserve_percent: float = 0.2) -> tuple[bool, int, int]:
        """Check if prompt will fit in context window.
        
        Args:
            prompt: Prompt text to check.
            reserve_percent: Percentage of context to reserve for response (default: 0.2).
            
        Returns:
            Tuple of (fits, prompt_tokens, available_tokens):
            - fits: True if prompt will fit, False otherwise
            - prompt_tokens: Estimated tokens for the prompt
            - available_tokens: Available tokens in context window
        """
        prompt_tokens = self._estimate_prompt_tokens(prompt)
        available_tokens = self.llm_client.context.estimate_available_tokens(reserve_percent)
        fits = prompt_tokens <= available_tokens
        return fits, prompt_tokens, available_tokens
    
    def _fix_markdown_formatting(self, summary: str) -> str:
        """Fix markdown formatting issues in summary.
        
        Ensures proper spacing between headers and content.
        Fixes common issues like:
        - Headers without newlines: "### HeaderText" -> "### Header\n\nText"
        - Missing double newlines between sections
        - Excessive consecutive newlines
        
        Args:
            summary: Summary text that may have formatting issues.
            
        Returns:
            Summary text with proper markdown formatting.
        """
        if not summary:
            return summary
        
        # Fix headers without newlines after them: "### HeaderText" -> "### Header\n\nText"
        # Match: header (### or ##) followed by text, then capital letter (start of content)
        summary = re.sub(r'(###\s+[^\n]+)([A-Z][a-z])', r'\1\n\n\2', summary)
        summary = re.sub(r'(##\s+[^\n]+)([A-Z][a-z])', r'\1\n\n\2', summary)
        
        # Ensure double newlines before section headers (but not if already present)
        summary = re.sub(r'\n(###\s)', r'\n\n\1', summary)
        summary = re.sub(r'\n(##\s)', r'\n\n\1', summary)
        
        # Fix single newline after headers that should have double newline
        summary = re.sub(r'(###\s+[^\n]+)\n([A-Z])', r'\1\n\n\2', summary)
        summary = re.sub(r'(##\s+[^\n]+)\n([A-Z])', r'\1\n\n\2', summary)
        
        # Remove excessive newlines (more than 2 consecutive)
        summary = re.sub(r'\n{3,}', r'\n\n', summary)
        
        # Ensure proper spacing around section breaks
        # Fix cases where content runs into headers: "text###Header" -> "text\n\n###Header"
        summary = re.sub(r'([a-z])(###)', r'\1\n\n\2', summary)
        summary = re.sub(r'([a-z])(##)', r'\1\n\n\2', summary)
        
        # Clean up any remaining formatting issues
        summary = summary.strip()
        
        return summary
    
    def generate_draft(
        self,
        context: SummarizationContext,
        metadata: dict,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None
    ) -> str:
        """Generate initial draft summary.
        
        Args:
            context: Structured context from paper.
            metadata: Paper metadata (title, authors, year, source).
            progress_callback: Optional callback for progress events.
            
        Returns:
            Draft summary text.
        """
        citation_key = metadata.get('citation_key')
        if not citation_key:
            # This should never happen if metadata is properly constructed, but log warning if it does
            logger.warning("citation_key missing from metadata, using fallback 'unknown'")
            citation_key = 'unknown'
        
        # Check if we should use two-stage mode based on text size threshold
        use_two_stage_by_size = (
            self.two_stage_enabled and 
            len(context.full_text) > self.two_stage_threshold
        )
        
        if use_two_stage_by_size:
            logger.info(
                f"[{citation_key}] Text size ({len(context.full_text):,} chars) exceeds threshold "
                f"({self.two_stage_threshold:,} chars), using two-stage mode..."
            )
            return self._generate_draft_two_stage(context, metadata, progress_callback)
        
        # Try single-stage mode first, but check if prompt will fit
        logger.info(f"[{citation_key}] Generating draft summary (single-stage mode, full paper context)...")
        
        # Build prompt to check size
        prompt = self.prompt_builder.build_draft_prompt(context, metadata)
        fits, prompt_tokens, available_tokens = self._check_context_fits(prompt)
        
        if not fits:
            if self.auto_two_stage_fallback and self.two_stage_enabled:
                logger.warning(
                    f"[{citation_key}] Prompt too large for context window "
                    f"({prompt_tokens:,} tokens > {available_tokens:,} available). "
                    f"Automatically switching to two-stage mode..."
                )
                return self._generate_draft_two_stage(context, metadata, progress_callback)
            else:
                logger.error(
                    f"[{citation_key}] Prompt too large for context window "
                    f"({prompt_tokens:,} tokens > {available_tokens:,} available). "
                    f"Two-stage mode is {'disabled' if not self.two_stage_enabled else 'not auto-enabled'}. "
                    f"Consider enabling LITERATURE_AUTO_TWO_STAGE=true or truncating content."
                )
                # Will raise ContextLimitError when trying to add message
                # This allows the error to propagate with proper context
        
        return self._generate_draft_single_stage(context, metadata, progress_callback)
    
    def _generate_draft_single_stage(
        self,
        context: SummarizationContext,
        metadata: dict,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None,
        has_repetition_issue: bool = False
    ) -> str:
        """Generate draft summary using single-stage mode (original method).
        
        Args:
            context: Structured context from paper.
            metadata: Paper metadata.
            progress_callback: Optional callback for progress events.
            
        Returns:
            Draft summary text.
        """
        citation_key = metadata.get('citation_key', 'unknown')
        
        # Emit progress event
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="draft_generation",
                status="started",
                message="Generating draft summary (single-stage)..."
            ))
        
        draft_start_time = time.time()
        
        # Build draft prompt
        logger.debug(f"[{citation_key}] Building draft prompt...")
        prompt = self.prompt_builder.build_draft_prompt(context, metadata)
        prompt_size = len(prompt)
        logger.info(f"[{citation_key}] Draft prompt built: {prompt_size:,} chars (full_text={len(context.full_text):,} chars)")
        
        # Pre-flight check: Verify prompt will fit in context window
        fits, prompt_tokens, available_tokens = self._check_context_fits(prompt)
        if not fits:
            error_msg = (
                f"Prompt too large for context window: {prompt_tokens:,} tokens > {available_tokens:,} available. "
                f"Consider using two-stage mode or truncating content."
            )
            logger.error(f"[{citation_key}] {error_msg}")
            raise ContextLimitError(
                error_msg,
                context={
                    "size": prompt_tokens,
                    "limit": self.llm_client.context.max_tokens,
                    "available": available_tokens,
                    "estimated_total": self.llm_client.context.estimated_tokens + prompt_tokens,
                    "suggestion": "Use two-stage mode or truncate content"
                }
            )
        
        logger.debug(
            f"[{citation_key}] Pre-flight check passed: {prompt_tokens:,} tokens <= {available_tokens:,} available"
        )
        
        # Save prompt to file for debugging
        try:
            debug_dir = Path("data/summaries") / "_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = debug_dir / f"{citation_key}_draft_prompt.txt"
            prompt_path.write_text(prompt, encoding='utf-8')
            logger.debug(f"[{citation_key}] Saved draft prompt: {prompt_path} ({prompt_size:,} chars)")
        except Exception as save_error:
            logger.debug(f"[{citation_key}] Failed to save prompt: {save_error}")
        
        # Pre-streaming validation: Check LLM connection
        logger.debug(f"[{citation_key}] Checking LLM connection before streaming...")
        try:
            if not self.llm_client.check_connection():
                raise ConnectionError(f"LLM connection check failed for {citation_key}")
            logger.debug(f"[{citation_key}] LLM connection verified")
        except Exception as conn_error:
            logger.error(f"[{citation_key}] LLM connection check failed: {conn_error}")
            raise
        
        # Get model-aware generation options (with repetition handling)
        gen_options = get_model_aware_generation_options(
            self.llm_client, metadata, stage="draft", has_repetition_issue=has_repetition_issue
        )
        
        # Generate summary using streaming with progress updates
        try:
            draft = stream_with_progress(
                llm_client=self.llm_client,
                prompt=prompt,
                progress_callback=progress_callback,
                citation_key=citation_key,
                stage="draft_generation",
                update_interval=5.0,
                options=gen_options
            )
            draft_time = time.time() - draft_start_time
            
            # Post-streaming validation: Check response quality
            if not draft or not draft.strip():
                raise ValueError(f"Empty draft response received for {citation_key}")
            
            draft_words = len(draft.split())
            draft_chars = len(draft)
            
            # Validate minimum response size (at least 50 words expected for a summary)
            if draft_words < 50:
                logger.warning(
                    f"[{citation_key}] Draft response is unusually short: {draft_words} words, {draft_chars} chars. "
                    f"This may indicate an incomplete response."
                )
            
            logger.info(
                f"[{citation_key}] Draft generated in {draft_time:.2f}s: "
                f"{draft_chars:,} chars, {draft_words} words "
                f"({draft_chars/max(draft_time, 0.01):.0f} chars/s)"
            )
            
            # Save draft to file for debugging
            try:
                debug_dir = Path("data/summaries") / "_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                draft_path = debug_dir / f"{citation_key}_draft.md"
                draft_path.write_text(draft, encoding='utf-8')
                logger.debug(f"[{citation_key}] Saved draft: {draft_path} ({draft_words} words)")
            except Exception as save_error:
                logger.debug(f"[{citation_key}] Failed to save draft: {save_error}")
            
            # Emit progress event
            if progress_callback:
                progress_callback(SummarizationProgressEvent(
                    citation_key=citation_key,
                    stage="draft_generation",
                    status="completed",
                    message=f"Draft generated: {draft_words} words",
                    metadata={"chars": len(draft), "words": draft_words, "time": draft_time}
                ))
            
            return draft
        except Exception as e:
            draft_time = time.time() - draft_start_time
            import traceback
            error_type = type(e).__name__
            error_context = {
                "stage": "draft_generation",
                "elapsed_time": f"{draft_time:.2f}s",
                "prompt_size": f"{len(prompt):,} chars",
                "context_full_text": f"{len(context.full_text):,} chars",
                "error_type": error_type
            }
            logger.error(
                f"[{citation_key}] Failed to generate draft after {draft_time:.2f}s: {e}\n"
                f"Context: {error_context}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            
            # If ContextLimitError and two-stage is available, suggest it
            if isinstance(e, ContextLimitError) and self.two_stage_enabled and self.auto_two_stage_fallback:
                logger.warning(
                    f"[{citation_key}] Context limit error in single-stage mode. "
                    f"Two-stage mode should have been used automatically. "
                    f"This may indicate a bug in the size checking logic."
                )
            
            # Emit progress event for failure
            if progress_callback:
                try:
                    progress_callback(SummarizationProgressEvent(
                        citation_key=citation_key,
                        stage="draft_generation",
                        status="failed",
                        message=f"Failed to generate draft: {e}"
                    ))
                except Exception as callback_error:
                    logger.warning(f"[{citation_key}] Progress callback error during failure: {callback_error}")
            
            raise
    
    def _generate_draft_two_stage(
        self,
        context: SummarizationContext,
        metadata: dict,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None
    ) -> str:
        """Generate draft summary using two-stage mode (chunk → summarize chunks → final summary).
        
        Args:
            context: Structured context from paper.
            metadata: Paper metadata.
            progress_callback: Optional callback for progress events.
            
        Returns:
            Draft summary text.
        """
        citation_key = metadata.get('citation_key', 'unknown')
        two_stage_start = time.time()
        
        # Stage 1.1: Chunk the PDF text
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="chunking",
                status="started",
                message="Chunking PDF text for two-stage summarization..."
            ))
        
        logger.info(f"[{citation_key}] Two-stage mode: Chunking PDF text ({len(context.full_text):,} chars)...")
        chunking_result = self.chunker.chunk_text(context.full_text, preserve_sections=True)
        
        logger.info(
            f"[{citation_key}] Chunking completed: {chunking_result.total_chunks} chunks, "
            f"avg size: {chunking_result.average_chunk_size:.0f} chars, "
            f"prioritized: {chunking_result.prioritized_chunks}"
        )
        
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="chunking",
                status="completed",
                message=f"Chunked into {chunking_result.total_chunks} chunks",
                metadata={
                    "total_chunks": chunking_result.total_chunks,
                    "average_chunk_size": chunking_result.average_chunk_size,
                    "prioritized_chunks": chunking_result.prioritized_chunks
                }
            ))
        
        # Stage 1.2: Summarize each chunk
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="chunk_summarization",
                status="started",
                message=f"Summarizing {chunking_result.total_chunks} chunks..."
            ))
        
        chunk_summaries = []
        for i, chunk in enumerate(chunking_result.chunks):
            chunk_num = i + 1
            logger.info(
                f"[{citation_key}] Summarizing chunk {chunk_num}/{chunking_result.total_chunks} "
                f"({len(chunk.text):,} chars, section: {chunk.section_name or 'unknown'})..."
            )
            
            if progress_callback:
                progress_callback(SummarizationProgressEvent(
                    citation_key=citation_key,
                    stage="chunk_summarization",
                    status="in_progress",
                    message=f"Summarizing chunk {chunk_num}/{chunking_result.total_chunks}...",
                    metadata={
                        "chunk_index": chunk_num,
                        "total_chunks": chunking_result.total_chunks,
                        "chunk_size": len(chunk.text),
                        "section_name": chunk.section_name
                    }
                ))
            
            # Create a simplified context for this chunk
            chunk_context = SummarizationContext(
                title=context.title,
                abstract=context.abstract if chunk.is_prioritized else "",
                introduction=context.introduction if chunk.is_prioritized else "",
                conclusion=context.conclusion if chunk.is_prioritized else "",
                key_terms=context.key_terms if chunk.is_prioritized else [],
                equations=context.equations if chunk.is_prioritized else [],
                full_text=chunk.text
            )
            
            # Build prompt for chunk summary
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_info'] = f"Chunk {chunk_num} of {chunking_result.total_chunks}"
            chunk_prompt = self.prompt_builder.build_draft_prompt(chunk_context, chunk_metadata)
            
            # Get generation options for chunk
            chunk_metadata = metadata.copy()
            chunk_gen_options = get_model_aware_generation_options(
                self.llm_client, chunk_metadata, stage="draft", has_repetition_issue=False
            )
            
            # Summarize chunk
            try:
                chunk_summary = stream_with_progress(
                    llm_client=self.llm_client,
                    prompt=chunk_prompt,
                    progress_callback=None,  # Don't emit progress for individual chunks
                    citation_key=f"{citation_key}_chunk{chunk_num}",
                    stage="chunk_summarization",
                    update_interval=5.0,
                    options=chunk_gen_options
                )
                
                if chunk_summary and chunk_summary.strip():
                    chunk_summaries.append(chunk_summary)
                    logger.info(
                        f"[{citation_key}] Chunk {chunk_num} summarized: {len(chunk_summary.split())} words"
                    )
                else:
                    logger.warning(f"[{citation_key}] Chunk {chunk_num} produced empty summary")
            except Exception as e:
                logger.error(f"[{citation_key}] Failed to summarize chunk {chunk_num}: {e}")
                # Continue with other chunks
        
        if not chunk_summaries:
            raise ValueError(f"No chunk summaries generated for {citation_key}")
        
        logger.info(
            f"[{citation_key}] Chunk summarization completed: {len(chunk_summaries)}/{chunking_result.total_chunks} chunks summarized"
        )
        
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="chunk_summarization",
                status="completed",
                message=f"Summarized {len(chunk_summaries)} chunks",
                metadata={"chunks_summarized": len(chunk_summaries), "total_chunks": chunking_result.total_chunks}
            ))
        
        # Stage 1.3: Combine chunk summaries and generate final summary
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="final_summarization",
                status="started",
                message="Generating final summary from chunk summaries..."
            ))
        
        combined_summaries = self.chunker.combine_chunk_summaries(chunk_summaries)
        combined_size = len(combined_summaries)
        
        logger.info(
            f"[{citation_key}] Two-stage mode: Generating final summary from {len(chunk_summaries)} chunk summaries "
            f"({combined_size:,} chars total)..."
        )
        
        # Create context for final summary (use combined summaries as full_text)
        final_context = SummarizationContext(
            title=context.title,
            abstract=context.abstract,
            introduction=context.introduction,
            conclusion=context.conclusion,
            key_terms=context.key_terms,
            equations=context.equations,
            full_text=combined_summaries
        )
        
        # Build final prompt
        final_metadata = metadata.copy()
        final_metadata['two_stage'] = True
        final_metadata['num_chunks'] = len(chunk_summaries)
        final_prompt = self.prompt_builder.build_draft_prompt(final_context, final_metadata)
        
        logger.info(
            f"[{citation_key}] Final summary prompt built: {len(final_prompt):,} chars "
            f"(combined_summaries: {combined_size:,} chars)"
        )
        
        # Get generation options for final summary
        final_gen_options = get_model_aware_generation_options(
            self.llm_client, metadata, stage="draft", has_repetition_issue=False
        )
        
        # Generate final summary
        final_summary = stream_with_progress(
            llm_client=self.llm_client,
            prompt=final_prompt,
            progress_callback=progress_callback,
            citation_key=citation_key,
            stage="final_summarization",
            update_interval=5.0,
            options=final_gen_options
        )
        
        two_stage_time = time.time() - two_stage_start
        final_words = len(final_summary.split())
        
        logger.info(
            f"[{citation_key}] Two-stage summarization completed in {two_stage_time:.2f}s: "
            f"{final_words} words (from {chunking_result.total_chunks} chunks)"
        )
        
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="final_summarization",
                status="completed",
                message=f"Final summary generated: {final_words} words",
                metadata={
                    "words": final_words,
                    "time": two_stage_time,
                    "chunks_used": chunking_result.total_chunks
                }
            ))
        
        return final_summary
    
    def refine_summary(
        self,
        draft: str,
        issues: List[str],
        context: SummarizationContext,
        citation_key: str,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None,
        attempt_number: int = 1,
        max_attempts: int = 2,
        use_simple_prompt: bool = False,
        temperature: Optional[float] = None,
        refinement_guidance: Optional[str] = None
    ) -> str:
        """Refine summary based on validation issues.
        
        Args:
            draft: Current draft summary.
            issues: List of specific issues found during validation.
            context: Structured context from paper.
            citation_key: Citation key for logging.
            progress_callback: Optional callback for progress events.
            attempt_number: Current refinement attempt number.
            max_attempts: Maximum refinement attempts.
            
        Returns:
            Refined summary text.
        """
        logger.info(
            f"[{citation_key}] Stage 2: Refining summary to address {len(issues)} issues: "
            f"{', '.join(issues[:3])}{'...' if len(issues) > 3 else ''}"
        )
        
        # Emit progress event
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="refinement",
                status="started",
                message=f"Refining summary (attempt {attempt_number}/{max_attempts})",
                metadata={"attempt": attempt_number, "max_attempts": max_attempts, "issues_count": len(issues)}
            ))
        
        refine_start_time = time.time()
        
        # Build refinement prompt (use simple prompt for fallback)
        logger.debug(f"[{citation_key}] Building refinement prompt (simple={use_simple_prompt})...")
        if use_simple_prompt:
            prompt = self.prompt_builder.build_simple_refinement_prompt(draft, issues, context)
        else:
            metadata = {'citation_key': citation_key, 'refinement_guidance': refinement_guidance}
            prompt = self.prompt_builder.build_refinement_prompt(draft, issues, context, metadata)
        prompt_size = len(prompt)
        logger.info(f"[{citation_key}] Refinement prompt built: {prompt_size:,} chars (full_text={len(context.full_text):,} chars)")
        
        # Save refinement prompt to file for debugging
        try:
            debug_dir = Path("data/summaries") / "_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = debug_dir / f"{citation_key}_refinement_prompt_attempt{attempt_number}.txt"
            prompt_path.write_text(prompt, encoding='utf-8')
            logger.debug(f"[{citation_key}] Saved refinement prompt: {prompt_path} ({prompt_size:,} chars)")
        except Exception as save_error:
            logger.debug(f"[{citation_key}] Failed to save refinement prompt: {save_error}")
        
        # Pre-streaming validation: Check LLM connection
        logger.debug(f"[{citation_key}] Checking LLM connection before refinement streaming...")
        try:
            if not self.llm_client.check_connection():
                raise ConnectionError(f"LLM connection check failed for {citation_key} (refinement)")
            logger.debug(f"[{citation_key}] LLM connection verified for refinement")
        except Exception as conn_error:
            logger.error(f"[{citation_key}] LLM connection check failed during refinement: {conn_error}")
            raise
        
        # Get model-aware generation options for refinement
        # Check if repetition is an issue - use lower temperature
        has_repetition_issue = any(
            "repetition" in issue.lower() 
            for issue in issues
        )
        
        refine_metadata = {'citation_key': citation_key}
        gen_options = get_model_aware_generation_options(
            self.llm_client, refine_metadata, stage="refinement", has_repetition_issue=has_repetition_issue
        )

        # Override temperature if specified
        if temperature is not None:
            gen_options.temperature = temperature
            logger.debug(f"[{citation_key}] Using specified temperature {temperature:.2f}")

        if has_repetition_issue or temperature is not None:
            logger.debug(f"[{citation_key}] Using temperature {gen_options.temperature:.2f}")
        
        # Generate refined summary using streaming with progress updates
        try:
            refined = stream_with_progress(
                llm_client=self.llm_client,
                prompt=prompt,
                progress_callback=progress_callback,
                citation_key=citation_key,
                stage="refinement",
                update_interval=5.0,
                options=gen_options
            )
            refine_time = time.time() - refine_start_time
            
            # Post-streaming validation: Check response quality
            if not refined or not refined.strip():
                raise ValueError(f"Empty refined response received for {citation_key} (attempt {attempt_number})")
            
            refined_words = len(refined.split())
            refined_chars = len(refined)
            
            # Validate minimum response size
            if refined_words < 50:
                logger.warning(
                    f"[{citation_key}] Refined response is unusually short: {refined_words} words, {refined_chars} chars "
                    f"(attempt {attempt_number}). This may indicate an incomplete response."
                )
            
            logger.info(
                f"[{citation_key}] Refined summary in {refine_time:.2f}s: "
                f"{refined_chars:,} chars, {refined_words} words "
                f"({refined_chars/max(refine_time, 0.01):.0f} chars/s, attempt {attempt_number})"
            )
            
            # Save refined summary to file for debugging
            try:
                debug_dir = Path("data/summaries") / "_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                refined_path = debug_dir / f"{citation_key}_refined_attempt{attempt_number}.md"
                refined_path.write_text(refined, encoding='utf-8')
                logger.debug(f"[{citation_key}] Saved refined summary: {refined_path} ({refined_words} words)")
            except Exception as save_error:
                logger.debug(f"[{citation_key}] Failed to save refined summary: {save_error}")
            
            # Emit progress event
            if progress_callback:
                progress_callback(SummarizationProgressEvent(
                    citation_key=citation_key,
                    stage="refinement",
                    status="completed",
                    message=f"Refinement completed: {refined_words} words",
                    metadata={"chars": len(refined), "words": refined_words, "time": refine_time, "attempt": attempt_number}
                ))
            
            return refined
        except Exception as e:
            refine_time = time.time() - refine_start_time
            import traceback
            error_type = type(e).__name__
            error_context = {
                "stage": "refinement",
                "attempt": f"{attempt_number}/{max_attempts}",
                "elapsed_time": f"{refine_time:.2f}s",
                "prompt_size": f"{len(prompt):,} chars",
                "context_full_text": f"{len(context.full_text):,} chars",
                "issues_count": len(issues),
                "draft_size": f"{len(draft):,} chars",
                "error_type": error_type
            }
            logger.error(
                f"[{citation_key}] Failed to refine summary after {refine_time:.2f}s: {e}\n"
                f"Context: {error_context}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            
            # Emit progress event for failure
            if progress_callback:
                try:
                    progress_callback(SummarizationProgressEvent(
                        citation_key=citation_key,
                        stage="refinement",
                        status="failed",
                        message=f"Failed to refine summary: {e}"
                    ))
                except Exception as callback_error:
                    logger.warning(f"[{citation_key}] Progress callback error during failure: {callback_error}")
            
            raise
    
    def validate_and_accept(
        self,
        summary: str,
        context: SummarizationContext,
        pdf_text: str,
        paper_title: str,
        citation_key: str,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None
    ) -> Tuple[bool, ValidationResult]:
        """Validate summary and determine if it should be accepted.
        
        Args:
            summary: Summary text to validate.
            context: Structured context from paper.
            pdf_text: Full PDF text for validation.
            paper_title: Paper title.
            citation_key: Citation key for logging.
            progress_callback: Optional callback for progress events.
            
        Returns:
            Tuple of (should_accept, ValidationResult).
        """
        logger.info(f"[{citation_key}] Validating summary quality...")
        
        # Emit progress event
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="validation",
                status="started",
                message="Validating summary quality..."
            ))
        
        validation_start_time = time.time()
        
        # Validate using detailed validation
        validation_result = self.validator.validate_summary_detailed(
            summary=summary,
            pdf_text=pdf_text,
            citation_key=citation_key,
            paper_title=paper_title,
            key_terms=context.key_terms
        )
        
        validation_time = time.time() - validation_start_time
        
        # Log validation results
        if validation_result.errors:
            logger.warning(
                f"[{citation_key}] Validation found {len(validation_result.errors)} errors: "
                f"{', '.join(validation_result.errors[:2])}{'...' if len(validation_result.errors) > 2 else ''}"
            )
        if validation_result.warnings:
            logger.debug(
                f"[{citation_key}] Validation found {len(validation_result.warnings)} warnings"
            )
        
        # Check for hard failures
        if validation_result.has_hard_failure():
            logger.error(
                f"[{citation_key}] Hard failure detected (validation time: {validation_time:.2f}s): "
                f"{validation_result.errors[0] if validation_result.errors else 'Unknown error'}"
            )
            return False, validation_result
        
        # Accept if valid or if score is acceptable (>= 0.5)
        should_accept = validation_result.is_valid or validation_result.score >= 0.5
        
        if should_accept:
            logger.info(
                f"[{citation_key}] Summary accepted (score: {validation_result.score:.2f}, "
                f"validation time: {validation_time:.2f}s, quotes: {validation_result.quote_count})"
            )
        else:
            logger.warning(
                f"[{citation_key}] Summary rejected (score: {validation_result.score:.2f}, "
                f"validation time: {validation_time:.2f}s)"
            )
        
        # Emit progress event
        if progress_callback:
            status_msg = f"Validation completed: score {validation_result.score:.2f}"
            if should_accept:
                status_msg += " (accepted)"
            else:
                status_msg += " (rejected)"
            
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage="validation",
                status="completed",
                message=status_msg,
                metadata={
                    "score": validation_result.score,
                    "is_valid": validation_result.is_valid,
                    "should_accept": should_accept,
                    "errors_count": len(validation_result.errors),
                    "warnings_count": len(validation_result.warnings),
                    "quote_count": validation_result.quote_count,
                    "time": validation_time
                }
            ))
        
        return should_accept, validation_result
    
    def summarize_with_refinement(
        self,
        context: SummarizationContext,
        pdf_text: str,
        metadata: dict,
        citation_key: str,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None
    ) -> Tuple[str, ValidationResult, int]:
        """Generate summary with automatic refinement.
        
        Args:
            context: Structured context from paper.
            pdf_text: Full PDF text for validation.
            metadata: Paper metadata.
            citation_key: Citation key for logging.
            progress_callback: Optional callback for progress events.
            
        Returns:
            Tuple of (final_summary, final_validation_result, total_attempts).
        """
        paper_title = metadata.get('title', context.title)
        total_attempts = 0
        overall_start_time = time.time()
        
        # Stage 1: Generate draft
        try:
            draft = self.generate_draft(context, metadata, progress_callback=progress_callback)
            total_attempts += 1
        except TimeoutError as e:
            overall_time = time.time() - overall_start_time
            logger.error(
                f"[{citation_key}] Draft generation timed out after {overall_time:.2f}s: {e}. "
                f"Cannot proceed with summarization. "
                f"Consider: 1) Increasing LLM_SUMMARIZATION_TIMEOUT, 2) Using two-stage mode, "
                f"3) Reducing prompt size"
            )
            # Re-raise to prevent refinement attempt
            raise
        except Exception as e:
            overall_time = time.time() - overall_start_time
            logger.error(
                f"[{citation_key}] Draft generation failed after {overall_time:.2f}s: {e}"
            )
            raise
        
        # Check draft validity before processing
        if not draft or not draft.strip() or len(draft.strip()) < 100:
            logger.error(
                f"[{citation_key}] Draft is empty or too short ({len(draft) if draft else 0} chars). "
                f"Cannot proceed with validation or refinement."
            )
            raise ValueError(f"Invalid draft for {citation_key}: empty or too short")
        
        # Apply post-generation deduplication before validation
        from infrastructure.llm.validation.repetition import deduplicate_sections
        draft = deduplicate_sections(
            draft,
            max_repetitions=1,
            mode="aggressive",
            similarity_threshold=0.85,
            min_content_preservation=0.7
        )
        
        # Apply formatting fixes after deduplication
        draft = self._fix_markdown_formatting(draft)
        
        # Stage 2: Validate draft
        should_accept, validation_result = self.validate_and_accept(
            draft, context, pdf_text, paper_title, citation_key, progress_callback=progress_callback
        )
        
        if should_accept:
            overall_time = time.time() - overall_start_time
            logger.info(
                f"[{citation_key}] Summarization completed in {overall_time:.2f}s "
                f"(draft accepted, {total_attempts} attempt(s))"
            )
            return draft, validation_result, total_attempts
        
        # Stage 3: Refine if needed
        current_summary = draft
        refinement_attempts = 0
        
        # Progressive fallback strategies based on validation failures
        fallback_strategy = self._determine_fallback_strategy(validation_result, attempt_number)

        if fallback_strategy:
            logger.info(
                f"[{citation_key}] Applying fallback strategy: {fallback_strategy['name']} "
                f"(attempt {attempt_number + 1}/{max_attempts})"
            )

            current_summary = self._apply_fallback_strategy(
                fallback_strategy,
                current_summary,
                context,
                pdf_text,
                paper_title,
                citation_key,
                progress_callback,
                validation_result
            )

            # Re-validate after fallback
            should_accept, validation_result = self.validate_and_accept(
                current_summary, context, pdf_text, paper_title, citation_key,
                progress_callback=progress_callback
            )

            if should_accept:
                overall_time = time.time() - overall_start_time
                logger.info(
                    f"[{citation_key}] Summarization completed in {overall_time:.2f}s "
                    f"(accepted after {fallback_strategy['name']}, {total_attempts} attempt(s))"
                )
                return current_summary, validation_result, total_attempts

        logger.info(
            f"[{citation_key}] Draft not accepted (score: {validation_result.score:.2f}), "
            f"starting refinement (max {self.max_refinement_attempts} attempts)..."
        )
        
        while refinement_attempts < self.max_refinement_attempts:
            refinement_attempts += 1
            total_attempts += 1
            
            logger.info(
                f"[{citation_key}] Refinement attempt {refinement_attempts}/{self.max_refinement_attempts}"
            )
            
            # Get issues for refinement
            issues = validation_result.errors + validation_result.warnings
            
            # Check if repetition is the main issue
            has_repetition_issue = any(
                "repetition" in issue.lower() 
                for issue in issues
            )
            
            # For repetition issues, add explicit instructions
            if has_repetition_issue:
                repetition_instructions = (
                    "CRITICAL: The current summary has severe repetition issues. "
                    "You MUST eliminate all repeated sentences, phrases, and paragraphs. "
                    "Each idea should be expressed only once. "
                    "If you find yourself repeating content, remove the duplicates entirely. "
                    "Focus on variety and uniqueness in your wording."
                )
                # Prepend repetition instructions to issues
                issues = [repetition_instructions] + issues
            
            # Use simple prompt for later attempts (fallback strategy)
            use_simple = refinement_attempts > 1
            
            # Refine summary
            try:
                refined = self.refine_summary(
                    current_summary, 
                    issues, 
                    context, 
                    citation_key,
                    progress_callback=progress_callback,
                    attempt_number=refinement_attempts,
                    max_attempts=self.max_refinement_attempts,
                    use_simple_prompt=use_simple
                )
            except Exception as e:
                logger.warning(f"[{citation_key}] Refinement attempt {refinement_attempts} failed: {e}")
                if refinement_attempts < self.max_refinement_attempts:
                    continue  # Try next attempt
                else:
                    break  # Give up
            
            # Apply post-generation deduplication before validation
            # Use more aggressive settings if repetition was an issue
            from infrastructure.llm.validation.repetition import deduplicate_sections
            has_repetition_issue = any(
                "repetition" in issue.lower() 
                for issue in issues
            )
            
            if has_repetition_issue:
                # More aggressive deduplication for repetition issues
                refined = deduplicate_sections(
                    refined,
                    max_repetitions=1,
                    mode="aggressive",
                    similarity_threshold=0.75,  # Lower threshold to catch more duplicates
                    min_content_preservation=0.5  # Allow more content removal
                )
            else:
                # Standard deduplication
                refined = deduplicate_sections(
                    refined,
                    max_repetitions=1,
                    mode="aggressive",
                    similarity_threshold=0.85,
                    min_content_preservation=0.7
                )
            
            # Apply formatting fixes after deduplication
            refined = self._fix_markdown_formatting(refined)
            
            # Validate refined summary
            should_accept, validation_result = self.validate_and_accept(
                refined, context, pdf_text, paper_title, citation_key, progress_callback=progress_callback
            )
            
            if should_accept:
                overall_time = time.time() - overall_start_time
                logger.info(
                    f"[{citation_key}] Summarization completed in {overall_time:.2f}s "
                    f"(accepted after {refinement_attempts} refinement(s), {total_attempts} total attempts)"
                )
                return refined, validation_result, total_attempts
            
            # Check for hard failure - don't continue refining
            if validation_result.has_hard_failure():
                logger.warning(f"[{citation_key}] Hard failure detected, stopping refinement")
                break
            
            current_summary = refined
        
        # Return best attempt even if not perfect
        # Apply formatting fixes before returning
        current_summary = self._fix_markdown_formatting(current_summary)
        
        overall_time = time.time() - overall_start_time
        logger.warning(
            f"[{citation_key}] Summarization completed in {overall_time:.2f}s "
            f"(not fully validated after {total_attempts} attempts, score: {validation_result.score:.2f})"
        )
        return current_summary, validation_result, total_attempts

    def _determine_fallback_strategy(
        self,
        validation_result: "ValidationResult",
        attempt_number: int
    ) -> Optional[Dict[str, Any]]:
        """Determine appropriate fallback strategy based on validation failures.

        Returns a strategy dictionary with 'name', 'type', and parameters,
        or None if no suitable strategy available.
        """
        errors = " ".join(validation_result.errors).lower()
        warnings = " ".join(validation_result.warnings).lower()

        # Strategy 1: Severe repetition (highest priority)
        if any("severe repetition" in error for error in validation_result.errors):
            return {
                "name": "severe_repetition_regeneration",
                "type": "regeneration",
                "temperature": 0.1,
                "max_tokens": 3000,
                "aggressive_deduplication": True,
                "description": "Regenerate with very low temperature and aggressive deduplication"
            }

        # Strategy 2: Standard repetition
        if any("repetition" in error.lower() for error in validation_result.errors):
            if attempt_number == 1:
                return {
                    "name": "repetition_refinement",
                    "type": "refinement",
                    "temperature": 0.2,
                    "simplified_prompt": True,
                    "description": "Refine with lower temperature and simplified prompt"
                }
            else:
                return {
                    "name": "repetition_regeneration",
                    "type": "regeneration",
                    "temperature": 0.15,
                    "max_tokens": 2500,
                    "description": "Regenerate with low temperature for repetition"
                }

        # Strategy 3: Low information density
        if any("low information density" in error.lower() for error in validation_result.errors):
            return {
                "name": "information_density_enhancement",
                "type": "refinement",
                "temperature": 0.3,
                "enhanced_specificity_prompt": True,
                "description": "Refine with enhanced specificity requirements"
            }

        # Strategy 4: Hallucination issues
        if any("hallucination" in error.lower() for error in validation_result.errors):
            return {
                "name": "hallucination_correction",
                "type": "refinement",
                "temperature": 0.25,
                "strict_evidence_prompt": True,
                "description": "Refine with strict evidence requirements"
            }

        # Strategy 5: Length issues
        if any("word count" in error.lower() or "length" in error.lower() for error in validation_result.errors):
            if validation_result.score < 0.3:  # Very low score
                return {
                    "name": "length_regeneration",
                    "type": "regeneration",
                    "temperature": 0.4,
                    "max_tokens": 4000,  # Allow more tokens
                    "description": "Regenerate with higher token limit"
                }

        # Strategy 6: Generic quality issues (catch-all)
        if validation_result.score < 0.5 and attempt_number < 3:
            return {
                "name": "quality_improvement",
                "type": "refinement",
                "temperature": 0.35,
                "comprehensive_prompt": True,
                "description": "Refine with comprehensive quality instructions"
            }

        return None

    def _apply_fallback_strategy(
        self,
        strategy: Dict[str, Any],
        current_summary: str,
        context: "SummarizationContext",
        pdf_text: str,
        paper_title: str,
        citation_key: str,
        progress_callback: Optional[Callable[["SummarizationProgressEvent"], None]],
        validation_result: Optional["ValidationResult"] = None
    ) -> str:
        """Apply the specified fallback strategy to improve the summary."""
        strategy_type = strategy["type"]

        if strategy_type == "regeneration":
            # Generate new summary with different parameters
            logger.info(f"[{citation_key}] Regenerating summary with {strategy['name']}")

            # Apply aggressive deduplication if requested
            if strategy.get("aggressive_deduplication", False):
                from infrastructure.llm.validation.repetition import deduplicate_sections
                current_summary = deduplicate_sections(
                    current_summary,
                    max_repetitions=1,
                    mode="aggressive",
                    similarity_threshold=0.75,
                    min_content_preservation=0.5
                )

            # Generate new draft with strategy parameters
            metadata = {
                "citation_key": citation_key,
                "title": paper_title,
                "temperature": strategy.get("temperature", 0.4),
                "max_tokens": strategy.get("max_tokens", 3000)
            }

            new_summary = self._generate_draft_single_stage(
                context,
                metadata,
                progress_callback=progress_callback,
                temperature=strategy.get("temperature", 0.4),
                max_tokens=strategy.get("max_tokens", 3000)
            )

            return new_summary

        elif strategy_type == "refinement":
            # Refine existing summary with specific guidance
            logger.info(f"[{citation_key}] Refining summary with {strategy['name']}")

            # Build refinement prompt with strategy-specific guidance
            refinement_guidance = self._build_strategy_guidance(strategy)
            issues = validation_result.errors + validation_result.warnings

            # Create issues list from validation result
            issues = validation_result.errors + validation_result.warnings

            refined = self.refine_summary(
                current_summary,
                issues,
                context,
                citation_key,
                progress_callback=progress_callback,
                attempt_number=1,
                max_attempts=1,
                temperature=strategy.get("temperature", 0.3),
                refinement_guidance=refinement_guidance
            )

            return refined

        else:
            logger.warning(f"[{citation_key}] Unknown strategy type: {strategy_type}")
            return current_summary

    def _build_strategy_guidance(self, strategy: Dict[str, Any]) -> str:
        """Build specific guidance text for refinement based on strategy."""
        guidance_parts = []

        if strategy.get("simplified_prompt"):
            guidance_parts.append(
                "Simplify the summary structure. Focus on the 3-5 most important points. "
                "Remove redundant explanations and examples."
            )

        if strategy.get("enhanced_specificity_prompt"):
            guidance_parts.append(
                "Add more specific details and numerical values. Replace vague terms like "
                "'high performance' with concrete metrics like '94.2% accuracy'. Include "
                "exact numbers, percentages, and quantitative results from the paper."
            )

        if strategy.get("strict_evidence_prompt"):
            guidance_parts.append(
                "Ensure every claim is supported by direct evidence from the paper. "
                "Add quotes or specific references for all key findings. Remove any "
                "inferences or assumptions not explicitly stated."
            )

        if strategy.get("comprehensive_prompt"):
            guidance_parts.append(
                "Provide a more comprehensive analysis covering methodology, results, "
                "and implications. Ensure balanced coverage of all major sections. "
                "Add specific technical details and concrete examples."
            )

        return " ".join(guidance_parts) if guidance_parts else "Improve overall quality and accuracy."
