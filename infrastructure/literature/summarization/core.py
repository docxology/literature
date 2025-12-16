"""Core summarization engine.

This module provides the main SummarizationEngine that orchestrates
the multi-stage summarization workflow.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from infrastructure.core.logging_utils import get_logger
from infrastructure.core.exceptions import LLMConnectionError, FileOperationError, ContextLimitError
from infrastructure.literature.sources import SearchResult
from infrastructure.validation.pdf_validator import extract_text_from_pdf, PDFValidationError

from infrastructure.literature.summarization.models import SummarizationResult, SummarizationProgressEvent
from infrastructure.literature.summarization.context_extractor import ContextExtractor
from infrastructure.literature.summarization.prompt_builder import SummarizationPromptBuilder
from infrastructure.literature.summarization.multi_stage_summarizer import MultiStageSummarizer
from infrastructure.literature.summarization.validator import SummaryQualityValidator
from infrastructure.literature.summarization.pdf_processor import PDFProcessor, PrioritizedPDFText
from infrastructure.literature.summarization.extractor import TextExtractor
from infrastructure.literature.summarization.utils import detect_model_size
from infrastructure.literature.summarization.chunker import PDFChunker

if TYPE_CHECKING:
    from infrastructure.llm.core.client import LLMClient
    from infrastructure.literature.core.config import LiteratureConfig

logger = get_logger(__name__)


class SummarizationEngine:
    """Main interface for paper summarization with multi-stage generation.

    Orchestrates the summarization workflow including:
    - PDF text extraction with prioritization
    - Context extraction and structuring
    - Multi-stage summary generation (draft + refine)
    - Quality validation
    - Result management
    - Library index updates (classification storage)

    Attributes:
        llm_client: LLM client for summary generation.
        context_extractor: Context extraction instance.
        prompt_builder: Prompt builder instance.
        multi_stage_summarizer: Multi-stage summarizer instance.
        validator: Quality validator instance.
        pdf_processor: PDF processor instance.
        text_extractor: Text extraction instance.
        config: Literature configuration (used for library index updates).
        max_pdf_chars: Maximum characters to send to LLM.
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        quality_validator: Optional[SummaryQualityValidator] = None,
        context_extractor: Optional[ContextExtractor] = None,
        prompt_builder: Optional[SummarizationPromptBuilder] = None,
        max_pdf_chars: Optional[int] = None,
        config: Optional["LiteratureConfig"] = None
    ):
        """Initialize summarization engine.

        Args:
            llm_client: Configured LLM client for summary generation.
            quality_validator: Quality validator instance (created if None).
            context_extractor: Context extractor instance (created if None).
            prompt_builder: Prompt builder instance (created if None).
            max_pdf_chars: Maximum PDF characters to send to LLM.
                          Defaults to 200000 (200K) or LLM_MAX_INPUT_LENGTH env var.
            config: Optional literature configuration. If None, will be created from environment.
                    Required for library index updates (e.g., classification storage).
        """
        import os
        from infrastructure.literature.core.config import LiteratureConfig
        
        self.llm_client = llm_client
        self.validator = quality_validator or SummaryQualityValidator()
        self.context_extractor = context_extractor or ContextExtractor()
        self.prompt_builder = prompt_builder or SummarizationPromptBuilder(llm_client=llm_client)
        self.pdf_processor = PDFProcessor()
        self.text_extractor = TextExtractor()
        self._last_ref_info: Optional[Dict[str, Any]] = None  # Store ref_info for metadata
        
        # Store config (create from environment if not provided)
        self.config = config or LiteratureConfig.from_env()
        
        # Get max_pdf_chars from parameter, environment variable, or model-aware default
        if max_pdf_chars is not None:
            self.max_pdf_chars = max_pdf_chars
        else:
            env_limit = os.getenv('LLM_MAX_INPUT_LENGTH')
            if env_limit:
                try:
                    self.max_pdf_chars = int(env_limit)
                except ValueError:
                    self.max_pdf_chars = self._get_model_aware_limit()
            else:
                self.max_pdf_chars = self._get_model_aware_limit()
        
        # Create multi-stage summarizer (with two-stage support)
        self.multi_stage_summarizer = MultiStageSummarizer(
            llm_client=llm_client,
            validator=self.validator,
            prompt_builder=self.prompt_builder,
            max_refinement_attempts=2,
            two_stage_enabled=None,  # Auto-detect from env
            chunk_size=None,  # Auto-detect from env
            chunk_overlap=None,  # Auto-detect from env
            two_stage_threshold=None  # Auto-detect from env
        )

    def summarize_paper(
        self,
        result: SearchResult,
        pdf_path: Path,
        max_retries: int = 2,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None
    ) -> SummarizationResult:
        """Generate summary for a single paper with quality validation.
        
        This method orchestrates the summarization workflow:
        1. PDF text extraction (from pre-extracted files or on-the-fly)
        2. Context extraction (abstract, intro, conclusion, key terms)
        3. Multi-stage summary generation (draft + optional refinement)
        4. Post-processing deduplication
        5. Quality validation
        
        The summary is always included in the result (even if validation fails)
        to ensure it can be saved for review. Validation metadata is available
        in the result for transparency.

        Args:
            result: Search result with paper metadata (title, authors, year, etc.).
            pdf_path: Path to PDF file for summarization.
            max_retries: Maximum retry attempts for LLM connection failures
                        (refinement attempts are handled internally by MultiStageSummarizer).
            progress_callback: Optional callback function for real-time progress events.
                            Receives SummarizationProgressEvent instances at each stage.

        Returns:
            SummarizationResult containing:
            - summary_text: Generated summary (always present, even if validation failed)
            - success: Whether validation passed (score >= 0.5)
            - quality_score: Validation score (0.0 to 1.0)
            - validation_errors: List of validation issues (if any)
            - Metadata: input/output sizes, generation time, attempts
            
        Note:
            Summaries are always saved with validation metadata, allowing review
            of rejected summaries. Use save_summary() to persist to disk.
        """
        citation_key = pdf_path.stem
        start_time = time.time()
        
        # Initialize variables used in error paths
        classification = None
        
        # CRITICAL: Clear context before processing each paper to prevent cross-paper contamination
        logger.info(
            f"[{citation_key}] Clearing LLM context before summarization",
            extra={
                "messages_before": len(self.llm_client.context.messages),
                "tokens_before": self.llm_client.context.estimated_tokens
            }
        )
        self.llm_client.context.clear()
        
        # Helper function to emit progress events
        def emit_progress(stage: str, status: str, message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
            if progress_callback:
                event = SummarizationProgressEvent(
                    citation_key=citation_key,
                    stage=stage,
                    status=status,
                    message=message,
                    metadata=metadata or {}
                )
                progress_callback(event)
        
        # Load extracted text from file (preferred) or extract on-the-fly (fallback)
        emit_progress("pdf_extraction", "started", f"Loading text for {pdf_path.name}")
        extraction_start = time.time()
        
        # Try to load from extracted_text directory first
        pdf_text = self.text_extractor.load_extracted_text(citation_key)
        
        if pdf_text:
            # Successfully loaded from file
            extraction_time = time.time() - extraction_start
            input_chars = len(pdf_text)
            input_words = len(pdf_text.split())
            
            logger.info(
                f"[{citation_key}] Loaded extracted text from file: "
                f"{input_chars:,} chars, {input_words:,} words ({extraction_time:.2f}s)"
            )
            
            emit_progress(
                "pdf_extraction",
                "completed",
                f"Loaded {input_chars:,} chars, {input_words:,} words from extracted text file",
                {"chars": input_chars, "words": input_words, "time": extraction_time, "source": "file"}
            )
        else:
            # Fallback: extract on-the-fly (with warning)
            logger.warning(
                f"[{citation_key}] No extracted text file found at "
                f"data/extracted_text/{citation_key}.txt. "
                f"Extracting on-the-fly (recommended: run extraction step first)."
            )
            
            emit_progress("pdf_extraction", "started", f"Extracting text from {pdf_path.name} (on-the-fly)")
            
            try:
                file_size = pdf_path.stat().st_size
                logger.debug(f"Starting on-the-fly PDF text extraction for {pdf_path.name} ({file_size:,} bytes)")
                
                # Extract full text without truncation (max_pdf_chars is set very high)
                # Use extract_text_from_pdf directly to ensure full paper is loaded
                pdf_text = extract_text_from_pdf(pdf_path)
                extraction_time = time.time() - extraction_start
                
                if not pdf_text or len(pdf_text.strip()) < 100:
                    error_msg = "Insufficient text extracted from PDF (less than 100 characters). PDF extraction may have failed or yielded insufficient text."
                    emit_progress("pdf_extraction", "failed", error_msg)
                    logger.warning(
                        f"PDF extraction yielded insufficient text: "
                        f"{len(pdf_text) if pdf_text else 0} chars, "
                        f"{len(pdf_text.split()) if pdf_text else 0} words from {pdf_path.name}"
                    )
                    return SummarizationResult(
                        citation_key=citation_key,
                        success=False,
                        input_chars=len(pdf_text) if pdf_text else 0,
                        input_words=len(pdf_text.split()) if pdf_text else 0,
                        generation_time=time.time() - start_time,
                        attempts=1,
                        error=error_msg
                    )
                
                input_chars = len(pdf_text)
                input_words = len(pdf_text.split())
                
                logger.info(
                    f"PDF text extracted: {input_chars:,} chars, {input_words:,} words (full paper)"
                )
                
                emit_progress(
                    "pdf_extraction",
                    "completed",
                    f"Extracted {input_chars:,} chars, {input_words:,} words (on-the-fly)",
                    {"chars": input_chars, "words": input_words, "time": extraction_time, "source": "on-the-fly"}
                )
            except PDFValidationError as e:
                extraction_time = time.time() - extraction_start
                emit_progress("pdf_extraction", "failed", f"PDF extraction failed: {e}")
                logger.error(f"PDF text extraction failed for {pdf_path.name}: {e}")
                return SummarizationResult(
                    citation_key=citation_key,
                    success=False,
                    input_chars=0,
                    input_words=0,
                    generation_time=time.time() - start_time,
                    attempts=1,
                    error=f"PDF extraction failed: {e}"
                )
        
        # Ensure we have valid text at this point
        if not pdf_text or len(pdf_text.strip()) < 100:
            error_msg = "No valid text available for summarization"
            return SummarizationResult(
                citation_key=citation_key,
                success=False,
                input_chars=len(pdf_text) if pdf_text else 0,
                input_words=len(pdf_text.split()) if pdf_text else 0,
                generation_time=time.time() - start_time,
                attempts=1,
                error=error_msg
            )
        
        input_chars = len(pdf_text)
        input_words = len(pdf_text.split())

        # Extract structured context
        emit_progress("context_extraction", "started", "Extracting structured context")
        context_start_time = time.time()
        try:
            context = self.context_extractor.create_summarization_context(
                pdf_text=pdf_text,
                title=result.title,
                max_chars=self.max_pdf_chars
            )
            context_time = time.time() - context_start_time
            
            # Save context summary to file for debugging
            # Note: Extracted text is already saved in data/extracted_text/ by extraction step
            try:
                debug_dir = Path("data/summaries") / "_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                
                # Save context summary
                context_summary = f"""# Context Summary for {citation_key}

## Paper Identity
- Title: {context.title}
- Full Text Length: {len(context.full_text):,} chars
- Full Text Words: {len(context.full_text.split()):,} words

## Extracted Sections
- Abstract: {len(context.abstract)} chars
- Introduction: {len(context.introduction)} chars
- Conclusion: {len(context.conclusion)} chars
- Methods: {len(context.methods) if context.methods else 0} chars
- Results: {len(context.results) if context.results else 0} chars
- Discussion: {len(context.discussion) if context.discussion else 0} chars

## Key Terms
{', '.join(context.key_terms) if context.key_terms else 'None'}

## Equations Found
{len(context.equations)} equations extracted

## Full Text Preview (first 500 chars)
{context.full_text[:500]}...
"""
                context_summary_path = debug_dir / f"{citation_key}_context_summary.md"
                context_summary_path.write_text(context_summary, encoding='utf-8')
                logger.debug(f"[{citation_key}] Saved context summary: {context_summary_path}")
            except Exception as save_error:
                logger.debug(f"[{citation_key}] Failed to save debug files: {save_error}")
            
            logger.info(
                f"[{citation_key}] Context extracted in {context_time:.2f}s: "
                f"abstract={len(context.abstract)} chars, intro={len(context.introduction)} chars, "
                f"conclusion={len(context.conclusion)} chars, key_terms={len(context.key_terms)}, "
                f"full_text={len(context.full_text):,} chars (COMPLETE)"
            )
            emit_progress(
                "context_extraction",
                "completed",
                f"Context extracted: abstract={len(context.abstract)} chars, intro={len(context.introduction)} chars, conclusion={len(context.conclusion)} chars, key_terms={len(context.key_terms)}, full_text={len(context.full_text):,} chars",
                {
                    "abstract_chars": len(context.abstract),
                    "intro_chars": len(context.introduction),
                    "conclusion_chars": len(context.conclusion),
                    "key_terms_count": len(context.key_terms),
                    "full_text_chars": len(context.full_text),
                    "time": context_time
                }
            )
        except Exception as e:
            context_time = time.time() - context_start_time
            import traceback
            error_type = type(e).__name__
            error_context = {
                "stage": "context_extraction",
                "elapsed_time": f"{context_time:.2f}s",
                "input_chars": f"{input_chars:,}",
                "input_words": f"{input_words:,}",
                "error_type": error_type
            }
            emit_progress("context_extraction", "failed", f"Context extraction failed: {e}")
            logger.error(
                f"[{citation_key}] Context extraction failed after {context_time:.2f}s: {e}\n"
                f"Context: {error_context}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            return SummarizationResult(
                citation_key=citation_key,
                success=False,
                input_chars=input_chars,
                input_words=input_words,
                generation_time=time.time() - start_time,
                attempts=1,
                error=f"Context extraction failed: {e}"
            )
        
        # Prepare metadata (include citation_key for proper logging)
        metadata = {
            'title': result.title,
            'authors': result.authors or [],
            'year': str(result.year) if result.year else '',
            'source': result.source or 'unknown',
            'citation_key': citation_key
        }
        
        logger.info(f"[{citation_key}] Starting multi-stage summarization...")
        generation_start_time = time.time()
        
        # Generate summary with multi-stage approach (with retry logic for LLM failures)
        attempts = 0
        summary = None
        validation_result = None
        total_attempts = 0
        base_backoff = 2.0  # Base backoff time in seconds
        
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            
            # Check LLM connection before each attempt
            if attempt > 0:
                # Exponential backoff: 2^attempt seconds (2s, 4s, 8s, etc.)
                backoff_time = base_backoff ** attempt
                logger.info(
                    f"[{citation_key}] Retry attempt {attempts}/{max_retries + 1} "
                    f"(waiting {backoff_time:.1f}s with exponential backoff)..."
                )
                time.sleep(backoff_time)
                
                # Verify LLM connection before retry
                logger.debug(f"[{citation_key}] Verifying LLM connection before retry...")
                if not self.llm_client.check_connection():
                    logger.warning(f"[{citation_key}] LLM connection check failed before retry {attempts}")
                    if attempt < max_retries:
                        continue  # Will retry again
                    else:
                        logger.error(f"[{citation_key}] LLM connection unavailable after {attempts} attempts")
                        return SummarizationResult(
                            citation_key=citation_key,
                            success=False,
                            input_chars=input_chars,
                            input_words=input_words,
                            generation_time=time.time() - start_time,
                            attempts=attempts,
                            error=f"LLM connection unavailable after {attempts} attempts"
                        )
                logger.debug(f"[{citation_key}] LLM connection verified before retry")
            
            try:
                summary, validation_result, total_attempts = self.multi_stage_summarizer.summarize_with_refinement(
                    context=context,
                    pdf_text=pdf_text,
                    metadata=metadata,
                    citation_key=citation_key,
                    progress_callback=progress_callback
                )
                
                # Validate summary is non-empty before proceeding
                if not summary or not summary.strip():
                    raise ValueError(f"Empty summary received for {citation_key}")
                
                generation_time = time.time() - generation_start_time
                summary_words = len(summary.split())
                logger.info(
                    f"[{citation_key}] Summary generation completed in {generation_time:.2f}s "
                    f"({total_attempts} total attempts, {summary_words} words, "
                    f"quality score: {validation_result.score:.2f})"
                )
                # Success - break out of retry loop
                break
            except LLMConnectionError as e:
                logger.warning(
                    f"[{citation_key}] LLM connection error on attempt {attempts}: {e}\n"
                    f"Context: prompt_size={len(prompt) if 'prompt' in locals() else 'unknown'}, "
                    f"elapsed={time.time() - generation_start_time:.2f}s"
                )
                if attempt < max_retries:
                    logger.info(f"[{citation_key}] Will retry with exponential backoff (attempt {attempts + 1}/{max_retries + 1})...")
                    continue
                else:
                    logger.error(f"[{citation_key}] LLM connection failed after {attempts} attempts")
                    return SummarizationResult(
                        citation_key=citation_key,
                        success=False,
                        input_chars=input_chars,
                        input_words=input_words,
                        generation_time=time.time() - start_time,
                        attempts=attempts,
                        error=f"LLM connection error after {attempts} attempts: {e}"
                    )
            except ContextLimitError as e:
                # Context limit error - try to retry with two-stage mode if available
                logger.warning(
                    f"[{citation_key}] Context limit error on attempt {attempts}: {e}\n"
                    f"Context: {getattr(e, 'context', {})}, "
                    f"elapsed={time.time() - generation_start_time:.2f}s"
                )
                
                # Check if two-stage mode is available and enabled
                two_stage_enabled = getattr(self.multi_stage_summarizer, 'two_stage_enabled', False)
                auto_two_stage = getattr(self.multi_stage_summarizer, 'auto_two_stage_fallback', True)
                
                if two_stage_enabled and auto_two_stage and attempt == 1:
                    # First attempt failed with context error - retry with two-stage mode
                    logger.info(
                        f"[{citation_key}] Context limit exceeded. Retrying with two-stage mode "
                        f"(attempt {attempts + 1}/{max_retries + 1})..."
                    )
                    # Force two-stage mode by modifying context temporarily
                    # The multi_stage_summarizer should handle this automatically, but we'll
                    # ensure it's triggered by clearing context and retrying
                    try:
                        self.llm_client.context.clear()
                        # Retry - the generate_draft method should detect size and use two-stage
                        summary, validation_result, total_attempts = self.multi_stage_summarizer.summarize_with_refinement(
                            context=context,
                            pdf_text=pdf_text,
                            metadata=metadata,
                            citation_key=citation_key,
                            progress_callback=progress_callback
                        )
                        
                        if summary and summary.strip():
                            generation_time = time.time() - generation_start_time
                            summary_words = len(summary.split())
                            logger.info(
                                f"[{citation_key}] Summary generation completed in {generation_time:.2f}s "
                                f"using two-stage mode ({total_attempts} total attempts, {summary_words} words, "
                                f"quality score: {validation_result.score:.2f})"
                            )
                            break
                    except Exception as retry_error:
                        logger.warning(
                            f"[{citation_key}] Two-stage retry also failed: {retry_error}. "
                            f"Continuing with normal retry logic..."
                        )
                        if attempt < max_retries:
                            logger.info(f"[{citation_key}] Will retry with exponential backoff (attempt {attempts + 1}/{max_retries + 1})...")
                            continue
                        else:
                            return SummarizationResult(
                                citation_key=citation_key,
                                success=False,
                                input_chars=input_chars,
                                input_words=input_words,
                                generation_time=time.time() - start_time,
                                attempts=attempts,
                                error=f"Context limit error after {attempts} attempts (including two-stage retry): {e}"
                            )
                elif attempt < max_retries:
                    logger.info(f"[{citation_key}] Will retry with exponential backoff (attempt {attempts + 1}/{max_retries + 1})...")
                    continue
                else:
                    logger.error(f"[{citation_key}] Context limit error after {attempts} attempts")
                    return SummarizationResult(
                        citation_key=citation_key,
                        success=False,
                        input_chars=input_chars,
                        input_words=input_words,
                        generation_time=time.time() - start_time,
                        attempts=attempts,
                        error=f"Context limit error after {attempts} attempts: {e}. "
                              f"Consider enabling two-stage mode (LITERATURE_TWO_STAGE_ENABLED=true) "
                              f"or using a model with larger context window."
                    )
            except ValueError as e:
                # Empty response or validation error
                logger.error(
                    f"[{citation_key}] Summary validation error on attempt {attempts}: {e}\n"
                    f"Context: elapsed={time.time() - generation_start_time:.2f}s"
                )
                if attempt < max_retries:
                    logger.info(f"[{citation_key}] Will retry with exponential backoff (attempt {attempts + 1}/{max_retries + 1})...")
                    continue
                else:
                    return SummarizationResult(
                        citation_key=citation_key,
                        success=False,
                        input_chars=input_chars,
                        input_words=input_words,
                        generation_time=time.time() - start_time,
                        attempts=attempts,
                        error=f"Summary validation failed after {attempts} attempts: {e}"
                    )
            except Exception as e:
                import traceback
                logger.error(
                    f"[{citation_key}] Summary generation failed on attempt {attempts}: {e}\n"
                    f"Context: elapsed={time.time() - generation_start_time:.2f}s\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                if attempt < max_retries:
                    logger.info(f"[{citation_key}] Will retry with exponential backoff (attempt {attempts + 1}/{max_retries + 1})...")
                    continue
                else:
                    return SummarizationResult(
                        citation_key=citation_key,
                        success=False,
                        input_chars=input_chars,
                        input_words=input_words,
                        generation_time=time.time() - start_time,
                        attempts=attempts,
                        error=f"Summary generation failed after {attempts} attempts: {e}"
                    )
        
        # If we exhausted retries without success, return failure
        if summary is None or validation_result is None:
            return SummarizationResult(
                citation_key=citation_key,
                success=False,
                input_chars=input_chars,
                input_words=input_words,
                generation_time=time.time() - start_time,
                attempts=attempts,
                error="Summary generation failed after all retry attempts"
            )
        
        generation_time = time.time() - start_time
        output_words = len(summary.split())
        
        # Check if summary should be accepted
        if validation_result.has_hard_failure():
            logger.warning(
                f"[{citation_key}] Summary rejected due to hard failure: {validation_result.errors[:3]}"
            )
            # Record quality metrics
        from infrastructure.literature.summarization.quality_tracker import QualityTracker, QualityMetrics
        tracker = QualityTracker()
        metrics = QualityMetrics(
            citation_key=citation_key,
            timestamp=time.time(),
            success=False,
            quality_score=validation_result.score,
            validation_errors=validation_result.errors,
            validation_warnings=validation_result.warnings,
            input_chars=input_chars,
            input_words=input_words,
            output_words=output_words,
            generation_time=generation_time,
            attempts=total_attempts,
            paper_classification=getattr(classification, 'category', None) if classification else None
        )
        tracker.record_operation(metrics)

        # Always include summary_text even when validation fails (for saving)
        return SummarizationResult(
                citation_key=citation_key,
                success=False,
                summary_text=summary,  # Always include summary for saving
                input_chars=input_chars,
                input_words=input_words,
                output_words=output_words,
                generation_time=generation_time,
                attempts=total_attempts,
                quality_score=validation_result.score,
                validation_errors=validation_result.errors,
                error=f"Hard failure: {', '.join(validation_result.errors[:3])}"
            )
        
        # Accept if validation passed or score is acceptable
        success = validation_result.is_valid or validation_result.score >= 0.5

        # Record quality metrics
        from infrastructure.literature.summarization.quality_tracker import QualityTracker, QualityMetrics
        tracker = QualityTracker()
        metrics = QualityMetrics(
            citation_key=citation_key,
            timestamp=time.time(),
            success=success,
            quality_score=validation_result.score,
            validation_errors=validation_result.errors,
            validation_warnings=validation_result.warnings,
            input_chars=input_chars,
            input_words=input_words,
            output_words=output_words,
            generation_time=generation_time,
            attempts=total_attempts,
            paper_classification=getattr(classification, 'category', None) if classification else None
        )
        tracker.record_operation(metrics)

        if success:
            logger.info(
                f"[{citation_key}] Summary accepted: {output_words} words, "
                f"quality score: {validation_result.score:.2f}, "
                f"compression ratio: {output_words/max(1, input_words):.2%}"
            )
        else:
            logger.warning(
                f"[{citation_key}] Summary rejected: quality score {validation_result.score:.2f} "
                f"below threshold (0.5) - will still be saved"
            )
        
        # Pass 2: Extract key claims and quotes
        logger.info(f"[{citation_key}] Starting Pass 2: Claims and quotes extraction")
        claims_quotes_text = None
        try:
            claims_quotes_text = self._extract_claims_and_quotes(
                pdf_text=pdf_text,
                result=result,
                citation_key=citation_key,
                progress_callback=progress_callback
            )
            logger.info(f"[{citation_key}] Pass 2 completed: claims/quotes extraction")
        except Exception as e:
            logger.warning(f"[{citation_key}] Pass 2 (claims/quotes) failed: {e}")
            claims_quotes_text = f"## Key Claims and Hypotheses\n\nExtraction failed: {e}\n\n## Important Quotes\n\nExtraction failed."
        
        # Pass 3: Analyze methods and tools
        logger.info(f"[{citation_key}] Starting Pass 3: Methods and tools analysis")
        methods_tools_text = None
        try:
            methods_tools_text = self._analyze_methods_and_tools(
                pdf_text=pdf_text,
                result=result,
                citation_key=citation_key,
                progress_callback=progress_callback
            )
            logger.info(f"[{citation_key}] Pass 3 completed: methods/tools analysis")
        except Exception as e:
            logger.warning(f"[{citation_key}] Pass 3 (methods/tools) failed: {e}")
            methods_tools_text = f"## Algorithms and Methodologies\n\nAnalysis failed: {e}\n\n## Software Frameworks and Libraries\n\nAnalysis failed.\n\n## Datasets\n\nAnalysis failed.\n\n## Evaluation Metrics\n\nAnalysis failed.\n\n## Software Tools and Platforms\n\nAnalysis failed."
        
        # Pass 4: Classify paper
        logger.info(f"[{citation_key}] Starting Pass 4: Paper classification")
        classification = None
        try:
            classification = self._classify_paper(
                pdf_text=pdf_text,
                result=result,
                citation_key=citation_key,
                progress_callback=progress_callback
            )
            if classification:
                logger.info(f"[{citation_key}] Pass 4 completed: classification = {classification.category}")
            else:
                logger.warning(f"[{citation_key}] Pass 4 (classification) returned None")
        except Exception as e:
            logger.warning(f"[{citation_key}] Pass 4 (classification) failed: {e}")
        
        # Always include summary_text even when validation fails (for saving)
        return SummarizationResult(
            citation_key=citation_key,
            success=success,
            summary_text=summary,  # Always include summary for saving
            claims_quotes_text=claims_quotes_text,
            methods_tools_text=methods_tools_text,
            classification=classification,
            input_chars=input_chars,
            input_words=input_words,
            output_words=output_words,
            generation_time=generation_time,
            attempts=total_attempts,
            quality_score=validation_result.score,
            validation_errors=validation_result.errors if not success else []
        )
    
    def _get_model_aware_limit(self) -> int:
        """Get model-aware max_pdf_chars limit based on model size.
        
        Always returns a very large limit to ensure full paper is loaded.
        The actual limit is set high enough that truncation should not occur
        for typical academic papers.
        
        Returns:
            Maximum PDF characters to send to LLM (set very high to allow full papers).
        """
        # Always use a very high limit to ensure full paper is loaded
        # This allows the full paper context to be available to the LLM
        return 1000000  # 1M chars - effectively no limit for typical papers
    
    @property
    @property
    def quality_validator(self) -> SummaryQualityValidator:
        """Get quality validator instance.
        
        Property alias for `self.validator` (backward compatibility).
        
        Returns:
            SummaryQualityValidator instance used for quality validation.
        """
        return self.validator
    
    def _analyze_references(self, pdf_text: str) -> Dict[str, Any]:
        """Analyze references in PDF text.
        
        Args:
            pdf_text: Extracted PDF text.
            
        Returns:
            Dictionary with reference analysis:
            {
                'count': Optional[int] - Number of references found,
                'section_found': bool - Whether references section exists,
                'section_start': Optional[int] - Character position of references section
            }
        """
        ref_info = {
            'count': None,
            'section_found': False,
            'section_start': None
        }
        
        # Look for references section header
        ref_section_patterns = [
            r'^\s*(?:References?|Bibliography|Works\s+Cited)\s*$',
            r'^#+\s*(?:References?|Bibliography)',
            r'\\begin\{thebibliography\}',
            r'\\bibliography'
        ]
        
        for pattern in ref_section_patterns:
            match = re.search(pattern, pdf_text, re.IGNORECASE | re.MULTILINE)
            if match:
                ref_info['section_found'] = True
                ref_info['section_start'] = match.start()
                break
        
        # Count references using multiple patterns
        citation_patterns = [
            (r'\[(\d+)\]', lambda m: int(m.group(1))),  # [1], [2], etc.
            (r'\\cite\{[^}]+\}', lambda m: 1),  # \cite{key} - count occurrences
            (r'\\bibitem', lambda m: 1),  # \bibitem entries
        ]
        
        max_ref_num = 0
        citation_count = 0
        
        for pattern, extractor in citation_patterns:
            matches = list(re.finditer(pattern, pdf_text))
            if matches:
                if pattern == r'\[(\d+)\]':
                    # Extract numbers and find max
                    numbers = [int(m.group(1)) for m in matches if m.group(1).isdigit()]
                    if numbers:
                        max_ref_num = max(max_ref_num, max(numbers))
                else:
                    # Count occurrences
                    citation_count += len(matches)
        
        # Use max reference number if found, otherwise use citation count
        if max_ref_num > 0:
            ref_info['count'] = max_ref_num
        elif citation_count > 0:
            ref_info['count'] = citation_count
        
        return ref_info
    
    def _generate_summary(
        self,
        result: SearchResult,
        pdf_text: str,
        ref_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate summary using LLM with paper-specific prompt.
        
        This is a compatibility method that provides the same interface as PaperSummarizer.
        For tests and simple use cases, it generates a single-stage summary.
        For production, use summarize_paper() which uses multi-stage refinement.
        
        Args:
            result: Search result with paper metadata.
            pdf_text: Extracted PDF text.
            ref_info: Optional reference analysis info.
            
        Returns:
            Generated summary text.
        """
        # Store ref_info for later use
        if ref_info:
            self._last_ref_info = ref_info
        
        # Use prompt builder to create the summary prompt (compatibility with old interface)
        from infrastructure.llm.templates.research import PaperSummarization
        from infrastructure.literature.analysis.domain_detector import DomainDetector
        
        # Detect domain for context-aware prompts
        domain_detector = DomainDetector()
        domain_result = domain_detector.detect_domain(
            text=pdf_text,
            title=result.title,
            abstract=result.abstract
        )
        
        # Get domain-specific instructions
        domain_instructions = domain_detector.get_domain_specific_instructions(domain_result.domain)
        
        # Create paper summarization prompt with domain context and reference info
        template = PaperSummarization()
        prompt = template.render(
            title=result.title,
            authors=', '.join(result.authors) if result.authors else 'Unknown',
            year=str(result.year) if result.year else 'Unknown',
            source=f"{result.source} ({result.venue or 'N/A'})",
            text=pdf_text,
            domain=domain_result.domain.value if domain_result.confidence > 0.5 else None,
            domain_instructions=domain_instructions if domain_result.confidence > 0.5 else None,
            reference_count=ref_info.get('count') if ref_info else None,
            references_section_found=ref_info.get('section_found', False) if ref_info else False
        )

        # Generate summary with improved parameters
        from infrastructure.llm.core.config import GenerationOptions
        options = GenerationOptions(
            temperature=0.45,
            max_tokens=4000
        )

        summary = self.llm_client.query(prompt, options=options, reset_context=True)
        cleaned = self._clean_summary_content(summary)
        # Apply deduplication BEFORE validation so validation sees clean content
        deduplicated = self._deduplicate_summary(cleaned)
        
        return deduplicated
    
    def _clean_summary_content(self, summary: str) -> str:
        """Remove unwanted sections and content from summary.
        
        Filters out sections that shouldn't appear in final summaries:
        - References, Citation, BibTeX sections
        - Abstract section (already extracted separately)
        - Summary section (redundant)
        - Keywords, Tags sections
        - AI-generated disclaimers and notes
        - Meta-commentary ("Okay, here's...", "This revised summary...", etc.)
        
        Args:
            summary: Raw summary text from LLM.
            
        Returns:
            Cleaned summary text with unwanted sections removed.
        """
        lines = summary.split('\n')
        cleaned_lines = []
        skip_section = False

        unwanted_sections = [
            '### References',
            '### Citation',
            '### BibTex',
            '### Abstract',
            '### Summary',  # Also remove Summary section
            '### Keywords',
            '### Tags',
            'Note:',
            'Note: The above',
            'This summary was generated',
            'generated by an AI model'
        ]
        
        # Patterns for meta-commentary to remove
        meta_commentary_patterns = [
            r'^Okay,?\s+here\'?s?\s+',
            r'^Here\'?s?\s+a\s+',
            r'^This\s+(?:is\s+)?(?:a\s+)?(?:comprehensive\s+)?(?:revised\s+)?summary',
            r'^I\'?ll\s+',
            r'^Let\s+me\s+',
            r'^I\'?m\s+',
            r'^As\s+(?:an\s+)?(?:AI\s+)?(?:assistant\s+)?,?\s*',
            r'^To\s+summarize\s+this\s+paper',
            r'^This\s+paper\s+presents.*?\.\s*Here\'?s?\s+',
            r'^Below\s+is\s+',
            r'^The\s+following\s+',
        ]

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Remove meta-commentary lines
            is_meta_commentary = False
            for pattern in meta_commentary_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_meta_commentary = True
                    break
            
            if is_meta_commentary:
                continue
            
            # Check if this line starts an unwanted section
            # Match both "### Section" and "## Section" formats
            is_unwanted_header = False
            for unwanted in unwanted_sections:
                unwanted_lower = unwanted.lower().strip('#').strip()
                # Check if line is a header matching unwanted section
                if (line_lower.startswith('###') or line_lower.startswith('##')) and unwanted_lower in line_lower:
                    is_unwanted_header = True
                    break
            
            if is_unwanted_header:
                skip_section = True
                continue

            # Stop skipping when we hit a valid section header (not unwanted)
            if (line.startswith('### ') or line.startswith('## ')) and skip_section:
                # Check if this new section is also unwanted
                line_lower_check = line.lower().strip()
                is_also_unwanted = any(
                    unwanted.lower().strip('#').strip() in line_lower_check 
                    for unwanted in unwanted_sections
                )
                if not is_also_unwanted:
                    skip_section = False

            if not skip_section:
                cleaned_lines.append(line)

        cleaned = '\n'.join(cleaned_lines).strip()

        # Remove trailing disclaimers and meta-commentary
        disclaimer_patterns = [
            r'Note:.*$',
            r'This summary.*$',
            r'generated by.*$',
            r'Do you want me to.*$',
            r'Would you like me to.*$',
            r'This revised summary.*$',
            r'approximately.*words in length.*$',
            r'adhering to the requested range.*$',
        ]

        for pattern in disclaimer_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove lines that are just meta-commentary at the start
        cleaned_lines_final = []
        found_first_real_content = False
        for line in cleaned.split('\n'):
            line_stripped = line.strip()
            if not line_stripped:
                if found_first_real_content:
                    cleaned_lines_final.append(line)
                continue
            
            # Check if this is still meta-commentary
            is_meta = False
            for pattern in meta_commentary_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_meta = True
                    break
            
            # Also check for common meta phrases in the line
            meta_phrases = [
                'comprehensive revised summary',
                'aiming for a polished',
                'suitable for a research paper',
                'adhering to the requested',
                'approximately',
                'words in length',
            ]
            if any(phrase in line_stripped.lower() for phrase in meta_phrases):
                is_meta = True
            
            if not is_meta:
                found_first_real_content = True
                cleaned_lines_final.append(line)
            elif found_first_real_content:
                # If we've found real content, keep everything after
                cleaned_lines_final.append(line)

        cleaned = '\n'.join(cleaned_lines_final).strip()

        return cleaned.strip()
    
    def _deduplicate_summary(self, summary: str) -> str:
        """Remove duplicate sections and paragraphs from summary.
        
        Args:
            summary: Summary text that may contain duplicates.
            
        Returns:
            Deduplicated summary text.
        """
        if not summary:
            return summary
        
        # First, remove duplicate sections at the header level
        # Group content by section to compare and remove duplicates
        lines = summary.split('\n')
        sections = []
        current_section_title = None
        current_section_content = []
        
        for line in lines:
            # Check if this is a section header
            if line.strip().startswith('###'):
                # Save previous section if any
                if current_section_title is not None:
                    sections.append((current_section_title, current_section_content))
                
                # Start new section
                current_section_title = line.strip()
                current_section_content = [line]
            else:
                if current_section_title is not None:
                    current_section_content.append(line)
                else:
                    # Content before any section - add as preamble
                    if not sections or sections[0][0] is None:
                        if not sections:
                            sections.append((None, []))
                        sections[0][1].append(line)
        
        # Add last section
        if current_section_title is not None:
            sections.append((current_section_title, current_section_content))
        
        # Deduplicate sections by title and content similarity
        # Keep first occurrence of each section, skip exact duplicates
        seen_sections = {}  # Maps normalized_title -> (section_title, section_content)
        deduplicated_sections = []
        
        for section_title, section_content in sections:
            if section_title is None:
                # Preamble - always include
                deduplicated_sections.append((None, section_content))
                continue
            
            # Normalize section title for comparison
            normalized_title = section_title.lower().strip('#').strip()
            # Get content text without the header for comparison
            content_lines = [line for line in section_content if not line.strip().startswith('###')]
            section_text = '\n'.join(content_lines).strip()
            
            # Check if we've seen this section title before
            if normalized_title in seen_sections:
                # Compare content - if identical, skip (duplicate)
                prev_title, prev_content = seen_sections[normalized_title]
                prev_content_lines = [line for line in prev_content if not line.strip().startswith('###')]
                previous_text = '\n'.join(prev_content_lines).strip()
                
                # If content is identical, it's a duplicate - skip
                if section_text == previous_text:
                    continue  # Exact duplicate, skip
                # If content is different, keep both (different content under same header)
                # This is rare but possible - same section title with different content
                deduplicated_sections.append((section_title, section_content))
            else:
                # First time seeing this section - keep it
                seen_sections[normalized_title] = (section_title, section_content)
                deduplicated_sections.append((section_title, section_content))
        
        # Reconstruct summary
        result_lines = []
        for section_title, section_content in deduplicated_sections:
            result_lines.extend(section_content)
        
        result = '\n'.join(result_lines)
        
        # Then use the deduplication from validation_repetition module for paragraph-level
        from infrastructure.llm.validation.repetition import deduplicate_sections
        
        # Use balanced mode for better paragraph deduplication
        return deduplicate_sections(
            result,
            max_repetitions=1,  # More aggressive for summaries
            mode="balanced",
            similarity_threshold=0.8,
            min_content_preservation=0.6
        )
    
    def _extract_claims_and_quotes(
        self,
        pdf_text: str,
        result: SearchResult,
        citation_key: str,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None
    ) -> str:
        """Extract key claims and important quotes from paper.
        
        Uses LLM to identify:
        - Main claims and hypotheses
        - Important direct quotes
        - Key findings and conclusions
        
        For large papers, uses chunking to process the paper in sections.
        
        Args:
            pdf_text: Full PDF text content.
            result: Search result with paper metadata.
            citation_key: Citation key for logging.
            progress_callback: Optional callback for progress events.
            
        Returns:
            Markdown-formatted text with claims and quotes.
        """
        from infrastructure.llm.templates.research import ClaimsQuotesExtraction
        from infrastructure.llm.core.config import GenerationOptions
        
        # Helper function to emit progress events
        def emit_progress(stage: str, status: str, message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
            if progress_callback:
                event = SummarizationProgressEvent(
                    citation_key=citation_key,
                    stage=stage,
                    status=status,
                    message=message,
                    metadata=metadata or {}
                )
                progress_callback(event)
        
        emit_progress("claims_extraction", "started", f"Extracting claims and quotes for {citation_key}")
        
        # Clear context before extraction
        logger.info(f"[{citation_key}] Clearing LLM context before claims/quotes extraction")
        self.llm_client.context.clear()
        
        # Check if paper fits in context
        # Estimate tokens: prompt template overhead (~2000 tokens) + paper text
        template = ClaimsQuotesExtraction()
        sample_prompt = template.render(
            title=result.title,
            authors=', '.join(result.authors) if result.authors else 'Unknown',
            year=str(result.year) if result.year else 'Unknown',
            source=result.source or 'unknown',
            text=pdf_text[:1000]  # Sample for estimation
        )
        estimated_prompt_tokens = len(sample_prompt) // 4
        paper_tokens = len(pdf_text) // 4
        total_estimated = estimated_prompt_tokens + paper_tokens
        
        # Get available context (with 20% reserve for response)
        available_tokens = self.llm_client.context.estimate_available_tokens(reserve_percent=0.2)
        
        # Determine if we need chunking
        use_chunking = total_estimated > available_tokens
        
        if use_chunking:
            logger.info(
                f"[{citation_key}] Paper text ({paper_tokens:,} tokens) exceeds context limit "
                f"({available_tokens:,} available). Using chunking mode..."
            )
            return self._extract_claims_and_quotes_chunked(
                pdf_text, result, citation_key, progress_callback, emit_progress
            )
        
        # Single-pass extraction for papers that fit in context
        logger.info(f"[{citation_key}] Paper fits in context ({paper_tokens:,} tokens <= {available_tokens:,} available). Using single-pass mode...")
        
        # Build full prompt
        prompt = template.render(
            title=result.title,
            authors=', '.join(result.authors) if result.authors else 'Unknown',
            year=str(result.year) if result.year else 'Unknown',
            source=result.source or 'unknown',
            text=pdf_text
        )
        
        # Generate extraction with appropriate options
        options = GenerationOptions(
            temperature=0.3,  # Lower temperature for more focused extraction
            max_tokens=3000,  # Allow for comprehensive extraction
        )
        
        try:
            extraction_start = time.time()
            claims_quotes = self.llm_client.query(prompt, options=options, reset_context=True)
            extraction_time = time.time() - extraction_start
            
            # Clean up the response
            claims_quotes = claims_quotes.strip()
            
            if not claims_quotes:
                logger.warning(f"[{citation_key}] Empty claims/quotes extraction result")
                return "## Key Claims and Hypotheses\n\nNo claims extracted.\n\n## Important Quotes\n\nNo quotes extracted."
            
            words = len(claims_quotes.split())
            
            # Detect and remove nested repetition patterns first
            from infrastructure.llm.validation.repetition import detect_nested_repetition, remove_nested_repetition
            has_nested, nested_patterns = detect_nested_repetition(claims_quotes)
            if has_nested:
                logger.warning(
                    f"[{citation_key}] Detected {len(nested_patterns)} nested repetition patterns in claims/quotes"
                )
                claims_quotes, nested_removed = remove_nested_repetition(claims_quotes)
                logger.info(
                    f"[{citation_key}] Removed {nested_removed} nested repetition artifacts"
                )
            
            # Validate quotes against source PDF text
            validated_claims_quotes = self._validate_quotes_against_source(
                claims_quotes, pdf_text, citation_key
            )
            
            # Apply deduplication to remove any remaining repetition
            from infrastructure.llm.validation.repetition import deduplicate_sections
            original_length = len(validated_claims_quotes)
            deduplicated_claims_quotes = deduplicate_sections(
                validated_claims_quotes,
                max_repetitions=1,
                mode="aggressive",
                similarity_threshold=0.75,
                min_content_preservation=0.7
            )
            dedup_removed = original_length - len(deduplicated_claims_quotes)
            
            if dedup_removed > 0:
                logger.info(
                    f"[{citation_key}] Deduplication removed {dedup_removed} chars "
                    f"({original_length} → {len(deduplicated_claims_quotes)} chars, "
                    f"{len(deduplicated_claims_quotes)/original_length:.1%} preserved)"
                )
            
            final_words = len(deduplicated_claims_quotes.split())
            
            logger.info(
                f"[{citation_key}] Claims/quotes extraction completed: "
                f"{final_words} words in {extraction_time:.2f}s "
                f"(removed {nested_removed if has_nested else 0} nested patterns, "
                f"{dedup_removed} chars via deduplication)"
            )
            
            emit_progress(
                "claims_extraction",
                "completed",
                f"Extracted {final_words} words of claims and quotes",
                {
                    "words": final_words,
                    "time": extraction_time,
                    "nested_patterns_removed": nested_removed if has_nested else 0,
                    "deduplication_chars_removed": dedup_removed,
                    "original_words": words
                }
            )
            
            return deduplicated_claims_quotes
            
        except Exception as e:
            extraction_time = time.time() - extraction_start if 'extraction_start' in locals() else 0.0
            logger.error(f"[{citation_key}] Claims/quotes extraction failed: {e}")
            emit_progress("claims_extraction", "failed", f"Extraction failed: {e}")
            return f"## Key Claims and Hypotheses\n\nExtraction failed: {e}\n\n## Important Quotes\n\nExtraction failed."
    
    def _validate_quotes_against_source(
        self,
        claims_quotes_text: str,
        pdf_text: str,
        citation_key: str
    ) -> str:
        """Validate that extracted quotes actually exist in the source PDF text.
        
        Removes hallucinated quotes that don't appear in the source, allowing for
        minor spacing variations (fuzzy matching).
        
        Args:
            claims_quotes_text: Extracted claims and quotes text.
            pdf_text: Source PDF text to validate against.
            citation_key: Citation key for logging.
            
        Returns:
            Validated claims/quotes text with invalid quotes removed.
        """
        import re
        
        if not claims_quotes_text or not pdf_text:
            return claims_quotes_text
        
        # Normalize PDF text for comparison (remove extra whitespace)
        pdf_normalized = re.sub(r'\s+', ' ', pdf_text.lower())
        
        # Extract quotes from claims/quotes text
        # Look for quote patterns: "quote text" or **Quote:** "quote text"
        quote_patterns = [
            r'\*\*Quote:\*\*\s*"([^"]+)"',  # **Quote:** "text"
            r'"([^"]{20,500})"',  # "text" (20-500 chars)
            r"'([^']{20,500})'",  # 'text' (20-500 chars)
        ]
        
        lines = claims_quotes_text.split('\n')
        validated_lines = []
        removed_count = 0

        # Track quotes to detect duplicates across the document
        seen_quotes = set()

        for line in lines:
            is_quote_line = False
            quote_found = False
            
            # Check if this line contains a quote
            for pattern in quote_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    is_quote_line = True
                    quote_text = match.group(1)

                    # Skip artifacts like "dynamicssimulationsand" → "dynamics simulations and"
                    if '→' in quote_text or '->' in quote_text:
                        removed_count += 1
                        logger.debug(f"[{citation_key}] Removed quote artifact: {quote_text[:50]}...")
                        quote_found = False
                        break

                    # Check for missing quote text (format validation)
                    if not quote_text.strip():
                        removed_count += 1
                        logger.debug(f"[{citation_key}] Removed quote with missing text: {line[:80]}...")
                        quote_found = False
                        break

                    # Check for meta-commentary embedded in quotes
                    meta_patterns = [
                        r'^according to the paper:',
                        r'^the authors? (state|note|argue|demonstrate):',
                        r'^they (state|note|argue|demonstrate):',
                        r'^the paper (argues|states|demonstrates):',
                        r'^in the [^:]*:\s*',
                    ]
                    for pattern in meta_patterns:
                        if re.search(pattern, quote_text.lower().strip()):
                            removed_count += 1
                            logger.debug(f"[{citation_key}] Removed quote with embedded meta-commentary: {quote_text[:50]}...")
                            quote_found = False
                            break
                    if not quote_found:
                        break

                    # Check for duplicate quotes across document
                    quote_key = quote_text.lower().strip()
                    if quote_key in seen_quotes:
                        removed_count += 1
                        logger.debug(f"[{citation_key}] Removed duplicate quote: {quote_text[:50]}...")
                        quote_found = False
                        break
                    else:
                        seen_quotes.add(quote_key)
                    
                    # Normalize quote for comparison (remove spaces, lowercase)
                    quote_normalized = re.sub(r'\s+', '', quote_text.lower())
                    
                    # Check if quote exists in PDF (fuzzy matching - allow spacing variations)
                    # Try exact match first
                    if quote_text.lower() in pdf_normalized:
                        quote_found = True
                        break
                    
                    # Try normalized match (no spaces)
                    if quote_normalized in pdf_normalized.replace(' ', ''):
                        quote_found = True
                        break
                    
                    # Try partial match (first 50 chars)
                    if len(quote_normalized) > 50:
                        quote_partial = quote_normalized[:50]
                        # Check if this partial appears in PDF
                        pdf_no_spaces = pdf_normalized.replace(' ', '')
                        if quote_partial in pdf_no_spaces:
                            quote_found = True
                            break
                
                if is_quote_line:
                    break
            
            # Keep the line if it's not a quote, or if the quote was found
            if not is_quote_line or quote_found:
                validated_lines.append(line)
            else:
                removed_count += 1
                logger.debug(f"[{citation_key}] Removed quote not found in source: {line[:80]}...")
        
        if removed_count > 0:
            logger.warning(
                f"[{citation_key}] Quote validation removed {removed_count} quotes "
                f"that were not found in source PDF"
            )
        
        return '\n'.join(validated_lines)
    
    def _extract_claims_and_quotes_chunked(
        self,
        pdf_text: str,
        result: SearchResult,
        citation_key: str,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]],
        emit_progress: Callable
    ) -> str:
        """Extract claims and quotes using chunking for large papers.
        
        Args:
            pdf_text: Full PDF text content.
            result: Search result with paper metadata.
            citation_key: Citation key for logging.
            progress_callback: Optional callback for progress events.
            emit_progress: Progress event emitter function.
            
        Returns:
            Combined markdown-formatted text with claims and quotes from all chunks.
        """
        from infrastructure.llm.templates.research import ClaimsQuotesExtraction
        from infrastructure.llm.core.config import GenerationOptions
        
        # Initialize chunker with same parameters as two-stage summarization
        chunker = PDFChunker(
            target_chunk_size=15000,
            chunk_overlap=500,
            min_chunk_size=1000
        )
        
        # Chunk the PDF text
        emit_progress("claims_extraction", "in_progress", "Chunking PDF text for claims/quotes extraction...")
        logger.info(f"[{citation_key}] Chunking PDF text ({len(pdf_text):,} chars) for claims/quotes extraction...")
        chunking_result = chunker.chunk_text(pdf_text, preserve_sections=True)
        
        logger.info(
            f"[{citation_key}] Chunking completed: {chunking_result.total_chunks} chunks, "
            f"avg size: {chunking_result.average_chunk_size:.0f} chars"
        )
        
        if chunking_result.total_chunks == 0:
            logger.error(f"[{citation_key}] No chunks created from PDF text")
            return "## Key Claims and Hypotheses\n\nChunking failed: No chunks created.\n\n## Important Quotes\n\nExtraction failed."
        
        # Process each chunk
        all_claims_quotes = []
        successful_chunks = 0
        failed_chunks = 0
        
        template = ClaimsQuotesExtraction()
        options = GenerationOptions(
            temperature=0.3,
            max_tokens=3000,
        )
        
        for i, chunk in enumerate(chunking_result.chunks):
            chunk_num = i + 1
            logger.info(
                f"[{citation_key}] Processing chunk {chunk_num}/{chunking_result.total_chunks} "
                f"for claims/quotes extraction ({len(chunk.text):,} chars)..."
            )
            
            emit_progress(
                "claims_extraction",
                "in_progress",
                f"Processing chunk {chunk_num}/{chunking_result.total_chunks}...",
                {"chunk_index": chunk_num, "total_chunks": chunking_result.total_chunks}
            )
            
            # Build prompt for this chunk
            chunk_prompt = template.render(
                title=result.title,
                authors=', '.join(result.authors) if result.authors else 'Unknown',
                year=str(result.year) if result.year else 'Unknown',
                source=result.source or 'unknown',
                text=chunk.text
            )
            
            try:
                # Clear context before each chunk to avoid accumulation
                self.llm_client.context.clear()
                
                chunk_start = time.time()
                chunk_result = self.llm_client.query(chunk_prompt, options=options, reset_context=True)
                chunk_time = time.time() - chunk_start
                
                if chunk_result and chunk_result.strip():
                    all_claims_quotes.append(chunk_result.strip())
                    successful_chunks += 1
                    logger.info(
                        f"[{citation_key}] Chunk {chunk_num} processed: "
                        f"{len(chunk_result.split())} words in {chunk_time:.2f}s"
                    )
                else:
                    logger.warning(f"[{citation_key}] Chunk {chunk_num} produced empty result")
                    failed_chunks += 1
                    
            except Exception as e:
                logger.error(f"[{citation_key}] Failed to process chunk {chunk_num}: {e}")
                failed_chunks += 1
                # Continue with other chunks
        
        if not all_claims_quotes:
            error_msg = f"All chunks failed for claims/quotes extraction (processed {chunking_result.total_chunks} chunks)"
            logger.error(f"[{citation_key}] {error_msg}")
            emit_progress("claims_extraction", "failed", error_msg)
            return f"## Key Claims and Hypotheses\n\nExtraction failed: {error_msg}\n\n## Important Quotes\n\nExtraction failed."
        
        # Combine results from all chunks
        logger.info(
            f"[{citation_key}] Combining results from {successful_chunks}/{chunking_result.total_chunks} chunks "
            f"({failed_chunks} failed)"
        )
        
        # Combine all chunks
        combined_claims_quotes = "\n\n---\n\n".join(all_claims_quotes)
        
        # Detect and remove nested repetition patterns first
        from infrastructure.llm.validation.repetition import detect_nested_repetition, remove_nested_repetition
        has_nested, nested_patterns = detect_nested_repetition(combined_claims_quotes)
        if has_nested:
            logger.warning(
                f"[{citation_key}] Detected {len(nested_patterns)} nested repetition patterns in combined claims/quotes"
            )
            combined_claims_quotes, nested_removed = remove_nested_repetition(combined_claims_quotes)
            logger.info(
                f"[{citation_key}] Removed {nested_removed} nested repetition artifacts from combined results"
            )
        
        # Validate quotes against source PDF text
        validated_claims_quotes = self._validate_quotes_against_source(
            combined_claims_quotes, pdf_text, citation_key
        )
        
        # Apply deduplication to remove any remaining repetition
        from infrastructure.llm.validation.repetition import deduplicate_sections
        original_length = len(validated_claims_quotes)
        deduplicated_claims_quotes = deduplicate_sections(
            validated_claims_quotes,
            max_repetitions=1,
            mode="aggressive",
            similarity_threshold=0.75,
            min_content_preservation=0.7
        )
        dedup_removed = original_length - len(deduplicated_claims_quotes)
        
        if dedup_removed > 0:
            logger.info(
                f"[{citation_key}] Deduplication removed {dedup_removed} chars from combined results "
                f"({original_length} → {len(deduplicated_claims_quotes)} chars, "
                f"{len(deduplicated_claims_quotes)/original_length:.1%} preserved)"
            )
        
        final_words = len(deduplicated_claims_quotes.split())
        
        logger.info(
            f"[{citation_key}] Claims/quotes extraction completed (chunked): "
            f"{final_words} words from {successful_chunks}/{chunking_result.total_chunks} chunks "
            f"(removed {nested_removed if has_nested else 0} nested patterns, "
            f"{dedup_removed} chars via deduplication)"
        )
        
        emit_progress(
            "claims_extraction",
            "completed",
            f"Extracted {final_words} words of claims and quotes from {successful_chunks} chunks",
            {
                "words": final_words,
                "successful_chunks": successful_chunks,
                "total_chunks": chunking_result.total_chunks,
                "nested_patterns_removed": nested_removed if has_nested else 0,
                "deduplication_chars_removed": dedup_removed
            }
        )
        
        return deduplicated_claims_quotes
    
    def _analyze_methods_and_tools(
        self,
        pdf_text: str,
        result: SearchResult,
        citation_key: str,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None
    ) -> str:
        """Analyze methods, algorithms, frameworks, datasets, and tools.
        
        Extracts:
        - Algorithms and methodologies used
        - Software frameworks and libraries
        - Datasets employed
        - Evaluation metrics
        - Software tools and platforms
        
        For large papers, uses chunking to process the paper in sections.
        
        Args:
            pdf_text: Full PDF text content.
            result: Search result with paper metadata.
            citation_key: Citation key for logging.
            progress_callback: Optional callback for progress events.
            
        Returns:
            Markdown-formatted text with methods and tools analysis.
        """
        from infrastructure.llm.templates.research import MethodsToolsAnalysis
        from infrastructure.llm.core.config import GenerationOptions
        
        # Helper function to emit progress events
        def emit_progress(stage: str, status: str, message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
            if progress_callback:
                event = SummarizationProgressEvent(
                    citation_key=citation_key,
                    stage=stage,
                    status=status,
                    message=message,
                    metadata=metadata or {}
                )
                progress_callback(event)
        
        emit_progress("methods_analysis", "started", f"Analyzing methods and tools for {citation_key}")
        
        # Clear context before analysis
        logger.info(f"[{citation_key}] Clearing LLM context before methods/tools analysis")
        self.llm_client.context.clear()
        
        # Check if paper fits in context
        # Estimate tokens: prompt template overhead (~2000 tokens) + paper text
        template = MethodsToolsAnalysis()
        sample_prompt = template.render(
            title=result.title,
            authors=', '.join(result.authors) if result.authors else 'Unknown',
            year=str(result.year) if result.year else 'Unknown',
            source=result.source or 'unknown',
            text=pdf_text[:1000]  # Sample for estimation
        )
        estimated_prompt_tokens = len(sample_prompt) // 4
        paper_tokens = len(pdf_text) // 4
        total_estimated = estimated_prompt_tokens + paper_tokens
        
        # Get available context (with 20% reserve for response)
        available_tokens = self.llm_client.context.estimate_available_tokens(reserve_percent=0.2)
        
        # Determine if we need chunking
        use_chunking = total_estimated > available_tokens
        
        if use_chunking:
            logger.info(
                f"[{citation_key}] Paper text ({paper_tokens:,} tokens) exceeds context limit "
                f"({available_tokens:,} available). Using chunking mode..."
            )
            return self._analyze_methods_and_tools_chunked(
                pdf_text, result, citation_key, progress_callback, emit_progress
            )
        
        # Single-pass analysis for papers that fit in context
        logger.info(f"[{citation_key}] Paper fits in context ({paper_tokens:,} tokens <= {available_tokens:,} available). Using single-pass mode...")
        
        # Build full prompt
        prompt = template.render(
            title=result.title,
            authors=', '.join(result.authors) if result.authors else 'Unknown',
            year=str(result.year) if result.year else 'Unknown',
            source=result.source or 'unknown',
            text=pdf_text
        )
        
        # Generate analysis with appropriate options
        options = GenerationOptions(
            temperature=0.3,  # Lower temperature for more focused analysis
            max_tokens=3000,  # Allow for comprehensive analysis
        )
        
        try:
            analysis_start = time.time()
            methods_tools = self.llm_client.query(prompt, options=options, reset_context=True)
            analysis_time = time.time() - analysis_start
            
            # Clean up the response
            methods_tools = methods_tools.strip()
            
            if not methods_tools:
                logger.warning(f"[{citation_key}] Empty methods/tools analysis result")
                return "## Algorithms and Methodologies\n\nNo methods extracted.\n\n## Software Frameworks and Libraries\n\nNo frameworks extracted.\n\n## Datasets\n\nNo datasets extracted.\n\n## Evaluation Metrics\n\nNo metrics extracted.\n\n## Software Tools and Platforms\n\nNo tools extracted."
            
            words = len(methods_tools.split())

            # Validate and clean the output
            from infrastructure.literature.summarization.methods_tools_validator import MethodsToolsValidator
            validator = MethodsToolsValidator()
            validated_methods_tools, validation_warnings = validator.validate_output(
                methods_tools, pdf_text, citation_key
            )

            # Log validation results
            if validation_warnings:
                logger.warning(
                    f"[{citation_key}] Methods/tools validation warnings: "
                    f"{len(validation_warnings)} issues found"
                )
                for warning in validation_warnings[:3]:  # Log first 3 warnings
                    logger.warning(f"[{citation_key}] {warning}")

            final_words = len(validated_methods_tools.split())
            logger.info(
                f"[{citation_key}] Methods/tools analysis completed: "
                f"{words} → {final_words} words in {analysis_time:.2f}s "
                f"(validation: {len(validation_warnings)} warnings)"
            )

            emit_progress(
                "methods_analysis",
                "completed",
                f"Analyzed and validated methods and tools: {final_words} words",
                {
                    "words": final_words,
                    "validation_warnings": len(validation_warnings),
                    "time": analysis_time
                }
            )

            return validated_methods_tools
            
        except Exception as e:
            analysis_time = time.time() - analysis_start if 'analysis_start' in locals() else 0.0
            logger.error(f"[{citation_key}] Methods/tools analysis failed: {e}")
            emit_progress("methods_analysis", "failed", f"Analysis failed: {e}")
            return f"## Algorithms and Methodologies\n\nAnalysis failed: {e}\n\n## Software Frameworks and Libraries\n\nAnalysis failed.\n\n## Datasets\n\nAnalysis failed.\n\n## Evaluation Metrics\n\nAnalysis failed.\n\n## Software Tools and Platforms\n\nAnalysis failed."
    
    def _analyze_methods_and_tools_chunked(
        self,
        pdf_text: str,
        result: SearchResult,
        citation_key: str,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]],
        emit_progress: Callable
    ) -> str:
        """Analyze methods and tools using chunking for large papers.
        
        Args:
            pdf_text: Full PDF text content.
            result: Search result with paper metadata.
            citation_key: Citation key for logging.
            progress_callback: Optional callback for progress events.
            emit_progress: Progress event emitter function.
            
        Returns:
            Combined markdown-formatted text with methods and tools from all chunks.
        """
        from infrastructure.llm.templates.research import MethodsToolsAnalysis
        from infrastructure.llm.core.config import GenerationOptions
        
        # Initialize chunker with same parameters as two-stage summarization
        chunker = PDFChunker(
            target_chunk_size=15000,
            chunk_overlap=500,
            min_chunk_size=1000
        )
        
        # Chunk the PDF text
        emit_progress("methods_analysis", "in_progress", "Chunking PDF text for methods/tools analysis...")
        logger.info(f"[{citation_key}] Chunking PDF text ({len(pdf_text):,} chars) for methods/tools analysis...")
        chunking_result = chunker.chunk_text(pdf_text, preserve_sections=True)
        
        logger.info(
            f"[{citation_key}] Chunking completed: {chunking_result.total_chunks} chunks, "
            f"avg size: {chunking_result.average_chunk_size:.0f} chars"
        )
        
        if chunking_result.total_chunks == 0:
            logger.error(f"[{citation_key}] No chunks created from PDF text")
            return "## Algorithms and Methodologies\n\nChunking failed: No chunks created.\n\n## Software Frameworks and Libraries\n\nAnalysis failed.\n\n## Datasets\n\nAnalysis failed.\n\n## Evaluation Metrics\n\nAnalysis failed.\n\n## Software Tools and Platforms\n\nAnalysis failed."
        
        # Process each chunk
        all_methods_tools = []
        successful_chunks = 0
        failed_chunks = 0
        
        template = MethodsToolsAnalysis()
        options = GenerationOptions(
            temperature=0.3,
            max_tokens=3000,
        )
        
        for i, chunk in enumerate(chunking_result.chunks):
            chunk_num = i + 1
            logger.info(
                f"[{citation_key}] Processing chunk {chunk_num}/{chunking_result.total_chunks} "
                f"for methods/tools analysis ({len(chunk.text):,} chars)..."
            )
            
            emit_progress(
                "methods_analysis",
                "in_progress",
                f"Processing chunk {chunk_num}/{chunking_result.total_chunks}...",
                {"chunk_index": chunk_num, "total_chunks": chunking_result.total_chunks}
            )
            
            # Build prompt for this chunk
            chunk_prompt = template.render(
                title=result.title,
                authors=', '.join(result.authors) if result.authors else 'Unknown',
                year=str(result.year) if result.year else 'Unknown',
                source=result.source or 'unknown',
                text=chunk.text
            )
            
            try:
                # Clear context before each chunk to avoid accumulation
                self.llm_client.context.clear()
                
                chunk_start = time.time()
                chunk_result = self.llm_client.query(chunk_prompt, options=options, reset_context=True)
                chunk_time = time.time() - chunk_start
                
                if chunk_result and chunk_result.strip():
                    all_methods_tools.append(chunk_result.strip())
                    successful_chunks += 1
                    logger.info(
                        f"[{citation_key}] Chunk {chunk_num} processed: "
                        f"{len(chunk_result.split())} words in {chunk_time:.2f}s"
                    )
                else:
                    logger.warning(f"[{citation_key}] Chunk {chunk_num} produced empty result")
                    failed_chunks += 1
                    
            except Exception as e:
                logger.error(f"[{citation_key}] Failed to process chunk {chunk_num}: {e}")
                failed_chunks += 1
                # Continue with other chunks
        
        if not all_methods_tools:
            error_msg = f"All chunks failed for methods/tools analysis (processed {chunking_result.total_chunks} chunks)"
            logger.error(f"[{citation_key}] {error_msg}")
            emit_progress("methods_analysis", "failed", error_msg)
            return f"## Algorithms and Methodologies\n\nAnalysis failed: {error_msg}\n\n## Software Frameworks and Libraries\n\nAnalysis failed.\n\n## Datasets\n\nAnalysis failed.\n\n## Evaluation Metrics\n\nAnalysis failed.\n\n## Software Tools and Platforms\n\nAnalysis failed."
        
        # Combine results from all chunks
        logger.info(
            f"[{citation_key}] Combining results from {successful_chunks}/{chunking_result.total_chunks} chunks "
            f"({failed_chunks} failed)"
        )
        
        # Merge results by section, deduplicating items
        # Collect all items by section
        sections = {
            "Algorithms and Methodologies": [],
            "Software Frameworks and Libraries": [],
            "Datasets": [],
            "Evaluation Metrics": [],
            "Software Tools and Platforms": []
        }
        
        seen_items = set()  # Track seen items to avoid duplicates
        
        for chunk_result in all_methods_tools:
            # Parse chunk result into sections
            current_section = None
            for line in chunk_result.split('\n'):
                line_stripped = line.strip()
                # Check if this is a section header
                if line_stripped.startswith('## '):
                    section_name = line_stripped[3:].strip()
                    if section_name in sections:
                        current_section = section_name
                elif current_section and line_stripped:
                    # Add item to section if not duplicate
                    # Use first 100 chars as key for deduplication
                    item_key = (current_section, line_stripped[:100].lower())
                    if item_key not in seen_items:
                        seen_items.add(item_key)
                        sections[current_section].append(line_stripped)
        
        # Build final result
        result_parts = []
        for section_name, items in sections.items():
            result_parts.append(f"## {section_name}")
            if items:
                for item in items:
                    result_parts.append(item)
            else:
                result_parts.append("Not specified in paper")
            result_parts.append("")  # Empty line between sections
        
        final_result = '\n'.join(result_parts).strip()

        # Validate and clean the combined result
        from infrastructure.literature.summarization.methods_tools_validator import MethodsToolsValidator
        validator = MethodsToolsValidator()
        validated_result, validation_warnings = validator.validate_output(
            final_result, pdf_text, citation_key
        )

        # Log validation results
        if validation_warnings:
            logger.warning(
                f"[{citation_key}] Methods/tools validation warnings (chunked): "
                f"{len(validation_warnings)} issues found"
            )
            for warning in validation_warnings[:3]:  # Log first 3 warnings
                logger.warning(f"[{citation_key}] {warning}")

        final_words = len(validated_result.split())
        logger.info(
            f"[{citation_key}] Methods/tools analysis completed (chunked mode): "
            f"{words} → {final_words} words from {successful_chunks} chunks "
            f"(validation: {len(validation_warnings)} warnings)"
        )

        emit_progress(
            "methods_analysis",
            "completed",
            f"Analyzed and validated methods and tools: {final_words} words from {successful_chunks} chunks",
            {
                "words": final_words,
                "chunks_processed": successful_chunks,
                "total_chunks": chunking_result.total_chunks,
                "validation_warnings": len(validation_warnings)
            }
        )

        return validated_result
    
    def _classify_paper(
        self,
        pdf_text: str,
        result: SearchResult,
        citation_key: str,
        progress_callback: Optional[Callable[[SummarizationProgressEvent], None]] = None
    ) -> Optional[PaperClassification]:
        """Classify paper by research type (Core/Theory/Math, Translation/Tool, or Applied).
        
        Uses LLM with structured JSON output to classify the paper based on its
        primary contribution and focus. Considers the entire paper content.
        
        Args:
            pdf_text: Full PDF text content.
            result: Search result with paper metadata.
            citation_key: Citation key for logging.
            progress_callback: Optional callback for progress events.
            
        Returns:
            PaperClassification object if successful, None on failure.
        """
        from infrastructure.literature.summarization.models import PaperClassification
        from infrastructure.llm.templates.research import PaperClassificationTemplate
        from infrastructure.llm.core.config import GenerationOptions
        
        # Helper function to emit progress events
        def emit_progress(stage: str, status: str, message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
            if progress_callback:
                event = SummarizationProgressEvent(
                    citation_key=citation_key,
                    stage=stage,
                    status=status,
                    message=message,
                    metadata=metadata or {}
                )
                progress_callback(event)
        
        emit_progress("classification", "started", f"Classifying paper: {citation_key}")
        
        # Clear context before classification
        logger.info(f"[{citation_key}] Clearing LLM context before classification")
        self.llm_client.context.clear()
        
        # Build classification prompt
        template = PaperClassificationTemplate()
        prompt = template.render(
            title=result.title,
            authors=', '.join(result.authors) if result.authors else 'Unknown',
            year=str(result.year) if result.year else 'Unknown',
            full_text=pdf_text
        )
        
        # Define JSON schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["core_theory_math", "translation_tool", "applied"]
                },
                "domain": {
                    "type": ["string", "null"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reasoning": {
                    "type": ["string", "null"]
                }
            },
            "required": ["category", "confidence"],
            "if": {
                "properties": {"category": {"const": "applied"}}
            },
            "then": {
                "required": ["domain"]
            }
        }
        
        # Generate classification with structured output
        options = GenerationOptions(
            temperature=0.3,  # Lower temperature for more consistent classification
            max_tokens=500,   # Classification should be brief
        )
        
        try:
            classification_start = time.time()
            classification_data = self.llm_client.query_structured(
                prompt=prompt,
                schema=schema,
                options=options,
                use_native_json=True
            )
            classification_time = time.time() - classification_start
            
            # Validate and create PaperClassification object
            category = classification_data.get("category")
            if category not in ["core_theory_math", "translation_tool", "applied"]:
                logger.warning(f"[{citation_key}] Invalid category: {category}")
                emit_progress("classification", "failed", f"Invalid category: {category}")
                return None
            
            # Validate domain requirement for applied papers
            domain = classification_data.get("domain")
            if category == "applied" and not domain:
                logger.warning(f"[{citation_key}] Applied paper missing domain")
                emit_progress("classification", "failed", "Applied paper missing domain")
                return None
            
            confidence = classification_data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
                logger.warning(f"[{citation_key}] Invalid confidence: {confidence}")
                confidence = 0.0
            
            reasoning = classification_data.get("reasoning")
            
            classification = PaperClassification(
                category=category,
                domain=domain,
                confidence=float(confidence),
                reasoning=reasoning
            )
            
            logger.info(
                f"[{citation_key}] Classification completed: {category}"
                f"{f' ({domain})' if domain else ''}, "
                f"confidence: {confidence:.2f} in {classification_time:.2f}s"
            )
            
            emit_progress(
                "classification",
                "completed",
                f"Classified as {category}{f' ({domain})' if domain else ''}",
                {
                    "category": category,
                    "domain": domain,
                    "confidence": confidence,
                    "time": classification_time
                }
            )
            
            return classification
            
        except Exception as e:
            classification_time = time.time() - classification_start if 'classification_start' in locals() else 0.0
            logger.error(f"[{citation_key}] Classification failed: {e}")
            emit_progress("classification", "failed", f"Classification failed: {e}")
            return None
    
    def save_summary(
        self,
        result: SearchResult,
        summary_result: SummarizationResult,
        output_dir: Path,
        pdf_path: Optional[Path] = None
    ) -> List[Path]:
        """Save comprehensive summary, claims/quotes, and methods/tools analysis to separate files.
        
        Saves three separate markdown files:
        - `{citation_key}_summary.md` - Comprehensive summary
        - `{citation_key}_claims_quotes.md` - Key claims and quotes extraction
        - `{citation_key}_methods_tools.md` - Methods and tools analysis
        
        Each file includes paper metadata. The summary file also includes
        validation status and quality score.
        
        Summaries are saved even if validation failed, with clear indication
        of validation status for review purposes.

        Args:
            result: Search result with paper metadata (title, authors, year, etc.).
            summary_result: SummarizationResult to save (must have summary_text).
            output_dir: Directory for summary files (created if doesn't exist).
            pdf_path: Optional path to PDF file (for metadata tracking).

        Returns:
            List of paths to saved files (summary, claims_quotes, methods_tools).

        Raises:
            FileOperationError: If summary_text is None or file writing fails.
            
        Note:
            This method always saves summaries, regardless of validation status.
            Validation metadata is included in the saved file for transparency.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        citation_key = summary_result.citation_key
        saved_paths = []

        # Check if summary text exists
        if not summary_result.summary_text:
            logger.warning(f"[{citation_key}] No summary text to save, skipping save operation")
            raise FileOperationError(
                f"No summary text to save for {citation_key}",
                context={"path": str(output_dir / f"{citation_key}_summary.md")}
            )

        # Build validation metadata section
        validation_section = ""
        if summary_result.quality_score is not None or summary_result.validation_errors:
            validation_status = "✓ Accepted" if summary_result.success else "⚠ Rejected"
            quality_score_str = f"{summary_result.quality_score:.2f}" if summary_result.quality_score is not None else 'N/A'
            validation_section = f"""
**Validation Status:** {validation_status}
**Quality Score:** {quality_score_str}
"""
            if summary_result.validation_errors:
                validation_section += f"**Validation Errors:** {len(summary_result.validation_errors)} error(s)\n"
                for error in summary_result.validation_errors[:5]:  # Show first 5 errors
                    validation_section += f"  - {error}\n"
            validation_section += "\n"

        # Build classification section
        classification_section = ""
        if summary_result.classification:
            category_display = summary_result.classification.category.replace('_', ' ').title()
            domain_display = summary_result.classification.domain or 'N/A'
            confidence_display = f"{summary_result.classification.confidence:.2f}"
            reasoning_display = summary_result.classification.reasoning or 'N/A'
            classification_section = f"""
## Classification

- **Category**: {category_display}
- **Domain**: {domain_display}
- **Confidence**: {confidence_display}
- **Reasoning**: {reasoning_display}

"""

        # Build markdown content with validation metadata
        content = f"""# {result.title}

**Authors:** {', '.join(result.authors) if result.authors else 'Unknown'}

**Year:** {result.year or 'Unknown'}

**Source:** {result.source}

**Venue:** {result.venue or 'N/A'}

**DOI:** {result.doi or 'N/A'}

**PDF:** [{citation_key}.pdf](../pdfs/{citation_key}.pdf)

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
{validation_section}{classification_section}---

{summary_result.summary_text}
"""

        # Save comprehensive summary
        summary_path = output_dir / f"{citation_key}_summary.md"
        try:
            summary_path.write_text(content, encoding='utf-8')
            file_size = summary_path.stat().st_size
            logger.info(f"Saved summary: {summary_path.name} ({file_size:,} bytes) -> {summary_path}")
            saved_paths.append(summary_path)
        except Exception as e:
            logger.error(f"Failed to save summary file: {e}")
            raise FileOperationError(
                f"Failed to save summary: {e}",
                context={"path": str(summary_path)}
            )
        
        # Save claims and quotes
        if summary_result.claims_quotes_text:
            claims_quotes_path = output_dir / f"{citation_key}_claims_quotes.md"
            try:
                claims_quotes_content = f"""# {result.title} - Key Claims and Quotes

**Authors:** {', '.join(result.authors) if result.authors else 'Unknown'}

**Year:** {result.year or 'Unknown'}

**Source:** {result.source}

**Venue:** {result.venue or 'N/A'}

**DOI:** {result.doi or 'N/A'}

**PDF:** [{citation_key}.pdf](../pdfs/{citation_key}.pdf)

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

{summary_result.claims_quotes_text}
"""
                claims_quotes_path.write_text(claims_quotes_content, encoding='utf-8')
                file_size = claims_quotes_path.stat().st_size
                logger.info(f"Saved claims/quotes: {claims_quotes_path.name} ({file_size:,} bytes) -> {claims_quotes_path}")
                saved_paths.append(claims_quotes_path)
            except Exception as e:
                logger.warning(f"Failed to save claims/quotes file: {e}")
        else:
            logger.debug(f"[{citation_key}] No claims/quotes text to save")
        
        # Save methods and tools analysis
        if summary_result.methods_tools_text:
            methods_tools_path = output_dir / f"{citation_key}_methods_tools.md"
            try:
                methods_tools_content = f"""# {result.title} - Methods and Tools Analysis

**Authors:** {', '.join(result.authors) if result.authors else 'Unknown'}

**Year:** {result.year or 'Unknown'}

**Source:** {result.source}

**Venue:** {result.venue or 'N/A'}

**DOI:** {result.doi or 'N/A'}

**PDF:** [{citation_key}.pdf](../pdfs/{citation_key}.pdf)

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

{summary_result.methods_tools_text}
"""
                methods_tools_path.write_text(methods_tools_content, encoding='utf-8')
                file_size = methods_tools_path.stat().st_size
                logger.info(f"Saved methods/tools: {methods_tools_path.name} ({file_size:,} bytes) -> {methods_tools_path}")
                saved_paths.append(methods_tools_path)
            except Exception as e:
                logger.warning(f"Failed to save methods/tools file: {e}")
        else:
            logger.debug(f"[{citation_key}] No methods/tools text to save")
        
        # Save metadata to JSON if available
        try:
            from infrastructure.literature.summarization.metadata import (
                SummaryMetadataManager,
                SummaryMetadata
            )
            
            metadata_path = output_dir.parent / "summaries_metadata.json"
            manager = SummaryMetadataManager(metadata_path=metadata_path)
            
            pdf_size_bytes = None
            pdf_relative_path = None
            if pdf_path and pdf_path.exists():
                pdf_size_bytes = pdf_path.stat().st_size
                try:
                    pdf_relative_path = str(pdf_path.relative_to(output_dir.parent.parent))
                except ValueError:
                    pdf_relative_path = str(pdf_path)
            
            metadata = SummaryMetadata(
                citation_key=citation_key,
                title=result.title,
                authors=result.authors or [],
                year=result.year,
                pdf_size_bytes=pdf_size_bytes,
                pdf_path=pdf_relative_path,
                summary_path=str(summary_path.relative_to(output_dir.parent)) if summary_path.exists() else None,
                input_chars=summary_result.input_chars,
                input_words=summary_result.input_words,
                output_words=summary_result.output_words,
                generation_time=summary_result.generation_time,
                quality_score=summary_result.quality_score,
                reference_count=self._last_ref_info.get('count') if self._last_ref_info else None
            )
            
            manager.save_metadata(metadata)
        except Exception as e:
            logger.debug(f"Failed to save summary metadata: {e}")
        
        # Update library entry with classification
        if summary_result.classification:
            try:
                from infrastructure.literature.library.index import LibraryIndex
                
                # Ensure config is available
                if not hasattr(self, 'config') or self.config is None:
                    logger.warning(f"[{citation_key}] Config not available, skipping library classification update")
                else:
                    library_index = LibraryIndex(self.config)
                    entry = library_index.get_entry(citation_key)
                    if entry:
                        entry.metadata["classification"] = {
                            "category": summary_result.classification.category,
                            "domain": summary_result.classification.domain,
                            "confidence": summary_result.classification.confidence,
                            "reasoning": summary_result.classification.reasoning
                        }
                        library_index.update_entry(entry)
                        logger.info(
                            f"[{citation_key}] Updated library entry with classification: "
                            f"{summary_result.classification.category} "
                            f"(confidence: {summary_result.classification.confidence:.2f})"
                        )
                    else:
                        logger.warning(
                            f"[{citation_key}] Library entry not found, cannot store classification. "
                            f"Entry may not have been added to library index."
                        )
            except ValueError as e:
                logger.warning(
                    f"[{citation_key}] Cannot update library entry: {e}. "
                    f"Classification will not be stored in library index."
                )
            except Exception as e:
                logger.warning(
                    f"[{citation_key}] Failed to update library entry with classification: {e}. "
                    f"Classification data: category={summary_result.classification.category}, "
                    f"domain={summary_result.classification.domain}"
                )
        
        return saved_paths
