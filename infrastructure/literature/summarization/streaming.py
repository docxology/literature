"""Streaming support for summarization with periodic progress updates.

This module provides streaming wrappers that emit progress events
during LLM generation, allowing real-time feedback on summarization progress.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional, TYPE_CHECKING

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.summarization.models import SummarizationProgressEvent

if TYPE_CHECKING:
    from infrastructure.llm.core.client import LLMClient
    from infrastructure.llm.core.config import GenerationOptions

logger = get_logger(__name__)


def stream_with_progress(
    llm_client: "LLMClient",
    prompt: str,
    progress_callback: Optional[Callable[[SummarizationProgressEvent], None]],
    citation_key: str,
    stage: str = "draft_generation",
    update_interval: float = 5.0,
    timeout: Optional[float] = None,
    options: Optional["GenerationOptions"] = None,
    reset_context: bool = False
) -> str:
    """Stream LLM response with periodic progress updates.
    
    Streams the LLM response and emits progress events at regular intervals
    (default: every 5 seconds) showing accumulated characters, words, chunks,
    and elapsed time.
    
    Args:
        llm_client: LLM client instance with streaming support.
        prompt: Prompt to send to LLM.
        progress_callback: Optional callback function for progress events.
        citation_key: Citation key for the paper being processed.
        stage: Stage name ("draft_generation" or "refinement").
        update_interval: Interval in seconds between progress updates (default: 5.0).
        timeout: Maximum time to wait for response. If None, reads from 
                LLM_SUMMARIZATION_TIMEOUT env var (default: 600.0). Adaptive timeout
                adds 1s per 1000 chars of prompt (max 1200s). Considers model type
                and historical performance for better estimates.
        options: Optional generation options (temperature, max_tokens, etc.).
        reset_context: Whether to clear LLM context before streaming (default: False).
                      Note: Context is already cleared at the start of each paper.
        
    Returns:
        Complete accumulated response text.
        
    Example:
        >>> from infrastructure.llm.core.client import LLMClient
        >>> client = LLMClient()
        >>> 
        >>> def on_progress(event):
        ...     print(f"Progress: {event.metadata['chars_received']} chars")
        >>> 
        >>> response = stream_with_progress(
        ...     client, prompt, on_progress, "paper123", "draft_generation"
        ... )
    """
    # Clear context before streaming if requested
    if reset_context:
        messages_before = len(llm_client.context.messages)
        tokens_before = llm_client.context.estimated_tokens
        logger.debug(
            f"[{citation_key}] Clearing context before streaming {stage}",
            extra={
                "messages_before": messages_before,
                "tokens_before": tokens_before
            }
        )
        llm_client.context.clear()
    
    # Read timeout from environment if not provided
    if timeout is None:
        timeout = float(os.environ.get('LLM_SUMMARIZATION_TIMEOUT', '600.0'))
    
    # Adaptive timeout based on prompt size and model considerations
    # Base: 300s, add 1s per 1000 chars of prompt (max 1200s)
    # For very large prompts (>100KB), add extra buffer
    prompt_size = len(prompt)
    prompt_size_kb = prompt_size / 1024
    prompt_size_mb = prompt_size_kb / 1024
    
    # Base adaptive timeout calculation
    base_adaptive = 300.0 + (prompt_size / 1000.0)
    
    # Add extra buffer for very large prompts
    if prompt_size_mb > 0.1:  # > 100KB
        extra_buffer = min(prompt_size_mb * 10, 300.0)  # Up to 5 minutes extra
        base_adaptive += extra_buffer
    
    adaptive_timeout = min(base_adaptive, 1200.0)
    configured_timeout = timeout
    timeout = max(timeout, adaptive_timeout)  # Use larger of configured or adaptive
    
    # Warn if prompt is very large and timeout might be insufficient
    if prompt_size_mb > 0.1 and timeout < 900:  # Large prompt but short timeout
        logger.warning(
            f"[{citation_key}] Large prompt ({prompt_size_mb:.2f}MB) with relatively short timeout "
            f"({timeout:.0f}s). Consider increasing timeout or reducing prompt size."
        )
    
    accumulated = []
    chars_received = 0
    words_received = 0
    chunks_received = 0
    start_time = time.time()
    last_update = start_time
    last_chunk_time = start_time
    last_heartbeat = start_time
    first_chunk_received = False
    heartbeat_interval = 2.5  # More frequent heartbeats (2.5s) for better feedback
    timeout_warning_threshold = timeout * 0.8  # Warn at 80% of timeout
    timeout_warning_logged = False
    last_timeout_warning_time = start_time
    timeout_warning_interval = 10.0  # Log warning at most once per 10 seconds
    slow_gen_warned = False
    slow_gen_warn_time = start_time
    
    # Estimate processing time based on prompt size
    # Rough estimates: small models ~100 chars/s, medium ~200 chars/s, large ~500 chars/s
    # For prompt processing (before first token), estimate based on prompt size
    estimated_prompt_processing = min(prompt_size / 1000, 30.0)  # Max 30s for prompt processing
    estimated_total_time = estimated_prompt_processing + (prompt_size * 0.1) / 200  # Rough estimate for generation
    
    # Log immediate feedback that streaming is starting
    logger.info(
        f"[{citation_key}] Starting streaming for {stage}... "
        f"(timeout: {timeout:.0f}s, configured: {configured_timeout:.0f}s, "
        f"adaptive: {adaptive_timeout:.0f}s based on {prompt_size:,} char prompt, "
        f"estimated processing: {estimated_prompt_processing:.1f}s)"
    )
    
    # Emit initial progress event with time estimate
    if progress_callback:
        progress_callback(SummarizationProgressEvent(
            citation_key=citation_key,
            stage=stage,
            status="started",
            message=f"Processing prompt ({prompt_size:,} chars, estimated: {estimated_prompt_processing:.1f}s)...",
            metadata={
                "streaming": True,
                "elapsed_time": 0.0,
                "prompt_size": prompt_size,
                "estimated_processing_time": estimated_prompt_processing,
                "estimated_total_time": estimated_total_time
            }
        ))
    
    try:
        # Stream from LLM with generation options
        stream_kwargs = {}
        if options:
            stream_kwargs['options'] = options
        
        for chunk in llm_client.stream_query(prompt, **stream_kwargs):
            now = time.time()
            elapsed = now - start_time
            
            # Check for timeout
            if elapsed > timeout:
                partial_text = "".join(accumulated)
                
                # Save partial result for debugging
                if partial_text and len(partial_text) > 100:
                    try:
                        from pathlib import Path
                        partial_path = Path("data/summaries") / "_debug" / f"{citation_key}_partial_{stage}_{int(time.time())}.md"
                        partial_path.parent.mkdir(parents=True, exist_ok=True)
                        partial_path.write_text(partial_text, encoding='utf-8')
                        logger.warning(
                            f"[{citation_key}] Saved partial result: {partial_path} "
                            f"({len(partial_text):,} chars, {len(partial_text.split())} words)"
                        )
                    except Exception as save_error:
                        logger.debug(f"[{citation_key}] Failed to save partial result: {save_error}")
                
                raise TimeoutError(
                    f"Streaming timeout after {timeout:.1f}s. "
                    f"Received {chunks_received} chunks, {chars_received:,} chars"
                )
            
            # Warn if approaching timeout (throttled to once per 10 seconds)
            if elapsed >= timeout_warning_threshold and elapsed < timeout:
                remaining = timeout - elapsed
                # Throttle warnings: only log once per interval
                if not timeout_warning_logged or (now - last_timeout_warning_time) >= timeout_warning_interval:
                    logger.warning(
                        f"[{citation_key}] Approaching timeout: {remaining:.1f}s remaining. "
                        f"Received {chunks_received} chunks, {chars_received:,} chars so far"
                    )
                    timeout_warning_logged = True
                    last_timeout_warning_time = now
                # After first warning, use DEBUG level for periodic updates
                elif (now - last_timeout_warning_time) >= timeout_warning_interval:
                    logger.debug(
                        f"[{citation_key}] Still approaching timeout: {remaining:.1f}s remaining. "
                        f"Received {chunks_received} chunks, {chars_received:,} chars"
                    )
                    last_timeout_warning_time = now
            
            # Handle empty chunks (waiting for response)
            if not chunk or not chunk.strip():
                # Check for heartbeat if no chunks received for a while
                if not first_chunk_received:
                    # Waiting for first chunk - more frequent heartbeats (every 2.5s)
                    if elapsed >= heartbeat_interval and now - last_heartbeat >= heartbeat_interval:
                        remaining_estimate = max(0, estimated_prompt_processing - elapsed)
                        logger.info(
                            f"[{citation_key}] Processing prompt... "
                            f"({elapsed:.1f}s elapsed, prompt: {len(prompt):,} chars, "
                            f"estimated remaining: {remaining_estimate:.1f}s)"
                        )
                        # Emit progress event
                        if progress_callback:
                            progress_callback(SummarizationProgressEvent(
                                citation_key=citation_key,
                                stage=stage,
                                status="in_progress",
                                message=f"Processing prompt... ({elapsed:.1f}s elapsed, estimated remaining: {remaining_estimate:.1f}s)",
                                metadata={
                                    "streaming": True,
                                    "elapsed_time": elapsed,
                                    "prompt_size": len(prompt),
                                    "estimated_remaining": remaining_estimate,
                                    "waiting_for_first_chunk": True
                                }
                            ))
                        last_heartbeat = now
                else:
                    # Already received chunks, less frequent heartbeats
                    if now - last_chunk_time >= heartbeat_interval * 2 and now - last_heartbeat >= heartbeat_interval * 2:
                        logger.info(
                            f"[{citation_key}] Waiting for more LLM response... "
                            f"({elapsed:.1f}s elapsed, {chunks_received} chunks, {chars_received:,} chars received so far)"
                        )
                        last_heartbeat = now
                continue
                
            # Chunk received - update tracking
            if not first_chunk_received:
                first_chunk_received = True
                time_to_first_chunk = elapsed
                logger.info(
                    f"[{citation_key}] First chunk received after {time_to_first_chunk:.2f}s "
                    f"(chunk size: {len(chunk)} chars)"
                )
                # Emit immediate progress event when first chunk arrives
                if progress_callback:
                    progress_callback(SummarizationProgressEvent(
                        citation_key=citation_key,
                        stage=stage,
                        status="in_progress",
                        message=f"LLM started generating (first chunk: {len(chunk)} chars)",
                        metadata={
                            "chars_received": len(chunk),
                            "words_received": len(chunk.split()),
                            "chunks_received": 1,
                            "elapsed_time": elapsed,
                            "streaming": True,
                            "first_chunk": True
                        }
                    ))
            
            last_chunk_time = now
            accumulated.append(chunk)
            chars_received += len(chunk)
            # Count words in this chunk
            chunk_words = len(chunk.split())
            words_received += chunk_words
            chunks_received += 1
            
            # Calculate chunk rate for monitoring
            if elapsed > 0:
                chars_per_sec = chars_received / elapsed
                chunks_per_sec = chunks_received / elapsed
            else:
                chars_per_sec = 0.0
                chunks_per_sec = 0.0
            
            # Performance monitoring: warn if generation rate is very slow
            if chars_per_sec > 0 and chars_per_sec < 10.0 and elapsed > 30.0:
                # Estimate remaining based on max_tokens if available
                if options and hasattr(options, 'max_tokens') and options.max_tokens:
                    estimated_total_chars = options.max_tokens * 4  # Rough estimate: 4 chars per token
                    estimated_remaining_chars = max(0, estimated_total_chars - chars_received)
                    estimated_remaining_time = estimated_remaining_chars / chars_per_sec if chars_per_sec > 0 else float('inf')
                    
                    if estimated_remaining_time > timeout - elapsed:
                        # Only warn once per 30 seconds to avoid spam
                        if not slow_gen_warned or (now - slow_gen_warn_time) >= 30.0:
                            # Calculate prompt size for suggestions
                            prompt_size_kb = len(prompt) / 1024
                            prompt_size_mb = prompt_size_kb / 1024
                            
                            # Generate actionable suggestions
                            suggestions = []
                            if prompt_size_mb > 0.1:  # > 100KB
                                suggestions.append(f"reduce prompt size (current: {prompt_size_mb:.2f}MB)")
                            if timeout < 1200:  # Less than 20 minutes
                                suggestions.append(f"increase timeout (current: {timeout:.0f}s)")
                            suggestions.append("check LLM model performance/availability")
                            suggestions.append("verify network connectivity if using remote LLM")
                            
                            suggestion_text = "; ".join(suggestions) if suggestions else "check system resources"
                            
                            logger.warning(
                                f"[{citation_key}] Slow generation rate: {chars_per_sec:.1f} chars/s. "
                                f"At this rate, completion would take {estimated_remaining_time:.0f}s more "
                                f"(exceeds remaining timeout: {timeout - elapsed:.0f}s). "
                                f"Suggestions: {suggestion_text}."
                            )
                            
                            # Emit structured progress event with metrics
                            if progress_callback:
                                progress_callback(SummarizationProgressEvent(
                                    citation_key=citation_key,
                                    stage=stage,
                                    status="warning",
                                    message=f"Slow generation rate detected: {chars_per_sec:.1f} chars/s",
                                    metadata={
                                        "chars_per_sec": chars_per_sec,
                                        "estimated_remaining_time": estimated_remaining_time,
                                        "remaining_timeout": timeout - elapsed,
                                        "prompt_size_chars": len(prompt),
                                        "prompt_size_kb": prompt_size_kb,
                                        "suggestions": suggestions,
                                        "warning_type": "slow_generation"
                                    }
                                ))
                            
                            slow_gen_warned = True
                            slow_gen_warn_time = now
            
            # Emit progress every update_interval seconds
            if now - last_update >= update_interval:
                # Calculate ETA if we have enough data
                eta_seconds = None
                if chars_per_sec > 0 and options and hasattr(options, 'max_tokens') and options.max_tokens:
                    estimated_total_chars = options.max_tokens * 4
                    estimated_remaining_chars = max(0, estimated_total_chars - chars_received)
                    eta_seconds = estimated_remaining_chars / chars_per_sec if chars_per_sec > 0 else None
                
                if progress_callback:
                    metadata = {
                        "chars_received": chars_received,
                        "words_received": words_received,
                        "chunks_received": chunks_received,
                        "elapsed_time": elapsed,
                        "chars_per_sec": chars_per_sec,
                        "words_per_sec": words_received / elapsed if elapsed > 0 else 0.0,
                        "chunks_per_sec": chunks_per_sec,
                        "streaming": True,
                        "prompt_size_chars": len(prompt),
                        "first_chunk_time": time_to_first_chunk if first_chunk_received else None
                    }
                    
                    if eta_seconds:
                        metadata["estimated_remaining_time"] = eta_seconds
                        metadata["estimated_completion_time"] = elapsed + eta_seconds
                    
                    progress_callback(SummarizationProgressEvent(
                        citation_key=citation_key,
                        stage=stage,
                        status="in_progress",
                        message=f"Streaming: {chars_received:,} chars, {words_received:,} words ({elapsed:.1f}s elapsed, {chars_per_sec:.1f} chars/s)" + 
                                (f", ETA: {eta_seconds:.0f}s" if eta_seconds else ""),
                        metadata=metadata
                    ))
                
                # Log at DEBUG level to reduce verbosity (deduplication will handle true duplicates)
                logger.debug(
                    f"[{citation_key}] Streaming progress: {chars_received:,} chars, "
                    f"{words_received:,} words, {chunks_received} chunks "
                    f"({elapsed:.1f}s, {chars_per_sec:.1f} chars/s, {chunks_per_sec:.2f} chunks/s)"
                )
                last_update = now
                last_heartbeat = now  # Reset heartbeat timer on progress update
        
        # Emit final progress event
        final_elapsed = time.time() - start_time
        
        # Join all chunks into final text
        full_response = "".join(accumulated)
        
        # Validate response is non-empty
        if not full_response or not full_response.strip():
            error_msg = f"Empty response received from LLM after {final_elapsed:.2f}s ({chunks_received} chunks)"
            logger.error(f"[{citation_key}] {error_msg}")
            if progress_callback:
                progress_callback(SummarizationProgressEvent(
                    citation_key=citation_key,
                    stage=stage,
                    status="failed",
                    message=error_msg,
                    metadata={
                        "chars_received": chars_received,
                        "words_received": words_received,
                        "chunks_received": chunks_received,
                        "elapsed_time": final_elapsed,
                        "error": "empty_response"
                    }
                ))
            raise ValueError(error_msg)
        
        # Calculate final rates
        if final_elapsed > 0:
            final_chars_per_sec = len(full_response) / final_elapsed
            final_words_per_sec = len(full_response.split()) / final_elapsed
        else:
            final_chars_per_sec = 0.0
            final_words_per_sec = 0.0
        
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage=stage,
                status="completed",
                message=f"Streaming completed: {len(full_response):,} chars, {len(full_response.split()):,} words ({final_elapsed:.1f}s)",
                metadata={
                    "chars_received": len(full_response),
                    "words_received": len(full_response.split()),
                    "chunks_received": chunks_received,
                    "elapsed_time": final_elapsed,
                    "chars_per_sec": final_chars_per_sec,
                    "words_per_sec": final_words_per_sec,
                    "streaming": True,
                    "final": True
                }
            ))
        
        logger.info(
            f"[{citation_key}] Streaming completed: {len(full_response):,} chars, "
            f"{len(full_response.split()):,} words, {chunks_received} chunks "
            f"({final_elapsed:.2f}s, {final_chars_per_sec:.1f} chars/s, {final_words_per_sec:.1f} words/s)"
        )
        
        return full_response
        
    except Exception as e:
        elapsed = time.time() - start_time
        import traceback
        
        # Enhanced error logging with context
        error_context = {
            "elapsed_time": elapsed,
            "chunks_received": chunks_received,
            "chars_received": chars_received,
            "words_received": words_received,
            "first_chunk_received": first_chunk_received,
            "prompt_size": len(prompt)
        }
        
        logger.error(
            f"[{citation_key}] Streaming failed after {elapsed:.2f}s: {e}\n"
            f"Context: {error_context}\n"
            f"Traceback: {traceback.format_exc()}"
        )
        
        if progress_callback:
            progress_callback(SummarizationProgressEvent(
                citation_key=citation_key,
                stage=stage,
                status="failed",
                message=f"Streaming failed: {e}",
                metadata={
                    "chars_received": chars_received,
                    "words_received": words_received,
                    "chunks_received": chunks_received,
                    "elapsed_time": elapsed,
                    "first_chunk_received": first_chunk_received,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            ))
        
        raise


