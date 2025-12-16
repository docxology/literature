"""Core embedding generation functionality.

Handles three-phase embedding generation:
1. Check cache for existing embeddings
2. Load cached embeddings
3. Generate missing embeddings with checkpoint resume support
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from requests.exceptions import Timeout

from typing import Any

from infrastructure.core.logging_utils import get_logger, format_duration
from infrastructure.core.progress import ProgressBar
from infrastructure.llm.core.embedding_client import EmbeddingClient, OLLAMA_UTILS_AVAILABLE
from infrastructure.llm.core.config import LLMConfig
from infrastructure.literature.core.config import LiteratureConfig
from .data import EmbeddingData
from .checkpoint import save_checkpoint, load_checkpoint, delete_checkpoint, get_checkpoint_path
from .shutdown import setup_signal_handlers, update_shutdown_state

logger = get_logger(__name__)


def check_cached_embeddings(
    texts: List[str],
    citation_keys: List[str],
    client: EmbeddingClient,
    show_progress: bool = True
) -> Tuple[List[int], List[int]]:
    """Check which embeddings are already cached.
    
    Args:
        texts: List of texts to check for cached embeddings.
        citation_keys: Citation keys for each text (for logging).
        client: EmbeddingClient instance with cache configured.
        show_progress: Whether to show progress during cache checking.
        
    Returns:
        Tuple of (cached_indices, missing_indices) where indices refer to
        positions in the texts list.
    """
    cached_indices = []
    missing_indices = []
    
    if not client.cache_dir:
        # No cache directory, all are missing
        return cached_indices, list(range(len(texts)))
    
    total = len(texts)
    start_time = time.time()
    
    # Use ProgressBar for large sets, periodic logging for smaller sets
    progress_bar = None
    if show_progress and total > 50:
        progress_bar = ProgressBar(
            total=total,
            task="Checking cache",
            show_eta=True
        )
    
    for idx, text in enumerate(texts):
        cached = client._load_from_cache(text)
        if cached is not None:
            cached_indices.append(idx)
        else:
            missing_indices.append(idx)
        
        # Update progress
        if progress_bar:
            progress_bar.update(idx + 1)
        elif show_progress and (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = total - (idx + 1)
            eta = remaining / rate if rate > 0 else 0
            logger.info(
                f"Checking cache: {idx + 1}/{total} ({100 * (idx + 1) // total}%) "
                f"| Elapsed: {format_duration(elapsed)}"
                + (f" | ETA: {format_duration(eta)}" if eta > 0 else "")
            )
    
    if progress_bar:
        progress_bar.finish()
    
    elapsed = time.time() - start_time
    if show_progress:
        logger.info(
            f"Cache check complete: {len(cached_indices)} cached, {len(missing_indices)} missing "
            f"in {format_duration(elapsed)} (avg: {elapsed/total:.3f}s per check)"
        )
    
    return cached_indices, missing_indices


def generate_document_embeddings(
    corpus: Any,  # TextCorpus - imported at runtime to avoid circular import
    config: Optional[LiteratureConfig] = None,
    embedding_model: Optional[str] = None,
    use_cache: bool = True,
    show_progress: bool = True
) -> EmbeddingData:
    """Generate embeddings for all documents in corpus.
    
    Handles three-phase embedding generation:
    1. Check cache for existing embeddings
    2. Load cached embeddings
    3. Generate missing embeddings with checkpoint resume support
    
    Documents that exceed embedding_max_text_length are skipped and assigned zero vectors.
    Failed embeddings also receive zero vectors as fallback.
    Zero vectors are expected and valid - they represent skipped or failed documents.
    
    Args:
        corpus: Text corpus from aggregator (TextCorpus type).
        config: Literature configuration (uses default if not provided).
        embedding_model: Embedding model name (uses config default if not provided).
        use_cache: Whether to use cached embeddings if available.
        show_progress: Whether to log progress.
        
    Returns:
        EmbeddingData with embeddings for all documents. Zero vectors are used for
        skipped documents (exceeded length limit) and failed embeddings.
        
    Raises:
        ValueError: If corpus is empty or has insufficient data.
    """
    # Import at runtime to avoid circular import
    from infrastructure.literature.meta_analysis.aggregator import TextCorpus
    
    if config is None:
        config = LiteratureConfig.from_env()
    
    if embedding_model is None:
        embedding_model = config.embedding_model
    
    # Validate corpus
    if not corpus.texts and not corpus.abstracts:
        raise ValueError("Corpus is empty: no texts or abstracts available")
    
    # Prepare texts for embedding (use full extracted text if available, else abstract)
    texts_to_embed = []
    valid_indices = []
    
    for idx in range(len(corpus.citation_keys)):
        # Prefer extracted text, fallback to abstract
        text = corpus.texts[idx] if idx < len(corpus.texts) and corpus.texts[idx].strip() else ""
        if not text and idx < len(corpus.abstracts):
            text = corpus.abstracts[idx] if corpus.abstracts[idx] else ""
        
        if text.strip():
            texts_to_embed.append(text)
            valid_indices.append(idx)
    
    if len(texts_to_embed) == 0:
        raise ValueError("No valid texts found for embedding")
    
    total_documents = len(texts_to_embed)
    overall_start_time = time.time()
    
    # Enhanced initial status message
    logger.info("")
    logger.info("=" * 60)
    logger.info("EMBEDDING GENERATION")
    logger.info("=" * 60)
    logger.info(f"Total documents: {total_documents}")
    logger.info(f"Model: {embedding_model}")
    cache_dir = Path(config.embedding_cache_dir) if config.embedding_cache_dir else None
    if cache_dir:
        logger.info(f"Cache directory: {cache_dir}")
    else:
        logger.info("Cache: Disabled")
    logger.info("")
    logger.info("Phases:")
    logger.info("  1/3: Checking embedding cache")
    logger.info("  2/3: Loading cached embeddings")
    logger.info("  3/3: Generating missing embeddings")
    logger.info("")
    
    # Initialize embedding client
    llm_config = LLMConfig.from_env()
    
    client = EmbeddingClient(
        config=llm_config,
        embedding_model=embedding_model,
        cache_dir=cache_dir,
        chunk_size=config.embedding_chunk_size,
        batch_size=config.embedding_batch_size,
        timeout=config.embedding_timeout,
        retry_attempts=config.embedding_retry_attempts,
        retry_delay=config.embedding_retry_delay,
        restart_ollama_on_timeout=config.embedding_restart_ollama_on_timeout
    )
    # Set chunk size reduction threshold for long documents
    client._chunk_size_reduction_threshold = config.embedding_chunk_size_reduction_threshold
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Initialize shutdown state for checkpoint saving
    update_shutdown_state(cache_dir=cache_dir, client=client)
    
    # Check for checkpoint first - if exists, skip validation (resume mode)
    checkpoint_data = None
    completed_from_checkpoint = set()
    if cache_dir and use_cache:
        checkpoint_data = load_checkpoint(cache_dir)
        if checkpoint_data:
            # Verify checkpoint matches current corpus
            checkpoint_keys = checkpoint_data.get("citation_keys", [])
            current_keys = [corpus.citation_keys[i] for i in valid_indices]
            
            if checkpoint_keys == current_keys:
                logger.info("Valid checkpoint found - skipping validation (resume mode)")
                logger.info("")
                # Skip validation when resuming from checkpoint
            else:
                logger.warning("Checkpoint citation keys don't match current corpus - will validate")
                delete_checkpoint(cache_dir)
                checkpoint_data = None
    
    # Pre-flight Ollama validation (only if not resuming from checkpoint)
    if not checkpoint_data:
        logger.info("Pre-flight Ollama validation...")
        is_available, error_msg = client.check_connection(timeout=5.0)
        if not is_available:
            logger.warning(f"Cannot connect to Ollama: {error_msg}")
            if config.embedding_restart_ollama_on_timeout:
                logger.info("Attempting to restart and validate Ollama...")
                validation_success = client._ensure_ollama_ready()
                if validation_success:
                    logger.info("Ollama validated successfully after restart")
                else:
                    logger.warning("Ollama validation failed after restart attempt")
                    logger.warning("Continuing anyway - will attempt restart on first embedding if needed")
                    # Non-blocking - allow process to continue
            else:
                logger.warning("Ollama not available but restart disabled")
                logger.warning("Continuing anyway - will fail on first embedding if Ollama is truly unavailable")
                # Non-blocking - allow process to continue
        else:
            # Even if connected, validate functionality (non-blocking)
            logger.info("Ollama connection OK, validating functionality...")
            validation_success = client._ensure_ollama_ready()
            if validation_success:
                logger.info("Ollama validation complete - ready for embedding generation")
            else:
                logger.warning("Ollama validation test failed or timed out")
                logger.warning("Continuing anyway - validation test may be slow but real embeddings may work")
                # Non-blocking - allow process to continue
        
        logger.info("")
    
    # Phase 1/3: Check which embeddings are already cached
    cached_indices = []
    missing_indices = []
    cache_check_time = None
    
    if use_cache and client.cache_dir:
        logger.info("Phase 1/3: Checking embedding cache...")
        cache_check_start = time.time()
        cached_indices, missing_indices = check_cached_embeddings(
            texts_to_embed,
            [corpus.citation_keys[i] for i in valid_indices],
            client,
            show_progress=show_progress
        )
        cache_check_time = time.time() - cache_check_start
        n_cached = len(cached_indices)
        n_missing = len(missing_indices)
        logger.info(
            f"Phase 1/3 complete: {n_cached} embeddings cached, {n_missing} need generation "
            f"({format_duration(cache_check_time)})"
        )
        logger.info("")
    else:
        missing_indices = list(range(total_documents))
        logger.info(f"Phase 1/3: Cache disabled or not configured, generating {total_documents} embeddings")
        logger.info("")
    
    # Phase 2/3: Load cached embeddings first
    embeddings_list: List[Optional[np.ndarray]] = [None] * total_documents
    cached_count = 0
    failed_cache_indices = []
    cache_load_time = None
    
    if cached_indices:
        logger.info(f"Phase 2/3: Loading {len(cached_indices)} cached embeddings...")
        cache_load_start = time.time()
        
        # Use ProgressBar for loading cached embeddings
        cache_progress_bar = None
        if show_progress and len(cached_indices) > 10:
            cache_progress_bar = ProgressBar(
                total=len(cached_indices),
                task="Loading cached",
                show_eta=True
            )
        
        for progress_idx, idx in enumerate(cached_indices, 1):
            try:
                embedding = client._load_from_cache(texts_to_embed[idx])
                if embedding is not None:
                    embeddings_list[idx] = embedding
                    cached_count += 1
                    
                    if cache_progress_bar:
                        cache_progress_bar.update(progress_idx)
                    elif show_progress and progress_idx % 50 == 0:
                        logger.info(f"  Loaded {progress_idx}/{len(cached_indices)} cached embeddings...")
                else:
                    failed_cache_indices.append(idx)
            except Exception as e:
                citation_key = corpus.citation_keys[valid_indices[idx]]
                logger.warning(f"Failed to load cached embedding for {citation_key}: {e}")
                failed_cache_indices.append(idx)
        
        if cache_progress_bar:
            cache_progress_bar.finish()
        
        cache_load_time = time.time() - cache_load_start
        logger.info(
            f"Phase 2/3 complete: Loaded {cached_count} cached embeddings "
            f"({format_duration(cache_load_time)}, "
            f"avg: {cache_load_time/len(cached_indices):.3f}s per embedding)"
        )
        
        # Add failed cache loads to missing indices for regeneration
        missing_indices.extend(failed_cache_indices)
        if failed_cache_indices:
            logger.info(f"  {len(failed_cache_indices)} cached embeddings failed to load, will regenerate")
        
        logger.info("")
        # Force flush logs to ensure visibility
        sys.stdout.flush()
        if hasattr(logger, 'handlers'):
            for handler in logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
    
    # Immediate diagnostic logging after Phase 2/3 (always executed)
    # Use both print() and logger.info() to ensure visibility
    print("=" * 60)
    print("Phase 2/3 Complete - Preparing for Phase 3/3")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("Phase 2/3 Complete - Preparing for Phase 3/3")
    logger.info("=" * 60)
    print(f"Missing indices count: {len(missing_indices)}")
    logger.info(f"Missing indices count: {len(missing_indices)}")
    print(f"Checkpoint data available: {'Yes' if checkpoint_data else 'No'}")
    logger.info(f"Checkpoint data available: {'Yes' if checkpoint_data else 'No'}")
    if checkpoint_data:
        checkpoint_completed = len(checkpoint_data.get("completed_indices", []))
        print(f"  Checkpoint indicates {checkpoint_completed} completed embeddings")
        logger.info(f"  Checkpoint indicates {checkpoint_completed} completed embeddings")
    else:
        print("  No checkpoint found - starting fresh")
        logger.info("  No checkpoint found - starting fresh")
    print(f"Phase 3/3 will start: {'Yes' if missing_indices else 'No (all embeddings complete)'}")
    logger.info(f"Phase 3/3 will start: {'Yes' if missing_indices else 'No (all embeddings complete)'}")
    print("")
    logger.info("")
    # Force flush after diagnostic logging
    sys.stdout.flush()
    
    # Pre-flight embedding test: verify Ollama can generate embeddings before Phase 3/3
    # Only run if Phase 3/3 will actually execute (missing_indices not empty)
    if missing_indices:
        print("=" * 60)
        print("Pre-Flight Embedding Test")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("Pre-Flight Embedding Test")
        logger.info("=" * 60)
        print("Testing Ollama embedding generation with actual embedding test...")
        logger.info("Testing Ollama embedding generation with actual embedding test...")
        sys.stdout.flush()
        
        # Test actual embedding generation (not just endpoint check)
        test_passed = False
        test_success, test_embedding, test_message = client.test_embedding_generation()
        
        if test_success:
            test_passed = True
            print("  ✓ Pre-flight embedding generation test passed - Ollama is ready for Phase 3/3")
            logger.info("  ✓ Pre-flight embedding generation test passed - Ollama is ready for Phase 3/3")
            print(f"  {test_message}")
            logger.info(f"  {test_message}")
            # Reset client state after successful test
            client._consecutive_timeouts = 0
        else:
            print(f"  ✗ Pre-flight embedding generation test failed: {test_message}")
            logger.warning(f"  ✗ Pre-flight embedding generation test failed: {test_message}")
            print("  Attempting Ollama restart...")
            logger.warning("  Attempting Ollama restart...")
            sys.stdout.flush()
            
            # Attempt restart if enabled
            if config.embedding_restart_ollama_on_timeout and OLLAMA_UTILS_AVAILABLE:
                try:
                    from infrastructure.llm.utils.ollama import restart_ollama_server
                    success, status_msg = restart_ollama_server(
                        base_url=client.config.base_url,
                        kill_existing=True,
                        wait_seconds=8.0,
                        test_embedding=True,
                        embedding_model=client.embedding_model
                    )
                    if success:
                        print(f"  Ollama restarted: {status_msg}")
                        logger.info(f"  Ollama restarted: {status_msg}")
                        # Reset client state after restart
                        client._consecutive_timeouts = 0
                        print("  Retrying pre-flight embedding generation test...")
                        logger.info("  Retrying pre-flight embedding generation test...")
                        sys.stdout.flush()
                        
                        # Retry actual embedding generation test after restart
                        test_success_retry, test_embedding_retry, test_message_retry = client.test_embedding_generation()
                        if test_success_retry:
                            test_passed = True
                            print("  ✓ Pre-flight embedding generation test passed after restart - Ollama is ready for Phase 3/3")
                            logger.info("  ✓ Pre-flight embedding generation test passed after restart - Ollama is ready for Phase 3/3")
                            print(f"  {test_message_retry}")
                            logger.info(f"  {test_message_retry}")
                            # Reset client state after successful retry
                            client._consecutive_timeouts = 0
                        else:
                            print(f"  ✗ Pre-flight embedding generation test still failed after restart: {test_message_retry}")
                            logger.warning(f"  ✗ Pre-flight embedding generation test still failed after restart: {test_message_retry}")
                    else:
                        print(f"  Ollama restart failed: {status_msg}")
                        logger.warning(f"  Ollama restart failed: {status_msg}")
                except Exception as e:
                    print(f"  Failed to restart Ollama: {e}")
                    logger.warning(f"  Failed to restart Ollama: {e}")
            
            # If test still failed, warn but continue (non-blocking)
            if not test_passed:
                print("  ⚠ Warning: Pre-flight test failed, but proceeding to Phase 3/3 anyway")
                print("  ⚠ Embeddings may timeout or fail - consider checking Ollama status manually")
                logger.warning("  ⚠ Warning: Pre-flight test failed, but proceeding to Phase 3/3 anyway")
                logger.warning("  ⚠ Embeddings may timeout or fail - consider checking Ollama status manually")
        
        print("")
        logger.info("")
        sys.stdout.flush()
    
    # Load checkpoint data and resume if available (after Phase 2)
    checkpoint_loaded_count = 0
    if cache_dir and use_cache and checkpoint_data:
        # Checkpoint already validated above
        completed_from_checkpoint = set(checkpoint_data.get("completed_indices", []))
        checkpoint_timestamp = checkpoint_data.get("timestamp", "unknown")
        checkpoint_path = get_checkpoint_path(cache_dir)
        
        print("")
        print("=" * 60)
        print("CHECKPOINT RESUME MODE")
        print("=" * 60)
        logger.info("")
        logger.info("=" * 60)
        logger.info("CHECKPOINT RESUME MODE")
        logger.info("=" * 60)
        print(f"Checkpoint file: {checkpoint_path}")
        logger.info(f"Checkpoint file: {checkpoint_path}")
        print(f"Checkpoint timestamp: {checkpoint_timestamp}")
        logger.info(f"Checkpoint timestamp: {checkpoint_timestamp}")
        print(f"Completed in checkpoint: {len(completed_from_checkpoint)} embeddings")
        logger.info(f"Completed in checkpoint: {len(completed_from_checkpoint)} embeddings")
        print("")
        logger.info("")
        print(f"Loading {len(completed_from_checkpoint)} checkpointed embeddings from cache...")
        logger.info(f"Loading {len(completed_from_checkpoint)} checkpointed embeddings from cache...")
        sys.stdout.flush()
        
        # Load completed embeddings from cache (they should be cached)
        checkpoint_load_start = time.time()
        logger.info(f"Starting to load {len(completed_from_checkpoint)} checkpointed embeddings...")
        sys.stdout.flush()
        
        for progress_idx, idx in enumerate(completed_from_checkpoint, 1):
            if idx < len(texts_to_embed):
                # Track time for individual embedding load
                load_start = time.time()
                
                try:
                    # Log first embedding
                    if progress_idx == 1:
                        logger.info(f"  Loading checkpointed embedding 1/{len(completed_from_checkpoint)}...")
                        sys.stdout.flush()
                    
                    embedding = client._load_from_cache(texts_to_embed[idx])
                    load_time = time.time() - load_start
                    
                    # Timeout detection for individual loads
                    if load_time > 5.0:
                        logger.warning(
                            f"  Loading checkpointed embedding {progress_idx}/{len(completed_from_checkpoint)} "
                            f"took {load_time:.1f}s (longer than expected)"
                        )
                        sys.stdout.flush()
                    
                    if embedding is not None:
                        embeddings_list[idx] = embedding
                        checkpoint_loaded_count += 1
                        
                        # Log progress every 5 embeddings (more frequent)
                        if progress_idx % 5 == 0 or progress_idx == len(completed_from_checkpoint):
                            elapsed = time.time() - checkpoint_load_start
                            rate = progress_idx / elapsed if elapsed > 0 else 0
                            remaining = len(completed_from_checkpoint) - progress_idx
                            eta = remaining / rate if rate > 0 else 0
                            logger.info(
                                f"  Loading checkpointed embeddings: {progress_idx}/{len(completed_from_checkpoint)} "
                                f"({100*progress_idx/len(completed_from_checkpoint):.1f}%) "
                                f"| Elapsed: {format_duration(elapsed)}"
                                + (f" | ETA: {format_duration(eta)}" if eta > 0 else "")
                            )
                            sys.stdout.flush()
                except Exception as e:
                    load_time = time.time() - load_start
                    logger.warning(
                        f"Failed to load checkpointed embedding {progress_idx}/{len(completed_from_checkpoint)} "
                        f"at index {idx} (took {load_time:.1f}s): {e}"
                    )
                    sys.stdout.flush()
                    # Add back to missing if can't load
                    if idx not in missing_indices:
                        missing_indices.append(idx)
        
        # Overall timeout detection and completion logging
        checkpoint_load_time = time.time() - checkpoint_load_start
        if checkpoint_load_time > 60.0:
            logger.warning(
                f"Checkpoint loading took {format_duration(checkpoint_load_time)} "
                f"(longer than expected - may indicate slow disk I/O)"
            )
        logger.info(f"Checkpoint loading complete: {checkpoint_loaded_count} loaded in {format_duration(checkpoint_load_time)}")
        
        # Validation: check if all checkpointed embeddings were found
        if checkpoint_loaded_count < len(completed_from_checkpoint):
            missing_from_checkpoint = len(completed_from_checkpoint) - checkpoint_loaded_count
            # Collect citation keys for missing embeddings
            missing_citation_keys = []
            for idx in completed_from_checkpoint:
                if idx < len(texts_to_embed) and embeddings_list[idx] is None:
                    if idx < len(valid_indices):
                        missing_citation_keys.append(corpus.citation_keys[valid_indices[idx]])
            
            logger.warning(
                f"Checkpoint indicates {len(completed_from_checkpoint)} completed, "
                f"but only {checkpoint_loaded_count} found in cache. "
                f"{missing_from_checkpoint} will be regenerated."
            )
            if missing_citation_keys:
                if len(missing_citation_keys) <= 10:
                    logger.info(f"  Missing citation keys: {', '.join(missing_citation_keys)}")
                else:
                    logger.info(f"  Missing citation keys (first 10): {', '.join(missing_citation_keys[:10])} and {len(missing_citation_keys) - 10} more")
                logger.info("  Note: This may occur if cache files were deleted or text content changed (cache key is based on text hash)")
        
        # Remove completed indices from missing_indices
        missing_indices_before = len(missing_indices)
        missing_indices = [idx for idx in missing_indices if idx not in completed_from_checkpoint]
        checkpoint_skipped_count = missing_indices_before - len(missing_indices)
        
        logger.info(f"  ✓ Loaded {checkpoint_loaded_count} embeddings from checkpoint")
        logger.info(f"  ✓ Skipped {checkpoint_skipped_count} already-completed embeddings")
        logger.info(f"  → Remaining to generate: {len(missing_indices)} embeddings")
        logger.info("")
        sys.stdout.flush()
    else:
        completed_from_checkpoint = set()
        print("No checkpoint found - starting fresh")
        print("Checkpoint section skipped - proceeding to Phase 3/3")
        logger.info("No checkpoint found - starting fresh")
        logger.info("Checkpoint section skipped - proceeding to Phase 3/3")
        print("")
        logger.info("")
        sys.stdout.flush()
    
    # Phase 3/3: Generate missing embeddings with progress bar
    # Always log entry to Phase 3/3 section - use both print() and logger
    print("=" * 60)
    print("ENTERING PHASE 3/3")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("ENTERING PHASE 3/3")
    logger.info("=" * 60)
    print(f"Missing indices: {len(missing_indices)}")
    logger.info(f"Missing indices: {len(missing_indices)}")
    print(f"Total documents: {total_documents}")
    logger.info(f"Total documents: {total_documents}")
    print("")
    logger.info("")
    sys.stdout.flush()
    
    generation_time = None
    skipped_count = 0  # Initialize for final summary
    failed_count = 0  # Initialize for final summary
    if not missing_indices:
        # All embeddings are complete - log summary and skip Phase 3/3
        logger.info("")
        logger.info("=" * 60)
        logger.info("All Embeddings Complete - No Generation Needed")
        logger.info("=" * 60)
        logger.info(f"Total documents: {total_documents}")
        logger.info(f"  • Cached: {cached_count}")
        logger.info(f"  • Checkpoint loaded: {checkpoint_loaded_count}")
        logger.info(f"  • Total complete: {cached_count + checkpoint_loaded_count} ({100*(cached_count + checkpoint_loaded_count)/total_documents:.1f}%)")
        logger.info("")
        logger.info("Skipping Phase 3/3 - all embeddings are available")
        logger.info("")
    elif missing_indices:
        n_to_generate = len(missing_indices)
        # checkpoint_loaded_count is defined in the checkpoint section above
        n_cached_final = cached_count + checkpoint_loaded_count
        n_total = total_documents
        
        # Initialize counters for tracking skipped documents
        skipped_count = 0
        skipped_citation_keys = []
        
        # Calculate percentages
        pct_cached = (n_cached_final / n_total * 100) if n_total > 0 else 0
        pct_remaining = (n_to_generate / n_total * 100) if n_total > 0 else 0
        
        # Diagnostic summary before Phase 3/3
        logger.info("")
        logger.info("=" * 60)
        logger.info("Diagnostic Summary Before Phase 3/3")
        logger.info("=" * 60)
        logger.info(f"Total documents: {n_total}")
        logger.info(f"  • Cached (Phase 2/3): {cached_count}")
        logger.info(f"  • Checkpoint loaded: {checkpoint_loaded_count}")
        logger.info(f"  • Total complete: {n_cached_final} ({pct_cached:.1f}%)")
        logger.info(f"  • Missing/remaining: {n_to_generate} ({pct_remaining:.1f}%)")
        logger.info(f"Phase 3/3 will proceed: Yes ({n_to_generate} embeddings to generate)")
        logger.info("")
        
        # Detailed Phase 3/3 start message
        logger.info("=" * 60)
        logger.info("Phase 3/3: Generating Missing Embeddings")
        logger.info("=" * 60)
        logger.info(f"Total documents: {n_total}")
        logger.info(f"  • Cached: {n_cached_final} ({pct_cached:.1f}%)")
        logger.info(f"  • Remaining: {n_to_generate} ({pct_remaining:.1f}%)")
        sys.stdout.flush()
        
        # Checkpoint resume status (checkpoint_data is loaded earlier in the function)
        checkpoint_was_loaded = checkpoint_loaded_count > 0
        if checkpoint_was_loaded:
            logger.info(f"  • Checkpoint resume: Yes ({checkpoint_loaded_count} loaded from checkpoint)")
        else:
            logger.info("  • Checkpoint resume: No")
        
        # Model and timeout info
        logger.info(f"Model: {embedding_model}")
        logger.info(f"Base timeout: {config.embedding_timeout}s (adaptive based on text length)")
        logger.info("")
        sys.stdout.flush()
        
        # Show first paper to be processed
        if missing_indices:
            first_idx = missing_indices[0]
            first_citation_key = corpus.citation_keys[valid_indices[first_idx]]
            first_text_length = len(texts_to_embed[first_idx])
            first_timeout = min(config.embedding_timeout, max(30.0, first_text_length / 40.0)) if first_text_length >= 100 else min(config.embedding_timeout, 30.0)
            logger.info(f"Starting with paper 1/{n_to_generate}: {first_citation_key}")
            logger.info(f"  Text length: {first_text_length:,} chars")
            logger.info(f"  Estimated timeout: {first_timeout:.1f}s")
            # Note for longer texts (changed from warning since embeddings are completing successfully)
            if first_text_length > 5000 and first_timeout < 45.0:
                logger.info(
                    f"  Note: Long text ({first_text_length:,} chars), estimated timeout: {first_timeout:.1f}s"
                )
            logger.info("")
            sys.stdout.flush()
        
        generation_start = time.time()
        
        progress_bar = None
        if show_progress:
            progress_bar = ProgressBar(
                total=n_to_generate,
                task="Generating embeddings",
                show_eta=True,
                track_success_failure=True
            )
        
        generated_count = 0
        failed_count = 0
        failed_citation_keys = []
        last_heartbeat_time = time.time()
        heartbeat_interval = 30.0  # Log heartbeat every 30 seconds
        embedding_times = []  # Track timing for each embedding
        close_to_timeout_count = 0  # Track embeddings that took >80% of timeout
        successfully_completed_indices = list(completed_from_checkpoint)  # Track successfully completed embeddings
        skipped_indices = []  # Track skipped document indices
        skipped_citation_keys = []  # Citation keys of skipped documents (re-initialize for this phase)
        failed_citation_keys = []  # Citation keys of failed embeddings (re-initialize for this phase)
        
        # Load skipped indices from checkpoint if available
        if checkpoint_data:
            skipped_from_checkpoint = checkpoint_data.get("skipped_indices", [])
            skipped_indices.extend(skipped_from_checkpoint)
            # Remove skipped indices from missing_indices so they're not attempted
            missing_indices = [idx for idx in missing_indices if idx not in skipped_from_checkpoint]
            if skipped_from_checkpoint:
                logger.info(f"  → Resuming: {len(skipped_from_checkpoint)} documents were previously skipped and will be skipped again")
        
        # Log before starting the loop
        print("")
        print("Starting embedding generation loop...")
        print(f"Will process {len(missing_indices)} documents")
        logger.info("")
        logger.info("Starting embedding generation loop...")
        logger.info(f"Will process {len(missing_indices)} documents")
        sys.stdout.flush()
        
        for progress_idx, text_idx in enumerate(missing_indices, 1):
            # Log at the very start of each iteration (before any processing)
            print(f"Processing document {progress_idx}/{n_to_generate}...")
            logger.info(f"Processing document {progress_idx}/{n_to_generate}...")
            sys.stdout.flush()
            
            citation_key = corpus.citation_keys[valid_indices[text_idx]]
            text_length = len(texts_to_embed[text_idx])
            item_start_time = time.time()
            retry_after_restart = False  # Flag to retry after successful restart
            
            # Check if document exceeds maximum text length limit
            if text_length > config.embedding_max_text_length:
                skipped_count += 1
                skipped_citation_keys.append(citation_key)
                # Set zero vector immediately for skipped documents
                if generated_count > 0:
                    # Use dimension from a successfully generated embedding
                    for existing_emb in embeddings_list:
                        if existing_emb is not None:
                            embeddings_list[text_idx] = np.zeros_like(existing_emb)
                            break
                    else:
                        embeddings_list[text_idx] = np.zeros(config.embedding_dimension)
                else:
                    embeddings_list[text_idx] = np.zeros(config.embedding_dimension)
                
                # Track skipped index
                if text_idx not in skipped_indices:
                    skipped_indices.append(text_idx)
                # Add to successfully_completed_indices so it's tracked as "handled"
                if text_idx not in successfully_completed_indices:
                    successfully_completed_indices.append(text_idx)
                
                logger.warning(
                    f"[{progress_idx}/{n_to_generate}] Skipping document {citation_key}: "
                    f"text length ({text_length:,} chars) exceeds maximum "
                    f"({config.embedding_max_text_length:,} chars)"
                )
                
                # Save checkpoint after skipping
                if cache_dir and use_cache:
                    save_checkpoint(
                        cache_dir,
                        [corpus.citation_keys[i] for i in valid_indices],
                        sorted(successfully_completed_indices),
                        [],
                        total_documents,
                        sorted(skipped_indices)
                    )
                    logger.debug(f"Checkpoint saved: {len(successfully_completed_indices)}/{total_documents} handled ({len(skipped_indices)} skipped)")
                
                if progress_bar:
                    progress_bar.update(progress_idx, success=False, item_time=0.0)
                continue
            
            # Calculate adaptive timeout for this text
            # Unified formula: min(config_timeout, max(30s, text_length/40))
            # For very long documents (>100K chars), apply multiplier
            if text_length < 100:
                estimated_timeout = min(config.embedding_timeout, 30.0)
            elif text_length > 100000:
                # Very long documents: use multiplier for timeout
                base_timeout = max(30.0, text_length / 40.0)
                multiplier = config.embedding_timeout_multiplier_for_long_docs
                estimated_timeout = min(config.embedding_timeout * multiplier, max(120.0, base_timeout))
            else:
                estimated_timeout = min(config.embedding_timeout, max(30.0, text_length / 40.0))
            
            # Log BEFORE each embedding generation attempt
            print(f"[{progress_idx}/{n_to_generate}] Generating embedding for: {citation_key}")
            logger.info(
                f"[{progress_idx}/{n_to_generate}] Generating embedding for: {citation_key}"
            )
            print(f"  Text length: {text_length:,} chars | Estimated timeout: {estimated_timeout:.1f}s")
            logger.info(
                f"  Text length: {text_length:,} chars | "
                f"Estimated timeout: {estimated_timeout:.1f}s"
            )
            
            # Log chunk count for long documents (if chunking will occur)
            estimated_chunks = 1
            if text_length > 4000:
                # Estimate chunk count (approximate: 1 token ≈ 4 chars, chunk_size tokens per chunk)
                estimated_chunks = max(1, int(text_length / (config.embedding_chunk_size * 4)))
                if text_length > config.embedding_chunk_size_reduction_threshold:
                    # Use reduced chunk size for very long documents
                    reduced_chunk_size = max(500, config.embedding_chunk_size // 2)
                    estimated_chunks = max(1, int(text_length / (reduced_chunk_size * 4)))
                if estimated_chunks > 1:
                    estimated_total_time = estimated_timeout * estimated_chunks
                    logger.info(
                        f"  Document will be chunked: ~{estimated_chunks} chunks estimated | "
                        f"Estimated total time: {estimated_total_time:.1f}s"
                    )
                    print(f"  Starting chunk processing for {citation_key}...")
                    logger.info(f"  Starting chunk processing for {citation_key}...")
            
            sys.stdout.flush()
            
            # Note for longer texts (changed from warning to info since embeddings are completing successfully)
            # Only log as info for very long texts, not as a warning
            if text_length > 5000 and estimated_timeout < 45.0:
                logger.info(
                    f"  Note: Long text ({text_length:,} chars), estimated timeout: {estimated_timeout:.1f}s"
                )
            
            try:
                # Use embed_document for full text (handles chunking automatically)
                # Note: embed_document has built-in retry logic and Ollama restart handling
                # Note: embed_document now logs chunk progress automatically
                print(f"  → Calling Ollama embedding API for {citation_key}...")
                logger.info(f"Calling Ollama embedding API for {citation_key}...")
                if estimated_chunks > 1:
                    print(f"  → Document will be processed in ~{estimated_chunks} chunks (progress will be shown per chunk)")
                    logger.info(f"  Processing {estimated_chunks} chunks (progress will be logged per chunk)...")
                sys.stdout.flush()
                embedding_start_time = time.time()
                
                # Check and log heartbeat before starting embedding (if enough time has passed)
                current_time = time.time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    elapsed = current_time - generation_start
                    logger.info(
                        f"  Heartbeat: Starting embedding {progress_idx}/{n_to_generate} "
                        f"({generated_count} completed, {format_duration(elapsed)} elapsed)"
                    )
                    sys.stdout.flush()
                    last_heartbeat_time = current_time
                
                # Call embed_document (handles retries and Ollama restarts internally)
                # Note: embed_document automatically saves to cache if use_cache=True and cache_dir is set
                embedding = client.embed_document(texts_to_embed[text_idx], use_cache=use_cache)
                embeddings_list[text_idx] = embedding
                generated_count += 1
                item_time = time.time() - item_start_time
                
                # Log per-file timing with comparison to estimated timeout
                timeout_ratio = item_time / estimated_timeout if estimated_timeout > 0 else 0
                embedding_times.append(item_time)
                
                # Track embeddings that took >80% of timeout
                if timeout_ratio > 0.8:
                    close_to_timeout_count += 1
                
                if timeout_ratio < 0.8:
                    logger.info(
                        f"  ✓ Embedding completed in {item_time:.1f}s "
                        f"(estimated timeout: {estimated_timeout:.1f}s, {100*timeout_ratio:.0f}% of timeout)"
                    )
                elif timeout_ratio > 0.9:
                    logger.warning(
                        f"  ⚠ Embedding completed in {item_time:.1f}s "
                        f"(estimated timeout: {estimated_timeout:.1f}s, {100*timeout_ratio:.0f}% of timeout - close to limit)"
                    )
                else:
                    logger.info(
                        f"  ✓ Embedding completed in {item_time:.1f}s "
                        f"(estimated timeout: {estimated_timeout:.1f}s, {100*timeout_ratio:.0f}% of timeout)"
                    )
                
                # Save checkpoint after each successful embedding
                # Embedding is automatically saved to cache by client.embed_document()
                if cache_dir and use_cache:
                    # Add current successfully completed embedding to tracking list
                    if text_idx not in successfully_completed_indices:
                        successfully_completed_indices.append(text_idx)
                    
                    # Save checkpoint with all successfully completed indices
                    save_checkpoint(
                        cache_dir,
                        [corpus.citation_keys[i] for i in valid_indices],
                        sorted(successfully_completed_indices),
                        [],
                        total_documents,
                        sorted(skipped_indices)
                    )
                    logger.debug(f"Checkpoint saved: {len(successfully_completed_indices)}/{total_documents} embeddings completed ({len(skipped_indices)} skipped)")
                    
                    # Update shutdown state (with error handling to prevent embedding failures)
                    try:
                        update_shutdown_state(
                            cache_dir=cache_dir,
                            citation_keys=[corpus.citation_keys[i] for i in valid_indices],
                            completed_indices=successfully_completed_indices,
                            failed_indices=failed_citation_keys,
                            skipped_indices=skipped_indices,
                            total=total_documents
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update shutdown state (non-critical): {e}")
                        # Continue processing - state update failure shouldn't break embedding generation
                
                if progress_bar:
                    progress_bar.update(progress_idx, success=True, item_time=item_time)
                
                # Log progress every 5 items (reduced from 10)
                if show_progress and generated_count % 5 == 0:
                    elapsed = time.time() - generation_start
                    avg_time = elapsed / generated_count if generated_count > 0 else 0
                    remaining = n_to_generate - generated_count
                    eta = avg_time * remaining if avg_time > 0 else 0
                    
                    # Include timing statistics if available
                    timing_info = ""
                    if embedding_times:
                        recent_avg = sum(embedding_times[-5:]) / min(5, len(embedding_times))
                        timing_info = f", recent avg: {recent_avg:.1f}s"
                    
                    logger.info(
                        f"  Progress: {generated_count}/{n_to_generate} generated "
                        f"({format_duration(elapsed)}, avg: {avg_time:.2f}s/embedding{timing_info}"
                        + (f", ETA: {format_duration(eta)}" if eta > 0 else "") + ")"
                    )
                    sys.stdout.flush()
                
                # Log if embedding took a while (may have included retries/restarts)
                if item_time > 30.0:
                    logger.info(
                        f"  Embedding completed in {item_time:.1f}s "
                        f"(may have included Ollama restart/retry)"
                    )
                
                # Heartbeat for long operations (every 30s)
                current_time = time.time()
                if current_time - last_heartbeat_time >= heartbeat_interval:
                    elapsed = current_time - generation_start
                    logger.info(
                        f"  Heartbeat: {generated_count}/{n_to_generate} embeddings completed "
                        f"({format_duration(elapsed)} elapsed)"
                    )
                    sys.stdout.flush()
                    last_heartbeat_time = current_time
                    
            except Exception as e:
                failed_count += 1
                failed_citation_keys.append(citation_key)
                error_type = type(e).__name__
                item_time = time.time() - item_start_time
                
                logger.error(
                    f"Failed to embed document {citation_key} after all retries "
                    f"({error_type}): {e} (text length: {text_length:,} chars, "
                    f"attempted for {format_duration(item_time)})"
                )
                
                # On timeout, save checkpoint before restart attempt
                if isinstance(e, Exception) and ("timeout" in str(e).lower() or isinstance(e.__cause__, Timeout)):
                    # Save checkpoint before restart attempt
                    if cache_dir and use_cache:
                        save_checkpoint(
                            cache_dir,
                            [corpus.citation_keys[i] for i in valid_indices],
                            sorted(successfully_completed_indices),
                            sorted(failed_citation_keys),
                            total_documents,
                            sorted(skipped_indices)
                        )
                        logger.info(f"Checkpoint saved before restart attempt: {len(successfully_completed_indices)}/{total_documents} completed")
                    
                    if config.embedding_restart_ollama_on_timeout:
                        logger.warning("Timeout detected, testing embedding endpoint and attempting Ollama force restart...")
                        # Check if embedding endpoint is hung
                        embedding_ok, embedding_error = client.check_embedding_endpoint(timeout=10.0)
                        if not embedding_ok:
                            logger.warning(f"Embedding endpoint is hung: {embedding_error}. Force restarting...")
                        
                        # Force restart Ollama with embedding endpoint test
                        if OLLAMA_UTILS_AVAILABLE:
                            try:
                                from infrastructure.llm.utils.ollama import restart_ollama_server
                                success, status_msg = restart_ollama_server(
                                    base_url=client.config.base_url,
                                    kill_existing=True,
                                    wait_seconds=8.0,
                                    test_embedding=True,  # Test embedding endpoint to detect hung state
                                    embedding_model=client.embedding_model
                                )
                                if success:
                                    print(f"  ✓ Ollama force restarted: {status_msg}")
                                    logger.info(f"Ollama force restarted: {status_msg}")
                                    time.sleep(2.0)  # Wait for model to be ready
                                    
                                    # Reset client state after restart
                                    print("  Resetting client state after restart...")
                                    logger.info("Resetting client state after restart...")
                                    client._consecutive_timeouts = 0
                                    # Clear any cached connection state by re-checking connection
                                    client.check_connection(timeout=2.0)
                                    
                                    # Test embedding generation to verify functionality
                                    print("  Testing embedding generation after restart...")
                                    logger.info("Testing embedding generation after restart...")
                                    sys.stdout.flush()
                                    test_success, test_embedding, test_message = client.test_embedding_generation()
                                    if test_success:
                                        print(f"  ✓ Embedding test passed after restart: {test_message}")
                                        logger.info(f"✓ Embedding test passed after restart: {test_message}")
                                        print("  State reset and validation complete, resuming document embedding...")
                                        logger.info("State reset and validation complete, resuming document embedding...")
                                        retry_after_restart = True  # Mark for retry
                                        sys.stdout.flush()
                                    else:
                                        print(f"  ⚠ Embedding test failed after restart: {test_message}")
                                        logger.warning(f"⚠ Embedding test failed after restart: {test_message}")
                                        print("  Continuing with document embeddings anyway...")
                                        logger.warning("Continuing with document embeddings anyway...")
                                        sys.stdout.flush()
                                else:
                                    logger.warning(f"Ollama restart failed: {status_msg}")
                            except Exception as restart_error:
                                logger.warning(f"Failed to restart Ollama: {restart_error}")
                        
                        # Also try standard health check
                        if not client._ensure_ollama_ready():
                            logger.warning("Ollama restart failed, saving checkpoint for manual recovery")
                
                # If restart was successful and test passed, retry the document embedding
                if retry_after_restart:
                    # Increase timeout by 50% for retry after restart
                    increased_timeout = estimated_timeout * 1.5
                    print(f"  Retrying embedding for {citation_key} after successful restart...")
                    logger.info(f"Retrying embedding for {citation_key} after successful restart...")
                    logger.info(f"  Increased timeout: {estimated_timeout:.1f}s → {increased_timeout:.1f}s (+50%)")
                    sys.stdout.flush()
                    
                    # Temporarily increase client timeout for this retry
                    client.timeout = min(increased_timeout, config.embedding_timeout * 2)  # Cap at 2x config timeout
                    
                    try:
                        # Retry with increased timeout
                        retry_start_time = time.time()
                        embedding = client.embed_document(texts_to_embed[text_idx], use_cache=use_cache)
                        embeddings_list[text_idx] = embedding
                        generated_count += 1
                        retry_time = time.time() - retry_start_time
                        total_item_time = time.time() - item_start_time
                        
                        print(f"  ✓ Embedding succeeded after restart retry: {retry_time:.1f}s (total time: {total_item_time:.1f}s)")
                        logger.info(
                            f"  ✓ Embedding succeeded after restart retry: {retry_time:.1f}s "
                            f"(total time: {total_item_time:.1f}s)"
                        )
                        sys.stdout.flush()
                        
                        # Save checkpoint after successful retry
                        if cache_dir and use_cache:
                            if text_idx not in successfully_completed_indices:
                                successfully_completed_indices.append(text_idx)
                            save_checkpoint(
                                cache_dir,
                                [corpus.citation_keys[i] for i in valid_indices],
                                sorted(successfully_completed_indices),
                                [],
                                total_documents,
                                sorted(skipped_indices)
                            )
                            logger.debug(f"Checkpoint saved after retry success: {len(successfully_completed_indices)}/{total_documents} embeddings completed")
                        
                        if progress_bar:
                            progress_bar.update(progress_idx, success=True, item_time=total_item_time)
                        
                        # Continue to next document (skip the fallback code below)
                        continue
                        
                    except Exception as retry_error:
                        retry_error_type = type(retry_error).__name__
                        print(f"  ✗ Retry after restart also failed for {citation_key} ({retry_error_type}): {retry_error}")
                        logger.error(
                            f"Retry after restart also failed for {citation_key} "
                            f"({retry_error_type}): {retry_error}"
                        )
                        sys.stdout.flush()
                        # Fall through to fallback handling below
                
                # Save checkpoint even on failure (to track progress)
                if cache_dir and use_cache:
                    # Use the tracking list of successfully completed indices
                    failed_indices = [text_idx]
                    save_checkpoint(
                        cache_dir,
                        [corpus.citation_keys[i] for i in valid_indices],
                        sorted(successfully_completed_indices),
                        failed_indices,
                        total_documents,
                        sorted(skipped_indices)
                    )
                    logger.debug(f"Checkpoint saved after failure: {len(successfully_completed_indices)}/{total_documents} embeddings completed, {len(failed_indices)} failed, {len(skipped_indices)} skipped")
                
                # Use zero vector as fallback
                if generated_count > 0:
                    # Use dimension from a successfully generated embedding
                    for existing_emb in embeddings_list:
                        if existing_emb is not None:
                            fallback = np.zeros_like(existing_emb)
                            break
                    else:
                        fallback = np.zeros(config.embedding_dimension)
                else:
                    fallback = np.zeros(config.embedding_dimension)
                embeddings_list[text_idx] = fallback
                
                if progress_bar:
                    progress_bar.update(progress_idx, success=False, item_time=item_time)
        
        generation_time = time.time() - generation_start
        
        # Log summary of skipped documents
        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} document(s) exceeding {config.embedding_max_text_length:,} character limit"
            )
            if len(skipped_citation_keys) <= 10:
                logger.info(f"  Skipped papers: {', '.join(skipped_citation_keys)}")
            else:
                logger.info(f"  Skipped papers: {', '.join(skipped_citation_keys[:10])} and {len(skipped_citation_keys) - 10} more")
        
        # Log summary of failures
        if failed_count > 0:
            logger.warning(
                f"Failed to generate {failed_count} embedding(s) out of {n_to_generate} "
                f"(using zero vectors as fallback). Failed papers: {', '.join(failed_citation_keys[:5])}"
                + (f" and {len(failed_citation_keys) - 5} more" if len(failed_citation_keys) > 5 else "")
            )
        
        if progress_bar:
            progress_bar.finish()
        
        # Delete checkpoint on successful completion
        if cache_dir and use_cache:
            delete_checkpoint(cache_dir)
            logger.debug("Checkpoint deleted - generation complete")
        
        avg_gen_time = generation_time / generated_count if generated_count > 0 else 0
        
        # Log timing statistics
        if embedding_times:
            min_time = min(embedding_times)
            max_time = max(embedding_times)
            avg_time = sum(embedding_times) / len(embedding_times)
            logger.info(
                f"Phase 3/3 complete: Generated {generated_count} new embeddings "
                f"({format_duration(generation_time)}, avg: {avg_gen_time:.2f}s/embedding)"
            )
            logger.info(
                f"Timing statistics: min: {min_time:.1f}s, max: {max_time:.1f}s, "
                f"avg: {avg_time:.1f}s, close to timeout (>80%): {close_to_timeout_count}/{generated_count}"
            )
        else:
            logger.info(
                f"Phase 3/3 complete: Generated {generated_count} new embeddings "
                f"({format_duration(generation_time)})"
            )
    else:
        logger.info("Phase 3/3: All embeddings loaded from cache, no generation needed")
        # Delete checkpoint if all done
        if cache_dir and use_cache:
            delete_checkpoint(cache_dir)
    
    # Convert to array (all should be filled now)
    embeddings_array = np.array([emb if emb is not None else np.zeros(config.embedding_dimension) 
                                 for emb in embeddings_list])
    
    # Filter corpus data to match valid indices
    filtered_citation_keys = [corpus.citation_keys[i] for i in valid_indices]
    filtered_titles = [corpus.titles[i] if i < len(corpus.titles) else "" for i in valid_indices]
    filtered_years = [corpus.years[i] if i < len(corpus.years) else None for i in valid_indices]
    
    # Final summary with timing information
    total_time = time.time() - overall_start_time
    
    # Count zero vectors for summary
    norms = np.linalg.norm(embeddings_array, axis=1)
    zero_vector_count = int(np.sum(norms < 1e-10))
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total documents: {embeddings_array.shape[0]}")
    logger.info(f"Embedding dimension: {embeddings_array.shape[1]}")
    logger.info("")
    logger.info("Embedding sources:")
    logger.info(f"  • From cache: {cached_count}")
    if missing_indices:
        logger.info(f"  • Newly generated: {generated_count}")
        if skipped_count > 0:
            logger.info(f"  • Skipped (exceeded {config.embedding_max_text_length:,} char limit): {skipped_count}")
        if failed_count > 0:
            logger.info(f"  • Failed (using zero vectors): {failed_count}")
    else:
        logger.info("  • Newly generated: 0")
    logger.info("")
    logger.info("Zero vectors:")
    logger.info(f"  • Total zero vectors: {zero_vector_count}")
    if zero_vector_count > 0:
        logger.info(f"    - Skipped documents: {skipped_count if missing_indices else 0}")
        logger.info(f"    - Failed embeddings: {failed_count if missing_indices else 0}")
        logger.info("    Note: Zero vectors are expected for skipped/failed documents and are valid")
    logger.info("")
    logger.info("Timing breakdown:")
    if cache_check_time is not None:
        logger.info(f"  Cache check: {format_duration(cache_check_time)}")
    if cache_load_time is not None:
        logger.info(f"  Cache load: {format_duration(cache_load_time)}")
    if generation_time is not None:
        logger.info(f"  Generation: {format_duration(generation_time)}")
    logger.info(f"  Total time: {format_duration(total_time)}")
    logger.info("=" * 60)
    logger.info("")
    
    return EmbeddingData(
        citation_keys=filtered_citation_keys,
        embeddings=embeddings_array,
        titles=filtered_titles,
        years=filtered_years,
        embedding_dimension=embeddings_array.shape[1]
    )

