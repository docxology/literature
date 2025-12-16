"""Signal handling and graceful shutdown for embedding generation.

Manages SIGINT/SIGTERM handlers to save checkpoint before exit.
"""
from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import List, Optional, Any

from infrastructure.core.logging_utils import get_logger
from infrastructure.llm.core.embedding_client import EmbeddingClient
from .checkpoint import save_checkpoint

logger = get_logger(__name__)

# Global state for graceful shutdown handling
_shutdown_state = {
    "cache_dir": None,
    "citation_keys": None,
    "completed_indices": [],
    "failed_indices": [],
    "skipped_indices": [],
    "total": 0,
    "client": None
}


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown (SIGINT, SIGTERM)."""
    def signal_handler(signum, frame):
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.warning(f"Received {signal_name} - saving checkpoint and exiting gracefully...")
        save_checkpoint_on_shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def save_checkpoint_on_shutdown() -> None:
    """Save checkpoint with current progress before shutdown."""
    state = _shutdown_state
    if state["cache_dir"] and state["citation_keys"] is not None:
        try:
            save_checkpoint(
                state["cache_dir"],
                state["citation_keys"],
                state["completed_indices"],
                state["failed_indices"],
                state["total"],
                state["skipped_indices"]
            )
            logger.info(f"Checkpoint saved: {len(state['completed_indices'])}/{state['total']} completed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint on shutdown: {e}")


def update_shutdown_state(
    cache_dir: Optional[Path] = None,
    citation_keys: Optional[List[str]] = None,
    completed_indices: Optional[List[int]] = None,
    failed_indices: Optional[List[int]] = None,
    skipped_indices: Optional[List[int]] = None,
    total: Optional[int] = None,
    client: Optional[EmbeddingClient] = None
) -> None:
    """Update global shutdown state for checkpoint saving.
    
    Allows partial updates - only provided parameters are updated.
    This enables updating state incrementally without requiring all parameters.
    
    Args:
        cache_dir: Cache directory path. If None, existing value is preserved.
        citation_keys: List of citation keys being processed. If None, existing value is preserved.
        completed_indices: List of completed indices. If None, existing value is preserved.
        failed_indices: List of failed indices. If None, existing value is preserved.
        skipped_indices: List of skipped indices. If None, existing value is preserved.
        total: Total number of embeddings. If None, existing value is preserved.
        client: EmbeddingClient instance. If None, existing value is preserved.
        
    Raises:
        TypeError: If indices are not lists of integers.
        ValueError: If cache_dir is provided but not a valid Path.
    """
    # Validate and update cache_dir
    if cache_dir is not None:
        if not isinstance(cache_dir, Path):
            raise TypeError(f"cache_dir must be a Path, got {type(cache_dir).__name__}")
        _shutdown_state["cache_dir"] = cache_dir
    
    # Validate and update citation_keys
    if citation_keys is not None:
        if not isinstance(citation_keys, list) or not all(isinstance(k, str) for k in citation_keys):
            raise TypeError("citation_keys must be a list of strings")
        _shutdown_state["citation_keys"] = citation_keys
    
    # Validate and update completed_indices
    if completed_indices is not None:
        if not isinstance(completed_indices, list) or not all(isinstance(i, int) for i in completed_indices):
            raise TypeError("completed_indices must be a list of integers")
        _shutdown_state["completed_indices"] = completed_indices
    
    # Validate and update failed_indices
    if failed_indices is not None:
        if not isinstance(failed_indices, list):
            # Handle case where failed_indices might be list of strings (citation keys)
            if all(isinstance(i, str) for i in failed_indices):
                # Convert citation keys to indices if needed (for backward compatibility)
                if _shutdown_state["citation_keys"]:
                    try:
                        failed_indices = [_shutdown_state["citation_keys"].index(k) for k in failed_indices]
                    except ValueError:
                        pass  # If conversion fails, treat as empty list
            elif not all(isinstance(i, int) for i in failed_indices):
                raise TypeError("failed_indices must be a list of integers or strings")
        _shutdown_state["failed_indices"] = failed_indices
    
    # Validate and update skipped_indices
    if skipped_indices is not None:
        if not isinstance(skipped_indices, list) or not all(isinstance(i, int) for i in skipped_indices):
            raise TypeError("skipped_indices must be a list of integers")
        _shutdown_state["skipped_indices"] = skipped_indices
    
    # Validate and update total
    if total is not None:
        if not isinstance(total, int) or total < 0:
            raise TypeError("total must be a non-negative integer")
        _shutdown_state["total"] = total
    
    # Update client (no validation needed - type hint handles it)
    if client is not None:
        _shutdown_state["client"] = client


