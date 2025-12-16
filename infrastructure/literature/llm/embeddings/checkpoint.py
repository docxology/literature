"""Checkpoint management for embedding generation progress.

Handles saving, loading, and deleting checkpoint files to enable
resumable embedding generation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from infrastructure.core.logging_utils import get_logger

logger = get_logger(__name__)


def get_checkpoint_path(cache_dir: Optional[Path]) -> Optional[Path]:
    """Get checkpoint file path for embedding progress.
    
    Args:
        cache_dir: Cache directory path.
        
    Returns:
        Path to checkpoint file, or None if cache_dir is None.
    """
    if cache_dir is None:
        return None
    return cache_dir / ".embedding_progress.json"


def save_checkpoint(
    cache_dir: Optional[Path],
    citation_keys: List[str],
    completed_indices: List[int],
    failed_indices: List[int],
    total: int,
    skipped_indices: Optional[List[int]] = None
) -> None:
    """Save embedding generation progress to checkpoint file.
    
    Args:
        cache_dir: Cache directory path.
        citation_keys: List of all citation keys being processed.
        completed_indices: List of indices that have been successfully embedded.
        failed_indices: List of indices that failed.
        total: Total number of embeddings to generate.
        skipped_indices: Optional list of indices that were skipped (e.g., exceeded length limit).
    """
    checkpoint_path = get_checkpoint_path(cache_dir)
    if checkpoint_path is None:
        return
    
    try:
        checkpoint_data = {
            "citation_keys": citation_keys,
            "completed_indices": completed_indices,
            "failed_indices": failed_indices,
            "skipped_indices": skipped_indices or [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total": total,
            "completed": len(completed_indices)
        }
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.debug(f"Checkpoint saved: {len(completed_indices)}/{total} completed")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint(cache_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load embedding generation progress from checkpoint file.
    
    Args:
        cache_dir: Cache directory path.
        
    Returns:
        Checkpoint data dictionary, or None if no checkpoint exists or cannot be loaded.
    """
    checkpoint_path = get_checkpoint_path(cache_dir)
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        logger.info(f"Found checkpoint from {checkpoint_data.get('timestamp', 'unknown time')}")
        logger.info(f"  Completed: {checkpoint_data.get('completed', 0)}/{checkpoint_data.get('total', 0)}")
        return checkpoint_data
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


def delete_checkpoint(cache_dir: Optional[Path]) -> None:
    """Delete checkpoint file.
    
    Args:
        cache_dir: Cache directory path.
    """
    checkpoint_path = get_checkpoint_path(cache_dir)
    if checkpoint_path is None or not checkpoint_path.exists():
        return
    
    try:
        checkpoint_path.unlink()
        logger.debug("Checkpoint file deleted")
    except Exception as e:
        logger.warning(f"Failed to delete checkpoint: {e}")




