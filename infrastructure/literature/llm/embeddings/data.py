"""Data structures for embedding analysis.

Contains dataclasses for embedding data and similarity results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class EmbeddingData:
    """Container for embedding data and metadata."""
    citation_keys: List[str]
    embeddings: np.ndarray  # Shape: (n_documents, embedding_dim)
    titles: List[str]
    years: List[Optional[int]]
    embedding_dimension: int


@dataclass
class SimilarityResults:
    """Container for similarity analysis results."""
    similarity_matrix: np.ndarray  # Shape: (n_documents, n_documents)
    citation_keys: List[str]
    titles: List[str]


