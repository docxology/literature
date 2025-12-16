"""Computation functions for embedding analysis.

Provides similarity computation, clustering, semantic search, and
dimensionality reduction for embeddings.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from infrastructure.core.logging_utils import get_logger
from infrastructure.llm.core.embedding_client import EmbeddingClient
from infrastructure.llm.core.config import LLMConfig
from infrastructure.literature.core.config import LiteratureConfig
from .data import EmbeddingData

logger = get_logger(__name__)

# Check for optional dependencies
try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Clustering functionality will be limited.")


def compute_similarity_matrix(
    embeddings: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity matrix from embeddings.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        
    Returns:
        Similarity matrix, shape (n_documents, n_documents), values in [-1, 1].
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms
    
    # Compute cosine similarity (dot product of normalized vectors)
    similarity_matrix = np.dot(normalized, normalized.T)
    
    return similarity_matrix


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 5
) -> np.ndarray:
    """Cluster papers using K-means on embeddings.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        n_clusters: Number of clusters.
        
    Returns:
        Cluster labels for each paper, shape (n_documents,).
        
    Raises:
        ImportError: If scikit-learn is not available.
        ValueError: If insufficient samples for clustering.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for clustering")
    
    n_samples = embeddings.shape[0]
    
    if n_samples < n_clusters:
        raise ValueError(
            f"Insufficient samples for clustering: need at least {n_clusters}, got {n_samples}"
        )
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return labels


def find_similar_papers(
    embedding_data: EmbeddingData,
    query_text: str,
    top_k: int = 10,
    config: Optional[LiteratureConfig] = None,
    embedding_model: Optional[str] = None
) -> List[Tuple[str, float, str]]:
    """Find papers similar to a query text using semantic search.
    
    Args:
        embedding_data: Embedding data for all papers.
        query_text: Query text to search for.
        top_k: Number of top results to return.
        config: Literature configuration (uses default if not provided).
        embedding_model: Embedding model name (uses config default if not provided).
        
    Returns:
        List of tuples (citation_key, similarity_score, title), sorted by similarity.
    """
    if config is None:
        config = LiteratureConfig.from_env()
    
    if embedding_model is None:
        embedding_model = config.embedding_model
    
    # Generate embedding for query
    cache_dir = Path(config.embedding_cache_dir) if config.embedding_cache_dir else None
    llm_config = LLMConfig.from_env()
    
    client = EmbeddingClient(
        config=llm_config,
        embedding_model=embedding_model,
        cache_dir=cache_dir,
        chunk_size=config.embedding_chunk_size
    )
    
    query_embedding = client.embed_document(query_text, use_cache=True)
    
    # Compute similarities
    similarities = compute_similarity_matrix(
        np.vstack([query_embedding, embedding_data.embeddings])
    )
    
    # Get similarities to query (first row, excluding self-similarity)
    query_similarities = similarities[0, 1:]
    
    # Get top-k indices
    top_indices = np.argsort(query_similarities)[::-1][:top_k]
    
    # Build results
    results = []
    for idx in top_indices:
        citation_key = embedding_data.citation_keys[idx]
        similarity = float(query_similarities[idx])
        title = embedding_data.titles[idx] if idx < len(embedding_data.titles) else ""
        results.append((citation_key, similarity, title))
    
    return results


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    method: str = "umap"
) -> np.ndarray:
    """Reduce embedding dimensions for visualization.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        n_components: Target number of dimensions (2 or 3 for visualization).
        method: Reduction method ("umap" or "tsne").
        
    Returns:
        Reduced embeddings, shape (n_documents, n_components).
        
    Raises:
        ImportError: If required libraries are not available.
        ValueError: If method is invalid or n_components is not 2 or 3.
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for visualization")
    
    if method == "umap":
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available, falling back to t-SNE")
            method = "tsne"
        else:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(embeddings)
    
    if method == "tsne":
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for t-SNE")
        
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings) - 1))
        return reducer.fit_transform(embeddings)
    
    raise ValueError(f"Unknown reduction method: {method}")


