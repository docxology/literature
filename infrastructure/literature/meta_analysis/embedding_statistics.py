"""Embedding statistics computation for meta-analysis.

Provides functions to compute comprehensive statistics on embeddings,
similarity matrices, clustering quality, and dimensionality analysis.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.llm.embeddings import EmbeddingData

logger = get_logger(__name__)

try:
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score
    )
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Some statistics will be limited.")


def compute_embedding_statistics(
    embedding_data: EmbeddingData
) -> Dict[str, Any]:
    """Compute comprehensive statistics on embeddings.
    
    Args:
        embedding_data: Embedding data to analyze.
        
    Returns:
        Dictionary with embedding statistics:
        - n_documents: Number of documents
        - embedding_dim: Embedding dimension
        - mean_norm: Mean L2 norm of embeddings
        - std_norm: Standard deviation of L2 norms
        - min_norm: Minimum L2 norm
        - max_norm: Maximum L2 norm
        - per_dimension: Dictionary with per-dimension statistics
    """
    embeddings = embedding_data.embeddings
    n_docs, n_dims = embeddings.shape
    
    # Compute norms
    norms = np.linalg.norm(embeddings, axis=1)
    
    # Per-dimension statistics
    per_dim = {
        "mean": embeddings.mean(axis=0).tolist(),
        "std": embeddings.std(axis=0).tolist(),
        "min": embeddings.min(axis=0).tolist(),
        "max": embeddings.max(axis=0).tolist(),
        "median": np.median(embeddings, axis=0).tolist()
    }
    
    results = {
        "n_documents": int(n_docs),
        "embedding_dim": int(n_dims),
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "median_norm": float(np.median(norms)),
        "per_dimension": per_dim
    }
    
    return results


def compute_similarity_statistics(
    similarity_matrix: np.ndarray,
    exclude_diagonal: bool = True
) -> Dict[str, Any]:
    """Compute statistics on similarity matrix.
    
    Args:
        similarity_matrix: Similarity matrix, shape (n_documents, n_documents).
        exclude_diagonal: Whether to exclude diagonal elements from statistics.
        
    Returns:
        Dictionary with similarity statistics:
        - mean: Mean similarity
        - median: Median similarity
        - std: Standard deviation
        - min: Minimum similarity
        - max: Maximum similarity
        - percentiles: Dictionary with percentile values
    """
    n = similarity_matrix.shape[0]
    
    if exclude_diagonal and n > 0:
        # Create mask to exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        values = similarity_matrix[mask]
    else:
        values = similarity_matrix.flatten()
    
    if len(values) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "percentiles": {}
        }
    
    percentiles = {
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95))
    }
    
    results = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": percentiles
    }
    
    return results


def compute_clustering_metrics(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray
) -> Dict[str, Any]:
    """Compute clustering quality metrics.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        cluster_labels: Cluster labels for each document.
        
    Returns:
        Dictionary with clustering metrics:
        - silhouette_score: Silhouette score (-1 to 1, higher is better)
        - davies_bouldin_index: Davies-Bouldin index (lower is better)
        - calinski_harabasz_score: Calinski-Harabasz score (higher is better)
        - inertia: K-means inertia (lower is better)
        - n_clusters: Number of clusters
        - cluster_sizes: Dictionary mapping cluster ID to size
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available for clustering metrics")
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_score": None,
            "inertia": None,
            "n_clusters": int(len(np.unique(cluster_labels))),
            "cluster_sizes": {}
        }
    
    n_clusters = len(np.unique(cluster_labels))
    n_samples = len(cluster_labels)
    
    results = {
        "n_clusters": int(n_clusters),
        "cluster_sizes": {}
    }
    
    # Compute cluster sizes
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        results["cluster_sizes"][int(label)] = int(count)
    
    # Compute metrics (require at least 2 clusters and 2 samples per cluster)
    if n_clusters < 2 or n_samples < 2:
        results["silhouette_score"] = None
        results["davies_bouldin_index"] = None
        results["calinski_harabasz_score"] = None
        results["inertia"] = None
        logger.warning("Insufficient data for clustering metrics")
        return results
    
    try:
        # Silhouette score (requires at least 2 clusters)
        if n_clusters >= 2:
            silhouette = silhouette_score(embeddings, cluster_labels)
            results["silhouette_score"] = float(silhouette)
        else:
            results["silhouette_score"] = None
    except Exception as e:
        logger.warning(f"Could not compute silhouette score: {e}")
        results["silhouette_score"] = None
    
    try:
        # Davies-Bouldin index
        if n_clusters >= 2:
            db_index = davies_bouldin_score(embeddings, cluster_labels)
            results["davies_bouldin_index"] = float(db_index)
        else:
            results["davies_bouldin_index"] = None
    except Exception as e:
        logger.warning(f"Could not compute Davies-Bouldin index: {e}")
        results["davies_bouldin_index"] = None
    
    try:
        # Calinski-Harabasz score
        if n_clusters >= 2:
            ch_score = calinski_harabasz_score(embeddings, cluster_labels)
            results["calinski_harabasz_score"] = float(ch_score)
        else:
            results["calinski_harabasz_score"] = None
    except Exception as e:
        logger.warning(f"Could not compute Calinski-Harabasz score: {e}")
        results["calinski_harabasz_score"] = None
    
    try:
        # Compute inertia (within-cluster sum of squares)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        results["inertia"] = float(kmeans.inertia_)
    except Exception as e:
        logger.warning(f"Could not compute inertia: {e}")
        results["inertia"] = None
    
    return results


def compute_dimensionality_analysis(
    embeddings: np.ndarray,
    n_components: Optional[int] = None
) -> Dict[str, Any]:
    """Analyze effective dimensionality of embeddings.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        n_components: Number of components for PCA (default: min(n_docs, n_dims)).
        
    Returns:
        Dictionary with dimensionality analysis:
        - n_documents: Number of documents
        - embedding_dim: Original embedding dimension
        - effective_dim: Effective dimension (number of components explaining 95% variance)
        - explained_variance: Explained variance ratio for each component
        - cumulative_variance: Cumulative explained variance
        - n_components_95: Number of components explaining 95% variance
        - n_components_99: Number of components explaining 99% variance
    """
    n_docs, n_dims = embeddings.shape
    
    if n_docs < 2:
        return {
            "n_documents": int(n_docs),
            "embedding_dim": int(n_dims),
            "effective_dim": int(n_dims),
            "explained_variance": [],
            "cumulative_variance": [],
            "n_components_95": int(n_dims),
            "n_components_99": int(n_dims)
        }
    
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available for dimensionality analysis")
        return {
            "n_documents": int(n_docs),
            "embedding_dim": int(n_dims),
            "effective_dim": int(n_dims),
            "explained_variance": [],
            "cumulative_variance": [],
            "n_components_95": int(n_dims),
            "n_components_99": int(n_dims)
        }
    
    # Determine number of components
    max_components = min(n_docs - 1, n_dims)
    if n_components is None:
        n_components = max_components
    else:
        n_components = min(n_components, max_components)
    
    try:
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)
        
        explained_var = pca.explained_variance_ratio_.tolist()
        cumulative_var = np.cumsum(explained_var).tolist()
        
        # Find number of components for 95% and 99% variance
        n_comp_95 = next((i + 1 for i, v in enumerate(cumulative_var) if v >= 0.95), n_components)
        n_comp_99 = next((i + 1 for i, v in enumerate(cumulative_var) if v >= 0.99), n_components)
        
        # Effective dimension is number of components explaining 95% variance
        effective_dim = n_comp_95
        
        results = {
            "n_documents": int(n_docs),
            "embedding_dim": int(n_dims),
            "effective_dim": int(effective_dim),
            "explained_variance": explained_var,
            "cumulative_variance": cumulative_var,
            "n_components_95": int(n_comp_95),
            "n_components_99": int(n_comp_99)
        }
        
        return results
    
    except Exception as e:
        logger.warning(f"Could not perform dimensionality analysis: {e}")
        return {
            "n_documents": int(n_docs),
            "embedding_dim": int(n_dims),
            "effective_dim": int(n_dims),
            "explained_variance": [],
            "cumulative_variance": [],
            "n_components_95": int(n_dims),
            "n_components_99": int(n_dims)
        }


def compute_outlier_statistics(
    embeddings: np.ndarray,
    outlier_indices: List[int],
    citation_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute statistics on detected outliers.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        outlier_indices: List of outlier indices.
        citation_keys: Optional citation keys for outliers.
        
    Returns:
        Dictionary with outlier statistics:
        - n_outliers: Number of outliers
        - outlier_percentage: Percentage of documents that are outliers
        - outlier_norms: Statistics on outlier norms
        - outlier_citation_keys: List of outlier citation keys (if provided)
    """
    n_docs = embeddings.shape[0]
    n_outliers = len(outlier_indices)
    
    results = {
        "n_outliers": int(n_outliers),
        "outlier_percentage": (n_outliers / n_docs * 100.0) if n_docs > 0 else 0.0,
        "outlier_norms": {},
        "outlier_citation_keys": []
    }
    
    if n_outliers > 0:
        outlier_embeddings = embeddings[outlier_indices]
        outlier_norms = np.linalg.norm(outlier_embeddings, axis=1)
        
        results["outlier_norms"] = {
            "mean": float(np.mean(outlier_norms)),
            "std": float(np.std(outlier_norms)),
            "min": float(np.min(outlier_norms)),
            "max": float(np.max(outlier_norms)),
            "median": float(np.median(outlier_norms))
        }
        
        if citation_keys:
            results["outlier_citation_keys"] = [citation_keys[i] for i in outlier_indices]
    
    return results


def compute_all_statistics(
    embedding_data: EmbeddingData,
    similarity_matrix: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute all statistics on embedding data.
    
    Args:
        embedding_data: Embedding data to analyze.
        similarity_matrix: Optional similarity matrix.
        cluster_labels: Optional cluster labels.
        
    Returns:
        Dictionary with all statistics:
        - embedding_stats: Embedding statistics
        - similarity_stats: Similarity statistics (if provided)
        - clustering_metrics: Clustering metrics (if provided)
        - dimensionality_analysis: Dimensionality analysis
    """
    results = {
        "embedding_stats": compute_embedding_statistics(embedding_data),
        "dimensionality_analysis": compute_dimensionality_analysis(embedding_data.embeddings)
    }
    
    if similarity_matrix is not None:
        results["similarity_stats"] = compute_similarity_statistics(similarity_matrix)
    
    if cluster_labels is not None:
        results["clustering_metrics"] = compute_clustering_metrics(
            embedding_data.embeddings,
            cluster_labels
        )
    
    return results

