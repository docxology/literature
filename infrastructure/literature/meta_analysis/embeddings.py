"""Embedding-based analysis for literature meta-analysis.

Uses semantic embeddings (via Ollama embeddinggemma) to analyze paper
similarities, perform clustering, and enable semantic search.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np

from infrastructure.core.logging_utils import get_logger
from infrastructure.llm.core.embedding_client import EmbeddingClient
from infrastructure.llm.core.config import LLMConfig
from infrastructure.literature.meta_analysis.aggregator import (
    DataAggregator,
    TextCorpus,
)
from infrastructure.literature.core.config import LiteratureConfig

logger = get_logger(__name__)

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


def generate_document_embeddings(
    corpus: TextCorpus,
    config: Optional[LiteratureConfig] = None,
    embedding_model: Optional[str] = None,
    use_cache: bool = True,
    show_progress: bool = True
) -> EmbeddingData:
    """Generate embeddings for all documents in corpus.
    
    Args:
        corpus: Text corpus from aggregator.
        config: Literature configuration (uses default if not provided).
        embedding_model: Embedding model name (uses config default if not provided).
        use_cache: Whether to use cached embeddings if available.
        show_progress: Whether to log progress.
        
    Returns:
        EmbeddingData with embeddings for all documents.
        
    Raises:
        ValueError: If corpus is empty or has insufficient data.
    """
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
    
    logger.info(f"Generating embeddings for {len(texts_to_embed)} documents using {embedding_model}...")
    
    # Initialize embedding client
    cache_dir = Path(config.embedding_cache_dir) if config.embedding_cache_dir else None
    llm_config = LLMConfig.from_env()
    
    client = EmbeddingClient(
        config=llm_config,
        embedding_model=embedding_model,
        cache_dir=cache_dir,
        chunk_size=config.embedding_chunk_size,
        batch_size=config.embedding_batch_size
    )
    
    # Check connection
    if not client.check_connection():
        raise RuntimeError(
            "Cannot connect to Ollama. Is it running? Start with: ollama serve"
        )
    
    # Generate embeddings
    embeddings_list = []
    for i, text in enumerate(texts_to_embed, 1):
        if show_progress:
            logger.info(f"Embedding document {i}/{len(texts_to_embed)}: {corpus.citation_keys[valid_indices[i-1]]}")
        
        try:
            # Use embed_document for full text (handles chunking automatically)
            embedding = client.embed_document(text, use_cache=use_cache)
            embeddings_list.append(embedding)
        except Exception as e:
            logger.error(f"Failed to embed document {corpus.citation_keys[valid_indices[i-1]]}: {e}")
            # Use zero vector as fallback
            if embeddings_list:
                fallback = np.zeros_like(embeddings_list[0])
            else:
                fallback = np.zeros(config.embedding_dimension)
            embeddings_list.append(fallback)
    
    embeddings_array = np.array(embeddings_list)
    
    # Filter corpus data to match valid indices
    filtered_citation_keys = [corpus.citation_keys[i] for i in valid_indices]
    filtered_titles = [corpus.titles[i] if i < len(corpus.titles) else "" for i in valid_indices]
    filtered_years = [corpus.years[i] if i < len(corpus.years) else None for i in valid_indices]
    
    logger.info(f"Generated embeddings: {embeddings_array.shape[0]} documents, "
               f"dimension {embeddings_array.shape[1]}")
    
    return EmbeddingData(
        citation_keys=filtered_citation_keys,
        embeddings=embeddings_array,
        titles=filtered_titles,
        years=filtered_years,
        embedding_dimension=embeddings_array.shape[1]
    )


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


def export_embeddings(
    embedding_data: EmbeddingData,
    output_path: Path
) -> Path:
    """Export embeddings to JSON file.
    
    Args:
        embedding_data: Embedding data to export.
        output_path: Output file path.
        
    Returns:
        Path to exported file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "citation_keys": embedding_data.citation_keys,
        "embeddings": embedding_data.embeddings.tolist(),
        "titles": embedding_data.titles,
        "years": embedding_data.years,
        "embedding_dimension": embedding_data.embedding_dimension
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Exported embeddings to {output_path}")
    return output_path


def export_similarity_matrix(
    similarity_results: SimilarityResults,
    output_path: Path
) -> Path:
    """Export similarity matrix to CSV file.
    
    Args:
        similarity_results: Similarity results to export.
        output_path: Output file path.
        
    Returns:
        Path to exported file.
    """
    import csv
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header row
        header = [''] + similarity_results.citation_keys
        writer.writerow(header)
        
        # Data rows
        for i, citation_key in enumerate(similarity_results.citation_keys):
            row = [citation_key] + similarity_results.similarity_matrix[i].tolist()
            writer.writerow(row)
    
    logger.info(f"Exported similarity matrix to {output_path}")
    return output_path


def export_clusters(
    citation_keys: List[str],
    cluster_labels: np.ndarray,
    output_path: Path
) -> Path:
    """Export cluster assignments to JSON file.
    
    Args:
        citation_keys: Citation keys for each document.
        cluster_labels: Cluster labels for each document.
        output_path: Output file path.
        
    Returns:
        Path to exported file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "clusters": [
            {
                "citation_key": key,
                "cluster": int(label)
            }
            for key, label in zip(citation_keys, cluster_labels)
        ],
        "n_clusters": int(len(np.unique(cluster_labels)))
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Exported cluster assignments to {output_path}")
    return output_path


def export_embedding_statistics(
    statistics: Dict[str, Any],
    output_path: Path,
    format: str = "json"
) -> Path:
    """Export comprehensive embedding statistics to file.
    
    Args:
        statistics: Statistics dictionary from compute_all_statistics().
        output_path: Output file path.
        format: Export format ("json" or "csv").
        
    Returns:
        Path to exported file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        logger.info(f"Exported embedding statistics to {output_path}")
    
    elif format == "csv":
        import csv
        
        # Flatten statistics for CSV
        rows = []
        
        # Embedding stats
        if "embedding_stats" in statistics:
            emb_stats = statistics["embedding_stats"]
            rows.append(["Metric", "Value"])
            rows.append(["n_documents", emb_stats.get("n_documents", "")])
            rows.append(["embedding_dim", emb_stats.get("embedding_dim", "")])
            rows.append(["mean_norm", emb_stats.get("mean_norm", "")])
            rows.append(["std_norm", emb_stats.get("std_norm", "")])
            rows.append(["min_norm", emb_stats.get("min_norm", "")])
            rows.append(["max_norm", emb_stats.get("max_norm", "")])
            rows.append([])
        
        # Similarity stats
        if "similarity_stats" in statistics:
            sim_stats = statistics["similarity_stats"]
            rows.append(["Similarity Metric", "Value"])
            rows.append(["mean", sim_stats.get("mean", "")])
            rows.append(["median", sim_stats.get("median", "")])
            rows.append(["std", sim_stats.get("std", "")])
            rows.append(["min", sim_stats.get("min", "")])
            rows.append(["max", sim_stats.get("max", "")])
            rows.append([])
        
        # Clustering metrics
        if "clustering_metrics" in statistics:
            clust_metrics = statistics["clustering_metrics"]
            rows.append(["Clustering Metric", "Value"])
            if clust_metrics.get("silhouette_score") is not None:
                rows.append(["silhouette_score", clust_metrics["silhouette_score"]])
            if clust_metrics.get("davies_bouldin_index") is not None:
                rows.append(["davies_bouldin_index", clust_metrics["davies_bouldin_index"]])
            if clust_metrics.get("calinski_harabasz_score") is not None:
                rows.append(["calinski_harabasz_score", clust_metrics["calinski_harabasz_score"]])
            if clust_metrics.get("inertia") is not None:
                rows.append(["inertia", clust_metrics["inertia"]])
            rows.append(["n_clusters", clust_metrics.get("n_clusters", "")])
            rows.append([])
        
        # Dimensionality analysis
        if "dimensionality_analysis" in statistics:
            dim_analysis = statistics["dimensionality_analysis"]
            rows.append(["Dimensionality Metric", "Value"])
            rows.append(["effective_dim", dim_analysis.get("effective_dim", "")])
            rows.append(["n_components_95", dim_analysis.get("n_components_95", "")])
            rows.append(["n_components_99", dim_analysis.get("n_components_99", "")])
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        logger.info(f"Exported embedding statistics to {output_path}")
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return output_path


def export_validation_report(
    validation_results: Dict[str, Any],
    output_path: Path
) -> Path:
    """Export validation results to JSON file.
    
    Args:
        validation_results: Validation results dictionary from validate_all().
        output_path: Output file path.
        
    Returns:
        Path to exported file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(validation_results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Exported validation report to {output_path}")
    return output_path


def export_clustering_metrics(
    metrics: Dict[str, Any],
    output_path: Path,
    format: str = "json"
) -> Path:
    """Export clustering quality metrics to file.
    
    Args:
        metrics: Clustering metrics dictionary from compute_clustering_metrics().
        output_path: Output file path.
        format: Export format ("json" or "csv").
        
    Returns:
        Path to exported file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Convert numpy types to native Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_metrics = convert_to_serializable(metrics)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        logger.info(f"Exported clustering metrics to {output_path}")
    
    elif format == "csv":
        import csv
        
        rows = [["Metric", "Value"]]
        
        if metrics.get("silhouette_score") is not None:
            rows.append(["silhouette_score", metrics["silhouette_score"]])
        if metrics.get("davies_bouldin_index") is not None:
            rows.append(["davies_bouldin_index", metrics["davies_bouldin_index"]])
        if metrics.get("calinski_harabasz_score") is not None:
            rows.append(["calinski_harabasz_score", metrics["calinski_harabasz_score"]])
        if metrics.get("inertia") is not None:
            rows.append(["inertia", metrics["inertia"]])
        rows.append(["n_clusters", metrics.get("n_clusters", "")])
        
        # Cluster sizes
        if "cluster_sizes" in metrics:
            rows.append([])
            rows.append(["Cluster ID", "Size"])
            for cluster_id, size in sorted(metrics["cluster_sizes"].items()):
                rows.append([cluster_id, size])
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        logger.info(f"Exported clustering metrics to {output_path}")
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return output_path

