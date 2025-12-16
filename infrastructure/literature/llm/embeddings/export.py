"""Export functions for embedding data and analysis results.

Provides functions to export embeddings, similarity matrices, clusters,
statistics, validation reports, and clustering metrics to various formats.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from infrastructure.core.logging_utils import get_logger
from .data import EmbeddingData, SimilarityResults

logger = get_logger(__name__)


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




