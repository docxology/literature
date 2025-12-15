"""Embedding validation functions for quality assurance.

Provides validation checks for embedding quality, completeness,
dimensionality, similarity matrices, and outlier detection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.meta_analysis.embeddings import EmbeddingData

logger = get_logger(__name__)


def validate_embedding_quality(
    embeddings: np.ndarray,
    citation_keys: Optional[List[str]] = None,
    variance_threshold: float = 1e-6
) -> Dict[str, Any]:
    """Validate embedding quality by checking for common issues.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        citation_keys: Optional citation keys for reporting problematic embeddings.
        variance_threshold: Minimum variance threshold for low-variance detection.
        
    Returns:
        Dictionary with quality validation results:
        - zero_vectors: Number of zero vectors found
        - nan_values: Number of NaN values found
        - inf_values: Number of Inf values found
        - low_variance_dimensions: List of dimension indices with low variance
        - problematic_embeddings: List of citation keys with issues (if provided)
        - warnings: List of warning messages
    """
    results = {
        "zero_vectors": 0,
        "nan_values": 0,
        "inf_values": 0,
        "low_variance_dimensions": [],
        "problematic_embeddings": [],
        "warnings": []
    }
    
    n_docs, n_dims = embeddings.shape
    
    # Check for zero vectors
    norms = np.linalg.norm(embeddings, axis=1)
    zero_mask = norms < 1e-10
    zero_count = np.sum(zero_mask)
    results["zero_vectors"] = int(zero_count)
    
    if zero_count > 0:
        results["warnings"].append(f"Found {zero_count} zero vectors (embeddings with near-zero norm)")
        if citation_keys:
            problematic = [citation_keys[i] for i in np.where(zero_mask)[0]]
            results["problematic_embeddings"].extend(problematic)
    
    # Check for NaN values
    nan_mask = np.isnan(embeddings)
    nan_count = np.sum(nan_mask)
    results["nan_values"] = int(nan_count)
    
    if nan_count > 0:
        results["warnings"].append(f"Found {nan_count} NaN values in embeddings")
        if citation_keys:
            nan_docs = np.any(nan_mask, axis=1)
            problematic = [citation_keys[i] for i in np.where(nan_docs)[0]]
            results["problematic_embeddings"].extend(problematic)
    
    # Check for Inf values
    inf_mask = np.isinf(embeddings)
    inf_count = np.sum(inf_mask)
    results["inf_values"] = int(inf_count)
    
    if inf_count > 0:
        results["warnings"].append(f"Found {inf_count} Inf values in embeddings")
        if citation_keys:
            inf_docs = np.any(inf_mask, axis=1)
            problematic = [citation_keys[i] for i in np.where(inf_docs)[0]]
            results["problematic_embeddings"].extend(problematic)
    
    # Check for low variance dimensions
    variances = np.var(embeddings, axis=0)
    low_var_mask = variances < variance_threshold
    low_var_dims = np.where(low_var_mask)[0].tolist()
    results["low_variance_dimensions"] = low_var_dims
    
    if len(low_var_dims) > 0:
        results["warnings"].append(
            f"Found {len(low_var_dims)} dimensions with variance < {variance_threshold}"
        )
    
    # Remove duplicates from problematic embeddings
    if results["problematic_embeddings"]:
        results["problematic_embeddings"] = list(set(results["problematic_embeddings"]))
    
    return results


def validate_embedding_completeness(
    embedding_data: EmbeddingData,
    expected_citation_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Validate embedding completeness and coverage.
    
    Args:
        embedding_data: Embedding data to validate.
        expected_citation_keys: Optional list of expected citation keys.
        
    Returns:
        Dictionary with completeness validation results:
        - total_expected: Total number of expected embeddings
        - total_generated: Total number of generated embeddings
        - coverage: Coverage percentage (0-100)
        - missing: List of missing citation keys (if expected provided)
    """
    results = {
        "total_expected": len(expected_citation_keys) if expected_citation_keys else len(embedding_data.citation_keys),
        "total_generated": len(embedding_data.citation_keys),
        "coverage": 0.0,
        "missing": []
    }
    
    if expected_citation_keys:
        generated_set = set(embedding_data.citation_keys)
        expected_set = set(expected_citation_keys)
        missing = expected_set - generated_set
        results["missing"] = sorted(list(missing))
        results["coverage"] = (len(generated_set) / len(expected_set)) * 100.0 if expected_set else 0.0
    else:
        results["coverage"] = 100.0
    
    return results


def validate_embedding_dimensions(
    embeddings: np.ndarray,
    expected_dim: Optional[int] = None
) -> Dict[str, Any]:
    """Validate embedding dimensionality consistency.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        expected_dim: Optional expected embedding dimension.
        
    Returns:
        Dictionary with dimension validation results:
        - n_documents: Number of documents
        - embedding_dim: Actual embedding dimension
        - expected_dim: Expected dimension (if provided)
        - is_consistent: Whether dimensions are consistent
        - warnings: List of warning messages
    """
    results = {
        "n_documents": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "expected_dim": expected_dim,
        "is_consistent": True,
        "warnings": []
    }
    
    # Check if all embeddings have same dimension
    if len(embeddings.shape) != 2:
        results["is_consistent"] = False
        results["warnings"].append(f"Invalid embedding shape: {embeddings.shape}")
        return results
    
    # Check against expected dimension
    if expected_dim is not None:
        if embeddings.shape[1] != expected_dim:
            results["is_consistent"] = False
            results["warnings"].append(
                f"Dimension mismatch: expected {expected_dim}, got {embeddings.shape[1]}"
            )
    
    # Check for empty embeddings
    if embeddings.shape[0] == 0:
        results["is_consistent"] = False
        results["warnings"].append("No embeddings found")
    
    if embeddings.shape[1] == 0:
        results["is_consistent"] = False
        results["warnings"].append("Embeddings have zero dimensions")
    
    return results


def validate_similarity_matrix(
    similarity_matrix: np.ndarray,
    citation_keys: Optional[List[str]] = None,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """Validate similarity matrix properties.
    
    Args:
        similarity_matrix: Similarity matrix, shape (n_documents, n_documents).
        citation_keys: Optional citation keys for reporting.
        tolerance: Numerical tolerance for equality checks.
        
    Returns:
        Dictionary with similarity matrix validation results:
        - is_symmetric: Whether matrix is symmetric
        - diagonal_all_one: Whether diagonal elements are all 1.0
        - range_valid: Whether values are in expected range [-1, 1]
        - min_value: Minimum value in matrix
        - max_value: Maximum value in matrix
        - mean_value: Mean value (excluding diagonal)
        - warnings: List of warning messages
    """
    results = {
        "is_symmetric": True,
        "diagonal_all_one": True,
        "range_valid": True,
        "min_value": float(np.min(similarity_matrix)),
        "max_value": float(np.max(similarity_matrix)),
        "mean_value": 0.0,
        "warnings": []
    }
    
    n = similarity_matrix.shape[0]
    
    # Check symmetry
    if n > 0:
        is_symmetric = np.allclose(similarity_matrix, similarity_matrix.T, atol=tolerance)
        results["is_symmetric"] = bool(is_symmetric)
        if not is_symmetric:
            max_diff = np.max(np.abs(similarity_matrix - similarity_matrix.T))
            results["warnings"].append(
                f"Similarity matrix is not symmetric (max difference: {max_diff:.2e})"
            )
        
        # Check diagonal
        diagonal = np.diag(similarity_matrix)
        diagonal_ones = np.allclose(diagonal, 1.0, atol=tolerance)
        results["diagonal_all_one"] = bool(diagonal_ones)
        if not diagonal_ones:
            min_diag = float(np.min(diagonal))
            max_diag = float(np.max(diagonal))
            results["warnings"].append(
                f"Diagonal elements not all 1.0 (range: [{min_diag:.6f}, {max_diag:.6f}])"
            )
        
        # Check range
        min_val = results["min_value"]
        max_val = results["max_value"]
        
        if min_val < -1.0 - tolerance or max_val > 1.0 + tolerance:
            results["range_valid"] = False
            results["warnings"].append(
                f"Values outside expected range [-1, 1]: [{min_val:.6f}, {max_val:.6f}]"
            )
        
        # Compute mean excluding diagonal
        mask = ~np.eye(n, dtype=bool)
        off_diagonal = similarity_matrix[mask]
        results["mean_value"] = float(np.mean(off_diagonal))
    else:
        results["warnings"].append("Empty similarity matrix")
    
    return results


def detect_embedding_outliers(
    embeddings: np.ndarray,
    citation_keys: Optional[List[str]] = None,
    method: str = "isolation_forest",
    contamination: float = 0.1
) -> Dict[str, Any]:
    """Detect outlier embeddings using statistical methods.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        citation_keys: Optional citation keys for reporting outliers.
        method: Outlier detection method ("isolation_forest", "zscore", "iqr").
        contamination: Expected proportion of outliers (for isolation forest).
        
    Returns:
        Dictionary with outlier detection results:
        - n_outliers: Number of outliers detected
        - outlier_indices: List of outlier indices
        - outlier_citation_keys: List of outlier citation keys (if provided)
        - outlier_scores: Outlier scores for each embedding
        - method_used: Method used for detection
    """
    results = {
        "n_outliers": 0,
        "outlier_indices": [],
        "outlier_citation_keys": [],
        "outlier_scores": [],
        "method_used": method
    }
    
    n_docs = embeddings.shape[0]
    
    if n_docs < 3:
        logger.warning("Too few documents for outlier detection (need at least 3)")
        return results
    
    if method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outlier_labels = iso_forest.fit_predict(embeddings)
            outlier_scores = iso_forest.score_samples(embeddings)
            
            # Outliers are labeled as -1
            outlier_mask = outlier_labels == -1
            outlier_indices = np.where(outlier_mask)[0].tolist()
            
            results["n_outliers"] = len(outlier_indices)
            results["outlier_indices"] = outlier_indices
            results["outlier_scores"] = outlier_scores.tolist()
            
            if citation_keys and outlier_indices:
                results["outlier_citation_keys"] = [citation_keys[i] for i in outlier_indices]
        
        except ImportError:
            logger.warning("scikit-learn not available, falling back to z-score method")
            method = "zscore"
    
    if method == "zscore":
        # Use Z-score method
        norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        if std_norm > 0:
            z_scores = np.abs((norms - mean_norm) / std_norm)
            outlier_mask = z_scores > 2.5  # 2.5 standard deviations
            outlier_indices = np.where(outlier_mask)[0].tolist()
            
            results["n_outliers"] = len(outlier_indices)
            results["outlier_indices"] = outlier_indices
            results["outlier_scores"] = z_scores.tolist()
            
            if citation_keys and outlier_indices:
                results["outlier_citation_keys"] = [citation_keys[i] for i in outlier_indices]
    
    elif method == "iqr":
        # Use IQR method
        norms = np.linalg.norm(embeddings, axis=1)
        q1 = np.percentile(norms, 25)
        q3 = np.percentile(norms, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (norms < lower_bound) | (norms > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        
        results["n_outliers"] = len(outlier_indices)
        results["outlier_indices"] = outlier_indices
        results["outlier_scores"] = norms.tolist()
        
        if citation_keys and outlier_indices:
            results["outlier_citation_keys"] = [citation_keys[i] for i in outlier_indices]
    
    return results


def validate_all(
    embedding_data: EmbeddingData,
    similarity_matrix: Optional[np.ndarray] = None,
    expected_citation_keys: Optional[List[str]] = None,
    expected_dim: Optional[int] = None
) -> Dict[str, Any]:
    """Run all validation checks on embedding data.
    
    Args:
        embedding_data: Embedding data to validate.
        similarity_matrix: Optional similarity matrix to validate.
        expected_citation_keys: Optional expected citation keys.
        expected_dim: Optional expected embedding dimension.
        
    Returns:
        Dictionary with all validation results:
        - quality: Quality validation results
        - completeness: Completeness validation results
        - dimensions: Dimension validation results
        - similarity: Similarity matrix validation results (if provided)
        - outliers: Outlier detection results
        - has_warnings: Whether any warnings were generated
        - has_errors: Whether any errors were found
    """
    results = {
        "quality": validate_embedding_quality(
            embedding_data.embeddings,
            citation_keys=embedding_data.citation_keys
        ),
        "completeness": validate_embedding_completeness(
            embedding_data,
            expected_citation_keys=expected_citation_keys
        ),
        "dimensions": validate_embedding_dimensions(
            embedding_data.embeddings,
            expected_dim=expected_dim
        ),
        "outliers": detect_embedding_outliers(
            embedding_data.embeddings,
            citation_keys=embedding_data.citation_keys
        )
    }
    
    if similarity_matrix is not None:
        results["similarity"] = validate_similarity_matrix(
            similarity_matrix,
            citation_keys=embedding_data.citation_keys
        )
    
    # Aggregate warnings and errors
    all_warnings = []
    for section in ["quality", "completeness", "dimensions", "outliers"]:
        if section in results and "warnings" in results[section]:
            all_warnings.extend(results[section]["warnings"])
    if "similarity" in results and "warnings" in results["similarity"]:
        all_warnings.extend(results["similarity"]["warnings"])
    
    results["has_warnings"] = len(all_warnings) > 0
    results["all_warnings"] = all_warnings
    
    # Check for critical errors
    has_errors = (
        results["quality"]["zero_vectors"] > 0 or
        results["quality"]["nan_values"] > 0 or
        results["quality"]["inf_values"] > 0 or
        not results["dimensions"]["is_consistent"]
    )
    results["has_errors"] = has_errors
    
    return results

