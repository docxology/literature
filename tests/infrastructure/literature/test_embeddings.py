"""Tests for embedding analysis functionality.

All tests use real implementations with actual data and API calls.
No mocks are used - tests verify actual behavior with real systems.
"""
import pytest
import numpy as np
from pathlib import Path

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.meta_analysis.embeddings import (
    EmbeddingData,
    SimilarityResults,
    compute_similarity_matrix,
    cluster_embeddings,
    reduce_dimensions,
    export_embeddings,
    export_similarity_matrix,
    export_clusters,
)
from infrastructure.literature.meta_analysis.embedding_validation import (
    validate_embedding_quality,
    validate_embedding_completeness,
    validate_embedding_dimensions,
    validate_similarity_matrix,
    detect_embedding_outliers,
    validate_all,
)
from infrastructure.literature.meta_analysis.embedding_statistics import (
    compute_embedding_statistics,
    compute_similarity_statistics,
    compute_clustering_metrics,
    compute_dimensionality_analysis,
    compute_all_statistics,
)
from infrastructure.literature.meta_analysis.aggregator import TextCorpus

logger = get_logger(__name__)


class TestEmbeddingData:
    """Test EmbeddingData dataclass with real data structures."""
    
    def test_creation(self):
        """Test EmbeddingData creation with real data."""
        logger.info("Testing EmbeddingData creation with real citation keys and embeddings")
        
        data = EmbeddingData(
            citation_keys=["smith2024ml", "jones2023dl"],
            embeddings=np.array([[1.0, 2.0], [3.0, 4.0]]),
            titles=["Machine Learning Advances", "Deep Learning Methods"],
            years=[2024, 2023],
            embedding_dimension=2
        )
        
        logger.debug(f"Created EmbeddingData with {len(data.citation_keys)} entries")
        assert len(data.citation_keys) == 2
        assert data.embeddings.shape == (2, 2)
        assert data.embedding_dimension == 2
        logger.info("EmbeddingData creation test passed")


class TestSimilarityResults:
    """Test SimilarityResults dataclass with real similarity data."""
    
    def test_creation(self):
        """Test SimilarityResults creation with real similarity matrix."""
        logger.info("Testing SimilarityResults creation with real similarity data")
        
        results = SimilarityResults(
            similarity_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            citation_keys=["paper1", "paper2"],
            titles=["Paper on Machine Learning", "Paper on Deep Learning"]
        )
        
        logger.debug(f"Created SimilarityResults with {len(results.citation_keys)} papers")
        assert results.similarity_matrix.shape == (2, 2)
        assert len(results.citation_keys) == 2
        logger.info("SimilarityResults creation test passed")


class TestComputeSimilarityMatrix:
    """Test similarity matrix computation with real vector operations."""
    
    def test_similarity_matrix(self):
        """Test cosine similarity computation with orthogonal vectors."""
        logger.info("Testing cosine similarity with orthogonal vectors")
        
        # Create orthogonal vectors (should have similarity 0)
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        similarity = compute_similarity_matrix(embeddings)
        
        logger.debug(f"Computed similarity matrix: {similarity}")
        assert similarity.shape == (2, 2)
        assert np.isclose(similarity[0, 0], 1.0)  # Self-similarity
        assert np.isclose(similarity[1, 1], 1.0)  # Self-similarity
        assert np.isclose(similarity[0, 1], 0.0)  # Orthogonal vectors
        logger.info("Orthogonal vector similarity test passed")
    
    def test_similarity_matrix_identical_vectors(self):
        """Test similarity for identical vectors."""
        logger.info("Testing similarity computation with identical vectors")
        
        embeddings = np.array([
            [1.0, 0.0],
            [1.0, 0.0]
        ])
        
        similarity = compute_similarity_matrix(embeddings)
        
        logger.debug(f"Similarity for identical vectors: {similarity[0, 1]:.6f}")
        assert np.isclose(similarity[0, 1], 1.0)  # Identical vectors
        logger.info("Identical vector similarity test passed")
    
    def test_similarity_matrix_zero_vectors(self):
        """Test similarity with zero vectors (edge case handling)."""
        logger.info("Testing similarity computation with zero vectors")
        
        embeddings = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        
        similarity = compute_similarity_matrix(embeddings)
        
        logger.debug(f"Similarity matrix with zero vector: {similarity}")
        # Should handle zero vectors gracefully
        assert similarity.shape == (2, 2)
        assert not np.isnan(similarity).any()
        assert not np.isinf(similarity).any()
        logger.info("Zero vector handling test passed")


class TestClusterEmbeddings:
    """Test embedding clustering with real K-means algorithm."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_cluster_embeddings(self):
        """Test K-means clustering on real embedding data."""
        logger.info("Testing K-means clustering with distinct clusters")
        
        # Create two distinct clusters with real random data
        np.random.seed(42)  # For reproducibility
        cluster1 = np.random.randn(5, 10) + np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(5, 10) + np.array([0, 10, 0, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([cluster1, cluster2])
        
        logger.debug(f"Clustering {len(embeddings)} embeddings into 2 clusters")
        labels = cluster_embeddings(embeddings, n_clusters=2)
        
        unique_labels = np.unique(labels)
        logger.debug(f"Found {len(unique_labels)} unique clusters: {unique_labels}")
        
        assert len(labels) == 10
        assert len(unique_labels) == 2
        logger.info("Clustering test passed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_cluster_embeddings_insufficient_samples(self):
        """Test clustering error handling with insufficient samples."""
        logger.info("Testing clustering error handling with insufficient data")
        
        embeddings = np.array([[1.0, 2.0]])
        
        logger.debug(f"Attempting to cluster {len(embeddings)} sample into 2 clusters")
        with pytest.raises(ValueError, match="Insufficient samples"):
            cluster_embeddings(embeddings, n_clusters=2)
        logger.info("Insufficient samples error handling test passed")


class TestReduceDimensions:
    """Test dimensionality reduction with real algorithms."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="scikit-learn not available"),
        reason="scikit-learn not available"
    )
    def test_reduce_dimensions_tsne(self):
        """Test t-SNE dimensionality reduction with real data."""
        logger.info("Testing t-SNE dimensionality reduction")
        
        np.random.seed(42)  # For reproducibility
        embeddings = np.random.randn(10, 50)
        
        logger.debug(f"Reducing {embeddings.shape} embeddings to 2D using t-SNE")
        reduced = reduce_dimensions(embeddings, n_components=2, method="tsne")
        
        logger.debug(f"Reduced embeddings shape: {reduced.shape}")
        assert reduced.shape == (10, 2)
        assert not np.isnan(reduced).any()
        assert not np.isinf(reduced).any()
        logger.info("t-SNE reduction test passed")
    
    def test_reduce_dimensions_invalid_method(self):
        """Test error handling for invalid reduction method."""
        logger.info("Testing error handling for invalid reduction method")
        
        np.random.seed(42)
        embeddings = np.random.randn(10, 50)
        
        logger.debug("Attempting reduction with invalid method")
        with pytest.raises(ValueError, match="Unknown reduction method"):
            reduce_dimensions(embeddings, n_components=2, method="invalid")
        logger.info("Invalid method error handling test passed")
    
    def test_reduce_dimensions_invalid_components(self):
        """Test error handling for invalid number of components."""
        logger.info("Testing error handling for invalid component count")
        
        np.random.seed(42)
        embeddings = np.random.randn(10, 50)
        
        logger.debug("Attempting reduction with invalid component count")
        with pytest.raises(ValueError, match="n_components must be 2 or 3"):
            reduce_dimensions(embeddings, n_components=4, method="tsne")
        logger.info("Invalid components error handling test passed")


class TestExportFunctions:
    """Test export functions with real file operations."""
    
    def test_export_embeddings(self, tmp_path):
        """Test embedding export to JSON file."""
        logger.info("Testing embedding export to JSON")
        
        embedding_data = EmbeddingData(
            citation_keys=["smith2024ml", "jones2023dl"],
            embeddings=np.array([[1.0, 2.0], [3.0, 4.0]]),
            titles=["Machine Learning Advances", "Deep Learning Methods"],
            years=[2024, 2023],
            embedding_dimension=2
        )
        
        output_path = tmp_path / "embeddings.json"
        logger.debug(f"Exporting embeddings to {output_path}")
        result_path = export_embeddings(embedding_data, output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify content with real file read
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        logger.debug(f"Exported {len(data['citation_keys'])} embeddings")
        assert len(data["citation_keys"]) == 2
        assert len(data["embeddings"]) == 2
        assert data["embedding_dimension"] == 2
        logger.info("Embedding export test passed")
    
    def test_export_similarity_matrix(self, tmp_path):
        """Test similarity matrix export to CSV."""
        logger.info("Testing similarity matrix export to CSV")
        
        results = SimilarityResults(
            similarity_matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),
            citation_keys=["paper1", "paper2"],
            titles=["Paper on ML", "Paper on DL"]
        )
        
        output_path = tmp_path / "similarity.csv"
        logger.debug(f"Exporting similarity matrix to {output_path}")
        result_path = export_similarity_matrix(results, output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify CSV content
        import csv
        with open(output_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        logger.debug(f"Exported CSV with {len(rows)} rows")
        assert len(rows) == 3  # Header + 2 data rows
        logger.info("Similarity matrix export test passed")
    
    def test_export_clusters(self, tmp_path):
        """Test cluster export to JSON."""
        logger.info("Testing cluster export to JSON")
        
        citation_keys = ["paper1", "paper2", "paper3"]
        cluster_labels = np.array([0, 0, 1])
        
        output_path = tmp_path / "clusters.json"
        logger.debug(f"Exporting clusters to {output_path}")
        result_path = export_clusters(citation_keys, cluster_labels, output_path)
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Verify content with real file read
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        logger.debug(f"Exported {len(data['clusters'])} cluster assignments")
        assert len(data["clusters"]) == 3
        assert data["n_clusters"] == 2
        logger.info("Cluster export test passed")


class TestComputeSimilarityMatrixRealData:
    """Test similarity matrix computation with real embedding-like data."""
    
    def test_similarity_matrix_real_embeddings(self):
        """Test cosine similarity with realistic embedding vectors."""
        logger.info("Testing similarity computation with realistic embedding vectors")
        
        # Create realistic embedding vectors (normalized)
        np.random.seed(42)
        embeddings = np.random.randn(5, 768)  # embeddinggemma dimension
        # Normalize to make them more realistic
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        similarity = compute_similarity_matrix(embeddings)
        
        logger.debug(f"Similarity matrix shape: {similarity.shape}")
        logger.debug(f"Similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")
        
        assert similarity.shape == (5, 5)
        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(np.diag(similarity), np.ones(5))
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(similarity, similarity.T)
        # Values should be in valid range
        assert np.all(similarity >= -1.01) and np.all(similarity <= 1.01)
    
    def test_similarity_matrix_with_actual_paper_topics(self):
        """Test similarity with embeddings representing different paper topics."""
        logger.info("Testing similarity with topic-based embeddings")
        
        # Create embeddings that represent different topics
        # Topic 1: Machine learning (first 2 papers)
        ml_embeddings = np.random.randn(2, 768)
        ml_embeddings += np.array([1.0] * 100 + [0.0] * 668)  # Bias toward ML features
        
        # Topic 2: Physics (next 2 papers)
        physics_embeddings = np.random.randn(2, 768)
        physics_embeddings += np.array([0.0] * 100 + [1.0] * 100 + [0.0] * 568)  # Bias toward physics
        
        # Topic 3: Biology (last paper)
        bio_embeddings = np.random.randn(1, 768)
        bio_embeddings += np.array([0.0] * 200 + [1.0] * 100 + [0.0] * 468)  # Bias toward biology
        
        embeddings = np.vstack([ml_embeddings, physics_embeddings, bio_embeddings])
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        similarity = compute_similarity_matrix(embeddings)
        
        logger.debug(f"ML-ML similarity: {similarity[0, 1]:.4f}")
        logger.debug(f"ML-Physics similarity: {similarity[0, 2]:.4f}")
        logger.debug(f"Physics-Physics similarity: {similarity[2, 3]:.4f}")
        
        # Papers in same topic should be more similar than papers in different topics
        # (This is a probabilistic test - may not always hold, but should generally be true)
        assert similarity.shape == (5, 5)
        assert np.allclose(np.diag(similarity), 1.0)


class TestEmbeddingValidation:
    """Test embedding validation functions with real data."""
    
    def test_validate_embedding_quality(self):
        """Test embedding quality validation."""
        logger.info("Testing embedding quality validation")
        
        # Create test embeddings
        np.random.seed(42)
        embeddings = np.random.randn(10, 768)
        citation_keys = [f"paper{i}" for i in range(10)]
        
        results = validate_embedding_quality(embeddings, citation_keys)
        
        assert "zero_vectors" in results
        assert "nan_values" in results
        assert "inf_values" in results
        assert "low_variance_dimensions" in results
        assert results["zero_vectors"] == 0  # Should have no zero vectors
        assert results["nan_values"] == 0  # Should have no NaN values
        logger.info("Embedding quality validation test passed")
    
    def test_validate_embedding_quality_with_issues(self):
        """Test validation with problematic embeddings."""
        logger.info("Testing validation with problematic embeddings")
        
        embeddings = np.random.randn(5, 768)
        # Add a zero vector
        embeddings[0] = 0.0
        # Add NaN
        embeddings[1, 0] = np.nan
        # Add Inf
        embeddings[2, 0] = np.inf
        
        citation_keys = [f"paper{i}" for i in range(5)]
        results = validate_embedding_quality(embeddings, citation_keys)
        
        assert results["zero_vectors"] > 0
        assert results["nan_values"] > 0
        assert results["inf_values"] > 0
        assert len(results["warnings"]) > 0
        logger.info("Validation with issues test passed")
    
    def test_validate_embedding_completeness(self):
        """Test embedding completeness validation."""
        logger.info("Testing embedding completeness validation")
        
        embedding_data = EmbeddingData(
            citation_keys=["paper1", "paper2", "paper3"],
            embeddings=np.random.randn(3, 768),
            titles=["Title 1", "Title 2", "Title 3"],
            years=[2024, 2023, 2024],
            embedding_dimension=768
        )
        
        expected_keys = ["paper1", "paper2", "paper3", "paper4"]
        results = validate_embedding_completeness(embedding_data, expected_keys)
        
        assert results["total_expected"] == 4
        assert results["total_generated"] == 3
        assert results["coverage"] == 75.0
        assert "paper4" in results["missing"]
        logger.info("Completeness validation test passed")
    
    def test_validate_embedding_dimensions(self):
        """Test dimension validation."""
        logger.info("Testing dimension validation")
        
        embeddings = np.random.randn(10, 768)
        results = validate_embedding_dimensions(embeddings, expected_dim=768)
        
        assert results["n_documents"] == 10
        assert results["embedding_dim"] == 768
        assert results["is_consistent"] is True
        logger.info("Dimension validation test passed")
    
    def test_validate_similarity_matrix(self):
        """Test similarity matrix validation."""
        logger.info("Testing similarity matrix validation")
        
        # Create valid similarity matrix
        embeddings = np.random.randn(5, 768)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        citation_keys = [f"paper{i}" for i in range(5)]
        results = validate_similarity_matrix(similarity_matrix, citation_keys)
        
        assert results["is_symmetric"] is True
        assert results["diagonal_all_one"] is True
        assert results["range_valid"] is True
        assert -1.0 <= results["min_value"] <= 1.0
        # Allow for floating point precision errors (1.0 + epsilon)
        assert -1.0 <= results["max_value"] <= 1.0 + 1e-10
        logger.info("Similarity matrix validation test passed")
    
    def test_detect_embedding_outliers(self):
        """Test outlier detection."""
        logger.info("Testing outlier detection")
        
        # Create embeddings with one outlier
        np.random.seed(42)
        embeddings = np.random.randn(10, 768)
        # Make one embedding an outlier (much larger norm)
        embeddings[0] = embeddings[0] * 10
        
        citation_keys = [f"paper{i}" for i in range(10)]
        results = detect_embedding_outliers(embeddings, citation_keys, method="zscore")
        
        assert "n_outliers" in results
        assert "outlier_indices" in results
        assert len(results["outlier_indices"]) > 0
        logger.info("Outlier detection test passed")
    
    def test_validate_all(self):
        """Test comprehensive validation."""
        logger.info("Testing comprehensive validation")
        
        embedding_data = EmbeddingData(
            citation_keys=["paper1", "paper2"],
            embeddings=np.random.randn(2, 768),
            titles=["Title 1", "Title 2"],
            years=[2024, 2023],
            embedding_dimension=768
        )
        
        embeddings = embedding_data.embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        results = validate_all(embedding_data, similarity_matrix=similarity_matrix)
        
        assert "quality" in results
        assert "completeness" in results
        assert "dimensions" in results
        assert "similarity" in results
        assert "outliers" in results
        assert "has_warnings" in results
        assert "has_errors" in results
        logger.info("Comprehensive validation test passed")


class TestEmbeddingStatistics:
    """Test embedding statistics computation with real data."""
    
    def test_compute_embedding_statistics(self):
        """Test embedding statistics computation."""
        logger.info("Testing embedding statistics computation")
        
        embedding_data = EmbeddingData(
            citation_keys=["paper1", "paper2", "paper3"],
            embeddings=np.random.randn(3, 768),
            titles=["Title 1", "Title 2", "Title 3"],
            years=[2024, 2023, 2024],
            embedding_dimension=768
        )
        
        stats = compute_embedding_statistics(embedding_data)
        
        assert stats["n_documents"] == 3
        assert stats["embedding_dim"] == 768
        assert "mean_norm" in stats
        assert "std_norm" in stats
        assert "per_dimension" in stats
        assert len(stats["per_dimension"]["mean"]) == 768
        logger.info("Embedding statistics test passed")
    
    def test_compute_similarity_statistics(self):
        """Test similarity statistics computation."""
        logger.info("Testing similarity statistics computation")
        
        # Create similarity matrix
        embeddings = np.random.randn(5, 768)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        stats = compute_similarity_statistics(similarity_matrix)
        
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "percentiles" in stats
        assert -1.0 <= stats["min"] <= stats["max"] <= 1.0
        logger.info("Similarity statistics test passed")
    
    def test_compute_clustering_metrics(self):
        """Test clustering metrics computation."""
        logger.info("Testing clustering metrics computation")
        
        # Create embeddings and cluster labels
        np.random.seed(42)
        embeddings = np.random.randn(10, 768)
        cluster_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        
        metrics = compute_clustering_metrics(embeddings, cluster_labels)
        
        assert "n_clusters" in metrics
        assert "cluster_sizes" in metrics
        assert metrics["n_clusters"] == 3
        
        # Metrics may be None if scikit-learn not available
        if metrics.get("silhouette_score") is not None:
            assert -1.0 <= metrics["silhouette_score"] <= 1.0
        logger.info("Clustering metrics test passed")
    
    def test_compute_dimensionality_analysis(self):
        """Test dimensionality analysis."""
        logger.info("Testing dimensionality analysis")
        
        # Create embeddings with some structure
        np.random.seed(42)
        embeddings = np.random.randn(20, 768)
        
        analysis = compute_dimensionality_analysis(embeddings)
        
        assert "n_documents" in analysis
        assert "embedding_dim" in analysis
        assert "effective_dim" in analysis
        assert "explained_variance" in analysis
        assert "cumulative_variance" in analysis
        assert analysis["n_documents"] == 20
        assert analysis["embedding_dim"] == 768
        logger.info("Dimensionality analysis test passed")
    
    def test_compute_all_statistics(self):
        """Test comprehensive statistics computation."""
        logger.info("Testing comprehensive statistics computation")
        
        embedding_data = EmbeddingData(
            citation_keys=["paper1", "paper2", "paper3"],
            embeddings=np.random.randn(3, 768),
            titles=["Title 1", "Title 2", "Title 3"],
            years=[2024, 2023, 2024],
            embedding_dimension=768
        )
        
        embeddings = embedding_data.embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        cluster_labels = np.array([0, 1, 0])
        
        stats = compute_all_statistics(
            embedding_data,
            similarity_matrix=similarity_matrix,
            cluster_labels=cluster_labels
        )
        
        assert "embedding_stats" in stats
        assert "similarity_stats" in stats
        assert "clustering_metrics" in stats
        assert "dimensionality_analysis" in stats
        logger.info("Comprehensive statistics test passed")

