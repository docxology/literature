"""Integration tests for embedding analysis with Ollama.

These tests use Ollama embedding API to generate embeddings, compute similarities,
perform clustering, and enable semantic search. Requires Ollama to be running with
embeddinggemma model installed.

All tests use implementations - no mocks. Tests verify behavior with
API calls, data processing, and logging of all operations.
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

from infrastructure.llm.core.embedding_client import EmbeddingClient
from infrastructure.llm.core.config import LLMConfig
from infrastructure.literature.meta_analysis.embeddings import (
    generate_document_embeddings,
    compute_similarity_matrix,
    cluster_embeddings,
    find_similar_papers,
    reduce_dimensions,
    EmbeddingData,
)
from infrastructure.literature.meta_analysis.aggregator import TextCorpus, DataAggregator
from infrastructure.literature.core.config import LiteratureConfig


@pytest.fixture
def ensure_ollama_available():
    """Ensure Ollama is available for embeddings, skip test if not.
    
    Returns:
        LLMConfig instance if Ollama is available.
    """
    from infrastructure.llm.utils.ollama import is_ollama_running
    from tests.test_config_loader import get_test_llm_config
    
    if not is_ollama_running():
        pytest.skip("Ollama not running - start with 'ollama serve'")
    
    # Use test configuration
    config = get_test_llm_config()
    
    # Check if embedding model is available
    try:
        import requests
        response = requests.get(f"{config.base_url}/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            if not any("embeddinggemma" in name for name in model_names):
                pytest.skip("embeddinggemma model not installed - run 'ollama pull embeddinggemma'")
    except Exception:
        pytest.skip("Cannot check Ollama models")
    
    return config


@pytest.fixture
def embedding_client(ensure_ollama_available, tmp_path):
    """Create embedding client for testing."""
    cache_dir = tmp_path / "embeddings_cache"
    return EmbeddingClient(
        config=ensure_ollama_available,
        embedding_model="embeddinggemma",
        cache_dir=cache_dir,
        chunk_size=2000,
        batch_size=5
    )


@pytest.fixture
def sample_texts():
    """Create sample texts for embedding testing."""
    return [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "Deep learning uses neural networks with multiple layers to learn representations of data.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and understand visual information from images.",
        "Reinforcement learning is a type of machine learning where agents learn through trial and error.",
    ]


@pytest.fixture
def sample_corpus():
    """Create sample text corpus for embedding testing."""
    return TextCorpus(
        citation_keys=["paper1", "paper2", "paper3", "paper4", "paper5"],
        texts=[
            "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. It enables systems to learn from data without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to learn representations of data. These networks can automatically discover patterns in complex datasets.",
            "Natural language processing enables computers to understand and generate human language. It combines computational linguistics with machine learning techniques.",
            "Computer vision allows machines to interpret and understand visual information from images. It uses deep learning models to recognize objects and scenes.",
            "Reinforcement learning is a type of machine learning where agents learn through trial and error. Agents receive rewards for good actions and penalties for bad ones.",
        ],
        titles=[
            "Introduction to Machine Learning",
            "Deep Learning Fundamentals",
            "Natural Language Processing",
            "Computer Vision Applications",
            "Reinforcement Learning Methods",
        ],
        abstracts=[
            "This paper introduces machine learning concepts and applications.",
            "This paper covers deep learning architectures and training methods.",
            "This paper discusses NLP techniques and language models.",
            "This paper presents computer vision algorithms and applications.",
            "This paper explores reinforcement learning algorithms and applications.",
        ],
        years=[2020, 2021, 2022, 2023, 2024]
    )


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestRealEmbeddingClient:
    """Embedding client tests with Ollama."""
    
    @pytest.mark.timeout(60)
    def test_real_generate_embedding(self, embedding_client):
        """Test generating a single embedding with Ollama."""
        text = "Machine learning is fascinating."
        
        embedding = embedding_client.generate_embedding(text, use_cache=False)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
        assert embedding.dtype == np.float64 or embedding.dtype == np.float32
        # embeddinggemma default dimension is 768
        assert len(embedding) == 768
    
    @pytest.mark.timeout(120)
    def test_real_generate_embeddings_batch(self, embedding_client, sample_texts):
        """Test generating embeddings in batch with Ollama."""
        embeddings = embedding_client.generate_embeddings_batch(
            sample_texts,
            use_cache=False,
            show_progress=False
        )
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_texts)
        assert embeddings.shape[1] == 768  # embeddinggemma dimension
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
    
    @pytest.mark.timeout(60)
    def test_real_chunk_text(self, embedding_client):
        """Test text chunking functionality."""
        # Create a long text
        long_text = " ".join(["This is a sentence."] * 1000)
        
        chunks = embedding_client.chunk_text(long_text, max_tokens=2000)
        
        assert len(chunks) > 1
        assert all(len(chunk) > 0 for chunk in chunks)
        # Check that chunks don't exceed approximate token limit
        # (rough estimate: 1 token ≈ 4 characters)
        for chunk in chunks:
            assert len(chunk) <= 2000 * 4 * 1.2  # Allow 20% margin
    
    @pytest.mark.timeout(180)
    def test_real_embed_document(self, embedding_client):
        """Test embedding a full document with chunking."""
        # Create a document that exceeds token limit
        long_document = " ".join([
            "Machine learning is a powerful tool for data analysis.",
            "It uses algorithms to find patterns in data.",
            "Deep learning is a subset of machine learning.",
            "Neural networks are the foundation of deep learning.",
        ] * 200)  # Make it long enough to require chunking
        
        embedding = embedding_client.embed_document(long_document, use_cache=False)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 768
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
    
    @pytest.mark.timeout(60)
    def test_real_embedding_caching(self, embedding_client, tmp_path):
        """Test embedding caching functionality."""
        text = "This is a test text for caching."
        
        # Generate embedding (should cache)
        embedding1 = embedding_client.generate_embedding(text, use_cache=True)
        
        # Generate again (should use cache)
        embedding2 = embedding_client.generate_embedding(text, use_cache=True)
        
        # Should be identical
        np.testing.assert_array_almost_equal(embedding1, embedding2)
        
        # Cache file should exist
        cache_files = list(embedding_client.cache_dir.glob("*.json"))
        assert len(cache_files) > 0


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestRealEmbeddingAnalysis:
    """Embedding analysis tests with Ollama."""
    
    @pytest.mark.timeout(300)
    def test_real_generate_document_embeddings(self, sample_corpus, tmp_path):
        """Test generating embeddings for all documents in corpus."""
        config = LiteratureConfig()
        config.embedding_cache_dir = str(tmp_path / "embeddings")
        
        embedding_data = generate_document_embeddings(
            sample_corpus,
            config=config,
            use_cache=False,
            show_progress=False
        )
        
        assert isinstance(embedding_data, EmbeddingData)
        assert len(embedding_data.citation_keys) == len(sample_corpus.citation_keys)
        assert embedding_data.embeddings.shape[0] == len(sample_corpus.citation_keys)
        assert embedding_data.embeddings.shape[1] == 768
        assert not np.isnan(embedding_data.embeddings).any()
    
    @pytest.mark.timeout(60)
    def test_real_compute_similarity_matrix(self, embedding_client, sample_texts):
        """Test computing similarity matrix from real embeddings."""
        # Generate embeddings
        embeddings = embedding_client.generate_embeddings_batch(
            sample_texts,
            use_cache=False,
            show_progress=False
        )
        
        # Compute similarity
        similarity_matrix = compute_similarity_matrix(embeddings)
        
        assert similarity_matrix.shape == (len(sample_texts), len(sample_texts))
        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(
            np.diag(similarity_matrix),
            np.ones(len(sample_texts))
        )
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            similarity_matrix,
            similarity_matrix.T
        )
        # Values should be in [-1, 1] range (allow small floating point errors)
        assert np.all(similarity_matrix >= -1.01)  # Allow small negative overflow
        assert np.all(similarity_matrix <= 1.01)  # Allow small positive overflow
        # But most values should be in proper range
        assert np.max(similarity_matrix) <= 1.0 + 1e-6
        assert np.min(similarity_matrix) >= -1.0 - 1e-6
    
    @pytest.mark.timeout(120)
    def test_real_cluster_embeddings(self, embedding_client, sample_texts):
        """Test clustering real embeddings."""
        # Generate embeddings
        embeddings = embedding_client.generate_embeddings_batch(
            sample_texts,
            use_cache=False,
            show_progress=False
        )
        
        # Cluster
        n_clusters = 3
        cluster_labels = cluster_embeddings(embeddings, n_clusters=n_clusters)
        
        assert len(cluster_labels) == len(sample_texts)
        assert len(np.unique(cluster_labels)) == n_clusters
        assert all(0 <= label < n_clusters for label in cluster_labels)
    
    @pytest.mark.timeout(180)
    def test_real_find_similar_papers(self, sample_corpus, tmp_path):
        """Test semantic search with real embeddings."""
        config = LiteratureConfig()
        config.embedding_cache_dir = str(tmp_path / "embeddings")
        
        # Generate embeddings
        embedding_data = generate_document_embeddings(
            sample_corpus,
            config=config,
            use_cache=False,
            show_progress=False
        )
        
        # Search for similar papers
        query = "artificial intelligence and neural networks"
        results = find_similar_papers(
            embedding_data,
            query,
            top_k=3,
            config=config
        )
        
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 3 for r in results)
        # Results should be sorted by similarity (descending)
        similarities = [r[1] for r in results]
        assert similarities == sorted(similarities, reverse=True)
        # Similarities should be in valid range
        assert all(0.0 <= sim <= 1.0 for sim in similarities)
    
    @pytest.mark.timeout(120)
    def test_real_reduce_dimensions(self, embedding_client, sample_texts):
        """Test dimensionality reduction with real embeddings."""
        # Generate embeddings
        embeddings = embedding_client.generate_embeddings_batch(
            sample_texts,
            use_cache=False,
            show_progress=False
        )
        
        # Reduce to 2D
        embeddings_2d = reduce_dimensions(embeddings, n_components=2, method="tsne")
        
        assert embeddings_2d.shape == (len(sample_texts), 2)
        assert not np.isnan(embeddings_2d).any()
        assert not np.isinf(embeddings_2d).any()
        
        # Reduce to 3D
        embeddings_3d = reduce_dimensions(embeddings, n_components=3, method="tsne")
        
        assert embeddings_3d.shape == (len(sample_texts), 3)
        assert not np.isnan(embeddings_3d).any()
        assert not np.isinf(embeddings_3d).any()


@pytest.mark.integration
@pytest.mark.requires_ollama
class TestRealEmbeddingIntegration:
    """End-to-end embedding integration tests."""
    
    @pytest.mark.timeout(600)
    def test_real_full_pipeline(self, sample_corpus, tmp_path):
        """Test full embedding analysis pipeline."""
        config = LiteratureConfig()
        config.embedding_cache_dir = str(tmp_path / "embeddings")
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        
        # 1. Generate embeddings
        embedding_data = generate_document_embeddings(
            sample_corpus,
            config=config,
            use_cache=True,
            show_progress=False
        )
        
        # 2. Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(embedding_data.embeddings)
        
        # 3. Cluster
        cluster_labels = cluster_embeddings(embedding_data.embeddings, n_clusters=3)
        
        # 4. Semantic search
        results = find_similar_papers(
            embedding_data,
            "machine learning algorithms",
            top_k=2,
            config=config
        )
        
        # 5. Dimensionality reduction
        embeddings_2d = reduce_dimensions(
            embedding_data.embeddings,
            n_components=2,
            method="tsne"
        )
        
        # Verify all results
        assert embedding_data.embeddings.shape[0] == len(sample_corpus.citation_keys)
        assert similarity_matrix.shape[0] == len(sample_corpus.citation_keys)
        assert len(cluster_labels) == len(sample_corpus.citation_keys)
        assert len(results) == 2
        assert embeddings_2d.shape[0] == len(sample_corpus.citation_keys)
        
        # Verify similarity matrix properties
        assert np.allclose(np.diag(similarity_matrix), 1.0)
        assert np.allclose(similarity_matrix, similarity_matrix.T)
        
        # Verify clusters
        assert len(np.unique(cluster_labels)) == 3
        
        # Verify search results
        assert all(0.0 <= r[1] <= 1.0 for r in results)
    
    @pytest.mark.timeout(300)
    def test_real_embedding_with_extracted_texts(self, tmp_path, real_extracted_texts, real_library_entries):
        """Test embedding generation with real extracted text files."""
        config = LiteratureConfig()
        config.embedding_cache_dir = str(tmp_path / "embeddings")
        
        # Create aggregator with data
        aggregator = DataAggregator(config, default_entries=real_library_entries)
        
        # Prepare corpus from extracted texts
        corpus = aggregator.prepare_text_corpus(
            extracted_text_dir=real_extracted_texts
        )
        
        # Generate embeddings
        embedding_data = generate_document_embeddings(
            corpus,
            config=config,
            use_cache=True,
            show_progress=False
        )
        
        # Verify embeddings were generated
        assert len(embedding_data.citation_keys) > 0
        assert embedding_data.embeddings.shape[0] == len(embedding_data.citation_keys)
        assert embedding_data.embeddings.shape[1] == 768
        
        # Compute similarity
        similarity_matrix = compute_similarity_matrix(embedding_data.embeddings)
        
        # Verify similarity matrix
        assert similarity_matrix.shape[0] == len(embedding_data.citation_keys)
        assert similarity_matrix.shape[1] == len(embedding_data.citation_keys)

