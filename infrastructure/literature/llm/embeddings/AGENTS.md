# Embeddings Module - Documentation

## Purpose

Modular embedding generation and analysis system for semantic paper analysis. Provides functions for generating embeddings, computing similarities, clustering, and semantic search using Ollama's embedding models.

## Structure

The embeddings module is organized into focused submodules:

### `data.py` (~50 lines)
Data structures for embedding data and results.

**Classes:**
- `EmbeddingData` - Container for embeddings, citation keys, titles, years, embedding dimension
- `SimilarityResults` - Container for similarity matrix and metadata

### `checkpoint.py` (~200 lines)
Checkpoint management for resumable embedding generation.

**Functions:**
- `get_checkpoint_path()` - Get checkpoint file path
- `save_checkpoint()` - Save embedding generation progress
- `load_checkpoint()` - Load embedding generation progress
- `delete_checkpoint()` - Delete checkpoint file

### `generation.py` (~1200 lines)
Core embedding generation with three-phase process.

**Functions:**
- `generate_document_embeddings()` - Main function for generating embeddings
- `check_cached_embeddings()` - Check which embeddings are already cached

**Three-Phase Process:**
1. **Phase 1/3**: Check cache for existing embeddings
2. **Phase 2/3**: Load cached embeddings
3. **Phase 3/3**: Generate missing embeddings with checkpoint resume support

**Features:**
- Automatic text chunking for large documents
- Adaptive timeout scaling based on text length
- Hung Ollama detection and automatic recovery
- Checkpoint resume capability
- Progress tracking and logging
- Graceful shutdown handling

### `computation.py` (~300 lines)
Similarity computation, clustering, search, and dimensionality reduction.

**Functions:**
- `compute_similarity_matrix()` - Compute cosine similarity matrix from embeddings
- `cluster_embeddings()` - K-means clustering on embedding space
- `find_similar_papers()` - Semantic search to find papers similar to a query
- `reduce_dimensions()` - Dimensionality reduction (UMAP/t-SNE) for visualization

### `export.py` (~400 lines)
Export functions for various formats.

**Functions:**
- `export_embeddings()` - Export embeddings to JSON
- `export_similarity_matrix()` - Export similarity matrix to CSV
- `export_clusters()` - Export cluster assignments to JSON
- `export_embedding_statistics()` - Export comprehensive statistics (JSON/CSV)
- `export_validation_report()` - Export validation results to JSON
- `export_clustering_metrics()` - Export clustering quality metrics (JSON/CSV)

### `shutdown.py` (~100 lines)
Signal handling and graceful shutdown.

**Functions:**
- `setup_signal_handlers()` - Set up SIGINT/SIGTERM handlers
- `save_checkpoint_on_shutdown()` - Save checkpoint before exit
- `update_shutdown_state()` - Update global shutdown state

## Usage

### Basic Usage

```python
from infrastructure.literature.llm.embeddings import (
    generate_document_embeddings,
    compute_similarity_matrix,
    cluster_embeddings,
    find_similar_papers,
)

# Generate embeddings
embedding_data = generate_document_embeddings(corpus)

# Compute similarity
similarity_matrix = compute_similarity_matrix(embedding_data.embeddings)

# Cluster papers
cluster_labels = cluster_embeddings(embedding_data.embeddings, n_clusters=5)

# Semantic search
results = find_similar_papers(embedding_data, "machine learning", top_k=10)
```

### Via Meta-Analysis Module

```python
from infrastructure.literature.meta_analysis import (
    generate_document_embeddings,
    compute_similarity_matrix,
)
```

## Integration

The embeddings module is integrated into the meta-analysis workflow:
- Exported via `infrastructure.literature.meta_analysis` module
- Used by `embedding_validation.py` and `embedding_statistics.py`
- Called from `workflow/operations/meta_analysis.py`

## Dependencies

- `numpy` - Numerical operations
- `scikit-learn` - Clustering and t-SNE (optional)
- `umap-learn` - UMAP dimensionality reduction (optional)
- `infrastructure.llm.core.embedding_client` - Ollama embedding client
- `infrastructure.literature.meta_analysis.aggregator` - Text corpus preparation

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - LLM module documentation
- [`../../meta_analysis/AGENTS.md`](../../meta_analysis/AGENTS.md) - Meta-analysis documentation

