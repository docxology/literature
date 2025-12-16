"""Embedding-based analysis for literature meta-analysis.

Uses semantic embeddings (via Ollama embeddinggemma) to analyze paper
similarities, perform clustering, and enable semantic search.

This module provides a modular structure for embedding operations:

**Structure:**
- `data.py` - Data structures (EmbeddingData, SimilarityResults)
- `checkpoint.py` - Checkpoint management for resumable generation
- `generation.py` - Core embedding generation with three-phase process
- `computation.py` - Similarity, clustering, search, dimensionality reduction
- `export.py` - Export functions for various formats
- `shutdown.py` - Signal handling and graceful shutdown

**Usage:**
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
```
"""
from __future__ import annotations

# Data structures
from .data import EmbeddingData, SimilarityResults

# Core generation
from .generation import generate_document_embeddings, check_cached_embeddings

# Computation functions
from .computation import (
    compute_similarity_matrix,
    cluster_embeddings,
    find_similar_papers,
    reduce_dimensions,
)

# Export functions
from .export import (
    export_embeddings,
    export_similarity_matrix,
    export_clusters,
    export_embedding_statistics,
    export_validation_report,
    export_clustering_metrics,
)

# Checkpoint functions (internal utilities, exported for use by other modules)
from .checkpoint import (
    get_checkpoint_path,
    save_checkpoint,
    load_checkpoint,
    delete_checkpoint,
)

# Shutdown functions (internal utilities, exported for use by other modules)
from .shutdown import (
    setup_signal_handlers,
    save_checkpoint_on_shutdown,
    update_shutdown_state,
)

__all__ = [
    # Data structures
    "EmbeddingData",
    "SimilarityResults",
    # Core generation
    "generate_document_embeddings",
    "check_cached_embeddings",
    # Computation
    "compute_similarity_matrix",
    "cluster_embeddings",
    "find_similar_papers",
    "reduce_dimensions",
    # Export
    "export_embeddings",
    "export_similarity_matrix",
    "export_clusters",
    "export_embedding_statistics",
    "export_validation_report",
    "export_clustering_metrics",
    # Checkpoint (internal utilities, exported for use by other modules)
    "get_checkpoint_path",
    "save_checkpoint",
    "load_checkpoint",
    "delete_checkpoint",
    # Shutdown (internal utilities, exported for use by other modules)
    "setup_signal_handlers",
    "save_checkpoint_on_shutdown",
    "update_shutdown_state",
]

