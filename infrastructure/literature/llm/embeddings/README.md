# Embeddings Module

## Purpose

Modular embedding generation and analysis system for semantic paper analysis. Provides functions for generating embeddings, computing similarities, clustering, and semantic search.

## Structure

The embeddings module is organized into focused submodules:

- **`data.py`** - Data structures (EmbeddingData, SimilarityResults)
- **`checkpoint.py`** - Checkpoint management for resumable generation
- **`generation.py`** - Core embedding generation with three-phase process
- **`computation.py`** - Similarity, clustering, search, dimensionality reduction
- **`export.py`** - Export functions for various formats
- **`shutdown.py`** - Signal handling and graceful shutdown

## Usage

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

## Integration

The embeddings module is integrated into the meta-analysis workflow and can be imported via:

```python
# Direct import
from infrastructure.literature.llm.embeddings import generate_document_embeddings

# Via meta_analysis module
from infrastructure.literature.meta_analysis import generate_document_embeddings
```

## See Also

- [`../AGENTS.md`](../AGENTS.md) - LLM module documentation
- [`../../meta_analysis/AGENTS.md`](../../meta_analysis/AGENTS.md) - Meta-analysis documentation

