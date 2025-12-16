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

## Progress Logging

The embedding generation process provides detailed progress visibility at multiple levels:

### Document-Level Progress

Each document being processed shows:
- Citation key and position in queue
- Text length and estimated timeout
- Chunk count estimate (for large documents)

Example:
```
[11/129] Generating embedding for: malekzadeh2022active
  Text length: 204,107 chars | Estimated timeout: 240.0s
  Starting chunk processing for malekzadeh2022active...
  → Calling Ollama embedding API for malekzadeh2022active...
```

### Chunk Processing

For documents that require chunking (>4000 chars), each chunk shows:
- Chunk number and total count
- Chunk size in characters
- Estimated processing time
- Completion status with average time and ETA

Example:
```
  → Document will be processed in 55 chunks (text length: 204,107 chars)
  → Average chunk size: 3,711 chars
  → Per-chunk timeout: 92.8s
  → Processing chunk 1/55 (3,979 chars, ~99.5s estimated)
  → Sending embedding request to Ollama (text: 3,979 chars, timeout: 92.8s, model: embeddinggemma)...
  → Request monitor started (heartbeat every 20s)
  ✓ Chunk 1/55 completed (avg: 1.0s/chunk, ETA: 55s remaining)
```

### Request Monitoring

For long-running requests (timeout > 30s), a background monitor provides periodic heartbeats to indicate the request is still processing:

Example:
```
  → Request monitor started (heartbeat every 20s)
  ↻ Still processing embedding request... (15.2s elapsed, 77.6s remaining, text length: 3,979 chars)
```

The heartbeat interval adapts based on text length:
- Very long documents (>50K chars): 5s intervals
- Medium documents (10K-50K chars): 10s intervals
- Shorter documents: 20s intervals

### Overall Progress

Periodic summaries show overall generation progress with:
- Completed count and total
- Elapsed time and average time per embedding
- Recent average (last 5 embeddings)
- Estimated time remaining (ETA)

Example:
```
  Progress: 11/129 generated (2m 15s, avg: 12.3s/embedding, recent avg: 1.2s, ETA: 1h 9m)
  Generating embeddings [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 11/129 (8%) ✓:11 ✗:0 ETA: 1h 9m
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

