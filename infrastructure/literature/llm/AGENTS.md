# LLM Module - Documentation

## Purpose

The LLM module provides advanced LLM operations for multi-paper synthesis, configurable paper selection, and embedding-based analysis.

## Components

### Embeddings Submodule (embeddings/)

Modular embedding generation and analysis system for semantic paper analysis.

**Structure:**
- `data.py` - Data structures (EmbeddingData, SimilarityResults)
- `checkpoint.py` - Checkpoint management for resumable generation
- `generation.py` - Core embedding generation with three-phase process
- `computation.py` - Similarity, clustering, search, dimensionality reduction
- `export.py` - Export functions for various formats
- `shutdown.py` - Signal handling and graceful shutdown

**Key Functions:**
- `generate_document_embeddings()` - Generate embeddings for all documents in corpus
- `compute_similarity_matrix()` - Compute cosine similarity matrix from embeddings
- `cluster_embeddings()` - K-means clustering on embedding space
- `find_similar_papers()` - Semantic search to find papers similar to a query
- `reduce_dimensions()` - Dimensionality reduction (UMAP/t-SNE) for visualization
- `export_embeddings()` - Export embeddings to JSON
- `export_similarity_matrix()` - Export similarity matrix to CSV
- `export_clusters()` - Export cluster assignments to JSON

**Features:**
- Automatic text chunking for large documents (handles 2048 token limit)
- Mean pooling aggregation for document-level embeddings
- Embedding caching to avoid recomputation
- Batch processing for efficient API usage
- Progress tracking for large document sets
- Checkpoint resume capability (saves progress, allows resuming after interruption)
- Hung Ollama detection and automatic recovery
- Adaptive timeout scaling based on text length

**Note:** The embeddings module is located at `infrastructure/literature/llm/embeddings/` but is also accessible via backward-compatible imports from `infrastructure.literature.meta_analysis.embeddings`.

### LiteratureLLMOperations (operations.py)

Advanced LLM operations for synthesizing information across multiple papers.

**Key Methods:**
- `generate_literature_review()` - Generate literature review synthesis (300-500 words, multi-paper)
- `generate_science_communication()` - Create science communication narratives (600-800 words, general audience)
- `generate_comparative_analysis()` - Compare methods/findings across papers (500-700 words)
- `generate_research_gaps()` - Identify research gaps and future directions (400-600 words)
- `analyze_citation_network()` - Analyze intellectual connections between papers (500-700 words, text-based)

**Note:** All operations generate text-based analyses. The citation network analysis is text-based (not graph visualization).

### PaperSelector (selector.py)

Configurable paper selection and filtering.

**Key Methods:**
- `select_papers(library_entries)` - Filter papers based on criteria, returns List[LibraryEntry]
- `from_config(config_path)` - Create selector from YAML config file (classmethod)
- `get_selection_summary(selected_papers, total_papers)` - Get selection statistics

**Selection Criteria:**
- Citation keys (specific papers by key)
- Year range (min/max)
- Source filtering (arxiv, semanticscholar, etc.)
- PDF availability (has_pdf: true/false)
- Summary availability (has_summary: true/false)
- Keyword matching (keywords in title or abstract)
- Limit (maximum number of papers to select)

## Usage Examples

### LLM Operations

```python
from infrastructure.literature.llm import LiteratureLLMOperations
from infrastructure.literature.library import LibraryIndex

llm_ops = LiteratureLLMOperations()
entries = LibraryIndex(config).list_entries()

# Generate literature review
result = llm_ops.generate_literature_review(
    papers=entries[:10],
    focus="methods"
)
```

### Paper Selection

```python
from infrastructure.literature.llm import PaperSelector

selector = PaperSelector.from_config("paper_selection.yaml")
selected = selector.select_papers(library_entries)
```

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


