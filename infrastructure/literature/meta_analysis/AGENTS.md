# Meta-Analysis Module - Documentation

## Purpose

The meta-analysis module provides analysis and visualization tools for literature libraries, including temporal trends, keyword evolution, metadata visualization, and PCA analysis.

## Components

### DataAggregator (aggregator.py)

Aggregates library data for analysis with logging.

**Key Methods:**
- `aggregate_library_data()` - Collect all library entries
- `validate_data_quality()` - Validate and report data quality metrics
- `prepare_temporal_data()` - Year-based aggregation
- `prepare_keyword_data()` - Keyword extraction and aggregation
- `prepare_metadata_data()` - Metadata aggregation
- `prepare_text_corpus()` - Text corpus for PCA (with detailed extraction statistics)
- `prepare_classification_data()` - Classification data aggregation (category and domain distributions)

**Recent Improvements:**
- `prepare_text_corpus()` with detailed logging of extracted text vs abstract fallback counts
- Added data quality validation and reporting
- Improved error messages and diagnostics

### Temporal Analysis (temporal.py)

Publication year analysis and trends.

**Key Functions:**
- `get_publication_trends()` - Get publication trends
- `filter_by_year_range()` - Filter by year range
- `analyze_publication_rate()` - Analyze publication rate
- `create_publication_timeline_plot()` - Create timeline plot

### Keyword Analysis (keywords.py)

Keyword evolution and frequency analysis.

**Key Functions:**
- `extract_keywords_over_time()` - Extract keywords
- `detect_emerging_keywords()` - Detect trending keywords
- `create_keyword_frequency_plot()` - Frequency plot
- `create_keyword_evolution_plot()` - Evolution plot

### Metadata Visualization (metadata.py)

Metadata visualization and statistics.

**Key Functions:**
- `create_venue_distribution_plot()` - Venue distribution
- `create_author_contributions_plot()` - Author contributions
- `create_classification_distribution_plot()` - Paper classification distribution (pie chart)
- `get_metadata_summary()` - Summary statistics

### PCA Analysis (pca.py)

Principal component analysis of paper texts with logging and error handling.

**Key Functions:**
- `extract_text_features()` - TF-IDF feature extraction (returns valid_indices for alignment)
- `compute_pca()` - Principal component analysis with variance logging
- `cluster_papers()` - K-means clustering
- `create_pca_2d_plot()` - 2D visualization (with array alignment)
- `create_pca_3d_plot()` - 3D visualization (with array alignment)
- `export_pca_loadings()` - Export PCA loadings in multiple formats

**Recent Fixes:**
- Fixed array size mismatch: `extract_text_features()` now returns `valid_indices` to track which documents were kept after filtering empty texts
- All PCA plotting functions now properly align arrays (titles, years, cluster_labels) with feature_matrix
- Added validation to prevent "boolean index did not match indexed array" errors
- Logging throughout feature extraction and PCA computation

### Embedding Analysis

Semantic embedding-based analysis using Ollama's embeddinggemma model. Provides semantic understanding of paper content, complementing TF-IDF-based PCA analysis.

**Location:** `infrastructure/literature/llm/embeddings/`

**Modular Structure:**
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

**Data Structures:**
- `EmbeddingData` - Container for embeddings, citation keys, titles, years
- `SimilarityResults` - Container for similarity matrix and metadata

**Features:**
- Automatic text chunking for large documents (handles 2048 token limit)
- Mean pooling aggregation for document-level embeddings
- Embedding caching to avoid recomputation
- Batch processing for efficient API usage
- Progress tracking for large document sets
- Checkpoint resume capability (saves progress, allows resuming after interruption)
- Hung Ollama detection and automatic recovery
- Adaptive timeout scaling based on text length

**Embedding Workflow:**

1. **Pre-flight Validation** (optional, skipped if resuming from checkpoint):
   - Quick connection check (2s timeout)
   - Verify embedding model is available
   - Test embedding generation with small text (10s timeout)
   - Non-blocking: warnings logged but process continues if validation fails

2. **Cache Checking** (Phase 1/3):
   - Check existing cached embeddings for all documents
   - Progress bar for sets >50 documents
   - Periodic logging every 50 items for smaller sets

3. **Cache Loading** (Phase 2/3):
   - Load cached embeddings from disk
   - Progress bar for sets >10 items
   - Track failed cache loads

4. **Embedding Generation** (Phase 3/3):
   - Generate embeddings for missing documents
   - Automatic text chunking for large texts (>4000 chars)
   - Adaptive timeout: `min(config_timeout, max(30s, text_length/50))`
   - Checkpoint saved after each successful embedding
   - Heartbeat logging every 30s for long operations
   - On timeout: test embedding endpoint, force restart Ollama if hung, save checkpoint

**Hung Ollama Detection and Recovery:**

The system detects hung Ollama instances where the general API responds but the embedding endpoint is stuck:

- **Detection**: Tests `/api/embed` endpoint with small text (10s timeout)
- **Recovery**: Force kills hung process and restarts Ollama server
- **Verification**: Tests embedding endpoint after restart to confirm recovery
- **Checkpointing**: Progress saved even on failures, allowing resume

**Timeout Handling:**

- **Adaptive Timeouts**: Scale with text length (0.02s per char, minimum 30s)
- **Short Text Detection**: Timeouts on very short texts (<100 chars) indicate hung state
- **Force Restart**: Automatically restarts Ollama when embedding endpoint is hung
- **Retry Logic**: Exponential backoff with configurable attempts

**Checkpoint Resume:**

- Checkpoint file: `.embedding_progress.json` in cache directory
- Tracks: completed indices, citation keys, timestamp
- Resume: Automatically loads checkpoint if citation keys match current corpus
- Validation: Skips pre-flight validation when resuming (assumes Ollama is ready)

**Requirements:**
- Ollama server running with embeddinggemma model installed
- `scikit-learn` for clustering and t-SNE
- `umap-learn` (optional) for UMAP dimensionality reduction

### Visualizations (visualizations.py)

Plotting utilities for all visualizations with error handling and logging.

**Key Functions:**
- `plot_publications_by_year()` - Year-based bar chart (red trend line removed)
- `plot_keyword_frequency()` - Keyword frequency
- `plot_keyword_cooccurrence()` - Keyword co-occurrence heatmap
- `plot_venue_distribution()` - Venue distribution
- `plot_author_contributions()` - Author contributions
- `plot_pca_2d()` - 2D PCA plot with features:
  - Confidence ellipses around clusters
  - Distance vectors from cluster centers
  - Word importance vectors overlay
  - Correlation circle for variable contributions
- `plot_pca_3d()` - 3D PCA plot with features:
  - Confidence ellipsoids around clusters
  - 3D distance vectors
  - 3D word importance vectors
- `plot_pca_loadings_heatmap()` - PCA loadings heatmap
- `plot_pca_loadings_barplot()` - PCA loadings bar charts
- `plot_pca_biplot()` - PCA biplot (papers and word vectors)
- `plot_pca_word_vectors()` - Word vectors in PC space
- `plot_metadata_completeness()` - Metadata completeness chart
- `plot_author_collaboration_network()` - Author collaboration network
- `plot_source_distribution()` - Source distribution pie chart
- `plot_topic_evolution()` - Topic evolution over time
- `plot_embedding_similarity_heatmap()` - Heatmap of paper-to-paper similarities (embeddings)
- `plot_embedding_clusters_2d()` - 2D embedding clusters with coloring
- `plot_embedding_clusters_3d()` - 3D embedding clusters with coloring
- `plot_semantic_search_results()` - Bar chart of semantic search results
- `save_plot()` - Save plot to file

**Recent Improvements:**
- Fixed array size mismatch issues in PCA visualizations (854 vs 853 bug)
- Added array alignment validation
- Error handling with detailed diagnostics
- Added extensive logging throughout visualization pipeline
- Removed red trend line from publications-by-year chart (bar chart only)
- PCA visualizations with confidence ellipses/ellipsoids, distance vectors, word vectors, and correlation circles

### Additional Visualizations (additional_visualizations.py)

Additional visualization types for meta-analysis.

**Key Functions:**
- `plot_citation_vs_year()` - Scatter plot of citations vs publication year
- `plot_venue_trends()` - Line plot showing publication trends by venue over time
- `plot_author_productivity()` - Bar chart of publications per author
- `plot_citation_network()` - Network graph of citation relationships (placeholder)
- `plot_topic_distribution()` - Distribution of topics/themes across papers
- `plot_word_cloud()` - Word cloud visualization of most common terms
- `plot_correlation_matrix()` - Correlation matrix of metadata fields
- `plot_publication_heatmap()` - Heatmap of publications by year and venue
- `create_citation_vs_year_plot()` - Convenience function with aggregator
- `create_venue_trends_plot()` - Convenience function with aggregator
- `create_author_productivity_plot()` - Convenience function with aggregator
- `create_publication_heatmap_plot()` - Convenience function with aggregator

### Graphical Abstracts (graphical_abstract.py)

Composite visualizations combining multiple plots.

**Key Functions:**
- `create_single_page_abstract()` - Single-page composite with 6 visualizations
- `create_multi_page_abstract()` - Multi-page PDF with one visualization per page
- `create_graphical_abstract()` - Graphical abstract (currently uses single-page)
- `create_composite_panel()` - Auto-sized composite panel:
  - Automatically determines optimal grid size (e.g., 5x4, 6x4)
  - Includes all available visualizations
  - Handles missing data gracefully
  - Supports high-resolution output

## Usage Examples

### Temporal Analysis

```python
from infrastructure.literature.meta_analysis import (
    DataAggregator,
    create_publication_timeline_plot
)

aggregator = DataAggregator()
create_publication_timeline_plot()
```

### Keyword Analysis

```python
from infrastructure.literature.meta_analysis import (
    extract_keywords_over_time,
    create_keyword_frequency_plot
)

keyword_data = extract_keywords_over_time(min_frequency=3)
create_keyword_frequency_plot(keyword_data, top_n=20)
```

### PCA Analysis

```python
from infrastructure.literature.meta_analysis import create_pca_2d_plot

create_pca_2d_plot(n_clusters=5)
```

### Embedding Analysis

```python
from infrastructure.literature.meta_analysis import (
    generate_document_embeddings,
    compute_similarity_matrix,
    cluster_embeddings,
    find_similar_papers,
    reduce_dimensions,
    plot_embedding_similarity_heatmap,
    plot_embedding_clusters_2d,
)

from infrastructure.literature.meta_analysis.aggregator import DataAggregator

# Generate embeddings
aggregator = DataAggregator()
corpus = aggregator.prepare_text_corpus()
embedding_data = generate_document_embeddings(corpus)

# Compute similarity
similarity_matrix = compute_similarity_matrix(embedding_data.embeddings)

# Cluster papers
cluster_labels = cluster_embeddings(embedding_data.embeddings, n_clusters=5)

# Semantic search
results = find_similar_papers(
    embedding_data,
    "machine learning neural networks",
    top_k=10
)

# Visualize
fig = plot_embedding_similarity_heatmap(
    similarity_matrix=similarity_matrix,
    citation_keys=embedding_data.citation_keys,
    titles=embedding_data.titles
)

# 2D cluster visualization
embeddings_2d = reduce_dimensions(embedding_data.embeddings, n_components=2)
fig = plot_embedding_clusters_2d(
    embeddings_2d=embeddings_2d,
    cluster_labels=cluster_labels,
    titles=embedding_data.titles
)
```

### PCA with Confidence Ellipses and Word Vectors

```python
from infrastructure.literature.meta_analysis import plot_pca_2d
from infrastructure.literature.meta_analysis.pca import extract_text_features, compute_pca

# Get feature matrix and loadings
feature_matrix, feature_names, valid_indices = extract_text_features(corpus)
pca_data, pca_model = compute_pca(feature_matrix, n_components=2)
loadings_matrix = pca_model.components_.T

# Create plot
fig = plot_pca_2d(
    pca_data=pca_data,
    titles=titles,
    years=years,
    cluster_labels=cluster_labels,
    explained_variance=pca_model.explained_variance_ratio_,
    show_confidence_ellipses=True,
    show_distance_vectors=True,
    show_word_vectors=True,
    show_correlation_circle=True,
    loadings_matrix=loadings_matrix,
    feature_names=feature_names
)
```

### Additional Visualizations

```python
from infrastructure.literature.meta_analysis import (
    create_citation_vs_year_plot,
    create_venue_trends_plot,
    create_author_productivity_plot,
    create_publication_heatmap_plot
)

# Create various visualizations
create_citation_vs_year_plot()
create_venue_trends_plot(top_n_venues=10)
create_author_productivity_plot(top_n_authors=20)
create_publication_heatmap_plot(top_n_venues=15)
```

### Composite Panel

```python
from infrastructure.literature.meta_analysis import create_composite_panel

# Create auto-sized composite panel with all available visualizations
create_composite_panel(max_panels=20)
```

## Dependencies

- `matplotlib` - Plotting
- `numpy` - Numerical operations
- `scikit-learn` - PCA and clustering (optional, for PCA features)
- `umap-learn` - UMAP dimensionality reduction (optional, for embedding visualizations)
- `seaborn` - Statistical visualizations (optional, for heatmaps)
- `networkx` - Network graphs (optional, for collaboration networks)
- `pandas` - Data manipulation (optional, for correlation matrix)
- `wordcloud` - Word cloud generation (optional, for word cloud visualization)
- `PIL/Pillow` - Image processing (optional, for multi-page abstracts)
- `requests` - HTTP requests (for Ollama API)
- Ollama server with embeddinggemma model (for embedding analysis)

## Recent Updates and Fixes

### Array Alignment Fix (Critical)
- **Issue**: Array size mismatches (854 vs 853) causing "boolean index did not match indexed array" errors
- **Solution**: Modified `extract_text_features()` to return `valid_indices` tracking which documents were kept after filtering empty texts
- **Impact**: All PCA visualizations now properly align arrays (titles, years, cluster_labels) with feature matrices
- **Files Changed**: `pca.py`, `pca_loadings.py`, `graphical_abstract.py`, `visualizations.py`

### Logging
- Added debug and info logging throughout the pipeline
- Logging includes:
  - Feature extraction statistics (samples, features, filtered counts)
  - Array alignment validation
  - PCA computation progress and variance explained
  - Clustering statistics
  - Text corpus preparation details

### Improved Error Handling
- Added array size validation in all plotting functions
- Error messages with detailed diagnostics
- Full traceback logging for debugging
- Graceful fallbacks for missing optional dependencies

### New Visualizations
- `plot_author_collaboration_network()` - Network graph of author collaborations
- `plot_source_distribution()` - Pie chart of paper sources
- `plot_topic_evolution()` - Topic/keyword evolution over time
- `plot_classification_distribution()` - Pie chart of paper classifications (category and domain)
- Additional visualizations module with 8+ visualization types
- Composite panel generator with auto-sizing

### PCA Visualizations
- Confidence ellipses/ellipsoids around clusters
- Distance vectors from cluster centers to data points
- Word importance vectors overlaid on PCA space
- Correlation circles showing variable contributions
- All features are toggleable via function parameters

### Embedding Analysis
- Semantic embedding generation using Ollama embeddinggemma model
- Cosine similarity matrix computation
- K-means clustering on embedding space
- Semantic search functionality
- 2D/3D visualization with UMAP/t-SNE
- Similarity heatmaps
- Automatic text chunking for large documents
- Embedding caching for performance
- **Validation**: Quality checks, completeness validation, dimension validation, similarity matrix validation, outlier detection
- **Statistics computation**: Embedding statistics, similarity statistics, clustering quality metrics (silhouette score, Davies-Bouldin index, Calinski-Harabasz score), dimensionality analysis
- **Visualizations**: Embedding quality plots, similarity distribution, cluster quality metrics, silhouette analysis, embedding coverage, outlier visualization, dimensionality analysis, cluster size distribution, similarity network graphs
- **Export capabilities**: Statistics (JSON/CSV), validation reports (JSON), clustering metrics (JSON/CSV)
- **Optional**: Controlled via `include_embeddings` parameter in `run_meta_analysis()`
- **Menu option**: 6.2 (full meta-analysis with embeddings) vs 6.1 (standard, no embeddings)
- **CLI flag**: `--with-embeddings` (requires `--meta-analysis`)
- **Requirements**: Ollama server running, embedding model installed, ≥2 papers with extracted text

### Documentation
- Updated all function docstrings with detailed parameter descriptions
- Added error handling documentation
- Usage examples

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


