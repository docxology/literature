# Meta-Analysis Guide

Complete guide to running meta-analysis on your literature library.

## Overview

Meta-analysis tools provide comprehensive analysis of your **existing** literature library, including temporal trends, keyword evolution, metadata visualization, and PCA analysis.

**Important**: Meta-analysis works only with existing library data (citations, PDFs, extracted text). It does not search, download, or extract. Use options 3.1 and 4.1-4.3 for those operations.

There are two modes available:
- **Standard meta-analysis** (option 6.1): Bibliographic analysis, citations, PCA, word use, source clarity, full text availability. No LLM/Ollama required.
- **Full meta-analysis with embeddings** (option 6.2): Includes all standard analysis plus Ollama-based semantic embedding analysis (similarity, clustering, semantic search). Requires Ollama server running.

## Quick Start

### Command Line

```bash
# Run standard meta-analysis on existing library (no embeddings)
python3 scripts/literature_search.py --meta-analysis

# Run full meta-analysis with embeddings on existing library (requires Ollama)
python3 scripts/literature_search.py --meta-analysis --with-embeddings
```

### Interactive Menu

```bash
# Start interactive menu
./run_literature.sh

# Select option 6.1 for standard meta-analysis
# Select option 6.2 for full meta-analysis with embeddings
```

### Python API

```python
from infrastructure.literature.meta_analysis import (
    DataAggregator,
    create_publication_timeline_plot,
    create_keyword_frequency_plot
)

# Create aggregator
aggregator = DataAggregator()

# Publication timeline
create_publication_timeline_plot()

# Keyword analysis
from infrastructure.literature.meta_analysis import extract_keywords_over_time
keyword_data = extract_keywords_over_time()
create_keyword_frequency_plot(keyword_data, top_n=20)
```

## Analysis Types

### Temporal Analysis

Analyze publication trends over time:

```python
from infrastructure.literature.meta_analysis import (
    get_publication_trends,
    create_publication_timeline_plot
)

# Get trends
trends = get_publication_trends()

# Create timeline plot
create_publication_timeline_plot()
```

### Keyword Analysis

Analyze keyword evolution and frequency:

```python
from infrastructure.literature.meta_analysis import (
    extract_keywords_over_time,
    detect_emerging_keywords,
    create_keyword_frequency_plot,
    create_keyword_evolution_plot
)

# Extract keywords
keyword_data = extract_keywords_over_time(min_frequency=3)

# Detect emerging keywords
emerging = detect_emerging_keywords(keyword_data)

# Create visualizations
create_keyword_frequency_plot(keyword_data, top_n=20)
create_keyword_evolution_plot(keyword_data)
```

### Metadata Analysis

Visualize metadata distributions:

```python
from infrastructure.literature.meta_analysis import (
    create_venue_distribution_plot,
    create_author_contributions_plot,
    create_citation_distribution_plot,
    create_metadata_completeness_plot
)

# Venue distribution
create_venue_distribution_plot()

# Author contributions
create_author_contributions_plot()

# Citation distribution
create_citation_distribution_plot()

# Metadata completeness
create_metadata_completeness_plot()
```

### PCA Analysis

Principal component analysis of paper texts:

```python
from infrastructure.literature.meta_analysis import (
    extract_text_features,
    compute_pca,
    cluster_papers,
    create_pca_2d_plot,
    create_pca_3d_plot
)

# Extract features
features = extract_text_features()

# Compute PCA
pca_result = compute_pca(features, n_components=3)

# Cluster papers
clusters = cluster_papers(pca_result, n_clusters=5)

# Visualize
create_pca_2d_plot(pca_result, clusters)
create_pca_3d_plot(pca_result, clusters)
```

## Embedding Analysis

When using `--with-embeddings` (option 6.2), comprehensive semantic analysis is performed:

### Core Features
- **Document embeddings**: Generate semantic embeddings using Ollama embeddinggemma model
- **Similarity matrix**: Compute cosine similarity between all paper pairs
- **Clustering**: K-means clustering on embedding space
- **Visualizations**: 2D/3D cluster visualizations using UMAP/t-SNE

### Validation & Quality Assurance
- **Quality validation**: Check for zero vectors, NaN/Inf values, low-variance dimensions
- **Completeness validation**: Verify embedding coverage and missing documents
- **Dimension validation**: Ensure consistent embedding dimensions
- **Similarity matrix validation**: Verify symmetry, diagonal, and value ranges
- **Outlier detection**: Identify statistical outliers using multiple methods

### Statistics & Metrics
- **Embedding statistics**: Distribution stats (mean, std, min, max per dimension)
- **Similarity statistics**: Distribution analysis with percentiles
- **Clustering metrics**: Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, inertia
- **Dimensionality analysis**: Effective dimensions, explained variance, PCA on embeddings
- **Outlier statistics**: Outlier detection and analysis

### Enhanced Visualizations
- **Quality plots**: Variance per dimension, norm distributions, mean/std per dimension
- **Similarity distribution**: Histogram of similarity values
- **Cluster quality metrics**: Bar chart of clustering quality scores
- **Silhouette analysis**: Per-sample silhouette scores visualization
- **Embedding coverage**: Space coverage visualization with convex hull
- **Outlier visualization**: Highlight outliers in 2D/3D space
- **Dimensionality analysis**: Explained variance plots, effective dimensions
- **Cluster size distribution**: Distribution of cluster sizes
- **Similarity network**: Network graph of high-similarity pairs

### Export Capabilities
- **Statistics export**: JSON and CSV formats
- **Validation reports**: Comprehensive validation results in JSON
- **Clustering metrics**: Quality metrics in JSON and CSV formats

**Requirements:**
- Ollama server running (`ollama serve`)
- Embedding model installed (`ollama pull embeddinggemma`)
- At least 2 papers with extracted text

**Configuration:**
Embedding generation includes automatic retry logic and Ollama health checks:
- **Timeout**: Default 120 seconds (configurable via `LITERATURE_EMBEDDING_TIMEOUT`)
- **Retries**: Default 3 attempts with exponential backoff (configurable via `LITERATURE_EMBEDDING_RETRY_ATTEMPTS`)
- **Health checks**: Automatically checks Ollama connection and attempts restart on timeout (configurable via `LITERATURE_EMBEDDING_RESTART_OLLAMA_ON_TIMEOUT`)
- **Caching**: Embeddings are cached to avoid regeneration (cache directory: `LITERATURE_EMBEDDING_CACHE_DIR`)

If embedding generation fails for individual papers after all retries, zero vectors are used as fallback (with warnings logged). The system will continue processing remaining papers.

**Output files (embedding analysis):**
- `embeddings.json` - Document embeddings
- `embedding_similarity_matrix.csv` - Similarity matrix
- `embedding_clusters.json` - Cluster assignments
- `embedding_validation_report.json` - Validation results
- `embedding_statistics.json` - Comprehensive statistics (JSON)
- `embedding_statistics.csv` - Comprehensive statistics (CSV)
- `clustering_metrics.json` - Clustering quality metrics (JSON)
- `clustering_metrics.csv` - Clustering quality metrics (CSV)
- `embedding_quality.png` - Quality metrics visualization
- `similarity_distribution.png` - Similarity histogram
- `embedding_coverage.png` - Space coverage visualization
- `dimensionality_analysis.png` - Dimensionality plots
- `embedding_outliers.png` - Outlier visualization (if outliers detected)
- `embedding_similarity_heatmap.png` - Similarity heatmap
- `similarity_network.png` - Similarity network graph (for smaller collections)
- `cluster_quality_metrics.png` - Clustering metrics bar chart
- `silhouette_analysis.png` - Silhouette plot
- `cluster_size_distribution.png` - Cluster size distribution
- `embedding_clusters_2d.png` - 2D cluster visualization
- `embedding_clusters_3d.png` - 3D cluster visualization (if ≥3 papers)

## Output Files

Meta-analysis outputs are saved to `data/output/`:

### Standard Analysis (always generated)
- `publications_by_year.png` - Publication timeline
- `keyword_frequency.png` - Keyword frequency plot
- `keyword_evolution.png` - Keyword evolution plot
- `venue_distribution.png` - Venue distribution
- `author_contributions.png` - Author contributions
- `citation_distribution.png` - Citation distribution
- `metadata_completeness.png` - Metadata completeness
- `pca_2d.png` - 2D PCA visualization
- `pca_3d.png` - 3D PCA visualization
- `meta_analysis_summary.json` - Summary data
- `meta_analysis_summary.md` - Text summary

### Embedding Analysis (only with --with-embeddings)
- `embeddings.json` - Document embeddings
- `embedding_similarity_matrix.csv` - Similarity matrix
- `embedding_clusters.json` - Cluster assignments
- `embedding_validation_report.json` - Validation results
- `embedding_statistics.json` / `.csv` - Comprehensive statistics
- `clustering_metrics.json` / `.csv` - Clustering quality metrics
- `embedding_quality.png` - Quality metrics visualization
- `similarity_distribution.png` - Similarity histogram
- `embedding_coverage.png` - Space coverage visualization
- `dimensionality_analysis.png` - Dimensionality plots
- `embedding_outliers.png` - Outlier visualization (if detected)
- `embedding_similarity_heatmap.png` - Similarity heatmap
- `similarity_network.png` - Similarity network graph
- `cluster_quality_metrics.png` - Clustering metrics bar chart
- `silhouette_analysis.png` - Silhouette plot
- `cluster_size_distribution.png` - Cluster size distribution
- `embedding_clusters_2d.png` - 2D cluster visualization
- `embedding_clusters_3d.png` - 3D cluster visualization

## Configuration

### Filtering

```python
# Filter by year range
from infrastructure.literature.meta_analysis import filter_by_year_range

filtered = filter_by_year_range(start_year=2020, end_year=2024)
```

### Keyword Settings

```python
# Extract keywords with custom settings
keyword_data = extract_keywords_over_time(
    min_frequency=3,      # Minimum keyword frequency
    top_n=50             # Top N keywords
)
```

## Best Practices

1. **Sufficient data** - Ensure library has enough papers for meaningful analysis
2. **Keyword selection** - Use relevant keywords for focused analysis
3. **Year filtering** - Filter by relevant time periods
4. **Visualization review** - Review generated visualizations for insights
5. **Embedding analysis** - Use option 6.2 (with embeddings) for semantic similarity analysis. Requires Ollama and extracted text from PDFs.
6. **Standard analysis** - Use option 6.1 (standard) for faster analysis without LLM dependencies

## See Also

- **[Meta-Analysis Module Documentation](../modules/literature.md)** - Module documentation
- **[API Reference](../reference/api-reference.md)** - API documentation
- **[Guides AGENTS.md](AGENTS.md)** - Guide organization and standards


