# Meta-Analysis Module - Complete Documentation

## Purpose

The meta-analysis module provides comprehensive analysis and visualization tools for literature libraries, including temporal trends, keyword evolution, metadata visualization, and PCA analysis.

## Components

### DataAggregator (aggregator.py)

Aggregates library data for analysis with comprehensive logging.

**Key Methods:**
- `aggregate_library_data()` - Collect all library entries
- `validate_data_quality()` - Validate and report data quality metrics
- `prepare_temporal_data()` - Year-based aggregation
- `prepare_keyword_data()` - Keyword extraction and aggregation
- `prepare_metadata_data()` - Metadata aggregation
- `prepare_text_corpus()` - Text corpus for PCA (with detailed extraction statistics)

**Recent Improvements:**
- Enhanced `prepare_text_corpus()` with detailed logging of extracted text vs abstract fallback counts
- Added comprehensive data quality validation and reporting
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
- `create_citation_distribution_plot()` - Citation distribution
- `get_metadata_summary()` - Summary statistics

### PCA Analysis (pca.py)

Principal component analysis of paper texts with comprehensive logging and error handling.

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
- Added comprehensive validation to prevent "boolean index did not match indexed array" errors
- Enhanced logging throughout feature extraction and PCA computation

### Visualizations (visualizations.py)

Plotting utilities for all visualizations with comprehensive error handling and logging.

**Key Functions:**
- `plot_publications_by_year()` - Year-based plot
- `plot_keyword_frequency()` - Keyword frequency
- `plot_keyword_cooccurrence()` - Keyword co-occurrence heatmap
- `plot_venue_distribution()` - Venue distribution
- `plot_author_contributions()` - Author contributions
- `plot_citation_distribution()` - Citation distribution histogram
- `plot_pca_2d()` - 2D PCA plot (with array alignment validation)
- `plot_pca_3d()` - 3D PCA plot (with array alignment validation)
- `plot_pca_loadings_heatmap()` - PCA loadings heatmap
- `plot_pca_loadings_barplot()` - PCA loadings bar charts
- `plot_pca_biplot()` - PCA biplot (papers and word vectors)
- `plot_pca_word_vectors()` - Word vectors in PC space
- `plot_metadata_completeness()` - Metadata completeness chart
- `plot_author_collaboration_network()` - Author collaboration network (NEW)
- `plot_source_distribution()` - Source distribution pie chart (NEW)
- `plot_topic_evolution()` - Topic evolution over time (NEW)
- `save_plot()` - Save plot to file

**Recent Improvements:**
- Fixed array size mismatch issues in PCA visualizations (854 vs 853 bug)
- Added comprehensive array alignment validation
- Enhanced error handling with detailed diagnostics
- Added extensive logging throughout visualization pipeline
- Added new visualization types (collaboration networks, source distribution, topic evolution)

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

## Dependencies

- `matplotlib` - Plotting
- `numpy` - Numerical operations
- `scikit-learn` - PCA and clustering (optional, for PCA features)
- `networkx` - Network graphs (optional, for collaboration networks)

## Recent Updates and Fixes

### Array Alignment Fix (Critical)
- **Issue**: Array size mismatches (854 vs 853) causing "boolean index did not match indexed array" errors
- **Solution**: Modified `extract_text_features()` to return `valid_indices` tracking which documents were kept after filtering empty texts
- **Impact**: All PCA visualizations now properly align arrays (titles, years, cluster_labels) with feature matrices
- **Files Changed**: `pca.py`, `pca_loadings.py`, `graphical_abstract.py`, `visualizations.py`

### Enhanced Logging
- Added comprehensive debug and info logging throughout the pipeline
- Logging includes:
  - Feature extraction statistics (samples, features, filtered counts)
  - Array alignment validation
  - PCA computation progress and variance explained
  - Clustering statistics
  - Text corpus preparation details

### Improved Error Handling
- Added array size validation in all plotting functions
- Enhanced error messages with detailed diagnostics
- Full traceback logging for debugging
- Graceful fallbacks for missing optional dependencies

### New Visualizations
- `plot_author_collaboration_network()` - Network graph of author collaborations
- `plot_source_distribution()` - Pie chart of paper sources
- `plot_topic_evolution()` - Topic/keyword evolution over time

### Documentation
- Updated all function docstrings with detailed parameter descriptions
- Added comprehensive error handling documentation
- Enhanced usage examples

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


