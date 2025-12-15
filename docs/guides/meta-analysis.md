# Meta-Analysis Guide

Complete guide to running meta-analysis on your literature library.

## Overview

Meta-analysis tools provide comprehensive analysis of your literature library, including temporal trends, keyword evolution, metadata visualization, and PCA analysis.

## Quick Start

### Command Line

```bash
# Run meta-analysis
python3 scripts/07_literature_search.py --meta-analysis --keywords "optimization"
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

## Output Files

Meta-analysis outputs are saved to `data/output/`:

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

## See Also

- **[Meta-Analysis Module Documentation](../infrastructure/literature/meta_analysis/AGENTS.md)** - Complete documentation
- **[API Reference](../reference/api-reference.md)** - API documentation


