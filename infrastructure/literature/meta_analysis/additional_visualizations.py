"""Additional visualization types for meta-analysis.

Provides additional visualization types beyond the core set, including
correlation matrices, network graphs, word clouds, and trend analyses.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.meta_analysis.aggregator import (
    DataAggregator,
    LibraryEntry,
)
from infrastructure.literature.meta_analysis.visualizations import (
    FONT_SIZE_LABELS,
    FONT_SIZE_TITLE,
    FONT_SIZE_LEGEND,
    GRID_ALPHA,
    EDGE_WIDTH,
    COLORMAP_YEAR,
    COLORMAP_CATEGORICAL,
    save_plot,
)

logger = get_logger(__name__)


def plot_citation_vs_year(
    entries: List[LibraryEntry],
    title: str = "Citations vs Publication Year"
) -> plt.Figure:
    """Plot scatter plot of citations vs publication year.
    
    Shows the relationship between publication year and citation count,
    useful for identifying trends in citation patterns over time.
    
    Args:
        entries: List of library entries with year and citation_count.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data
    years = []
    citations = []
    for entry in entries:
        if entry.year and entry.citation_count is not None:
            years.append(entry.year)
            citations.append(entry.citation_count)
    
    if not years:
        ax.text(0.5, 0.5, 'No citation data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Create scatter plot
    scatter = ax.scatter(
        years, citations,
        alpha=0.6, s=100, c=years,
        cmap=COLORMAP_YEAR, edgecolors='black', linewidth=EDGE_WIDTH
    )
    
    # Add trend line
    if len(years) > 1:
        z = np.polyfit(years, citations, 1)
        p = np.poly1d(z)
        ax.plot(years, p(years), "r--", alpha=0.8, linewidth=2, label='Trend')
        ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND, framealpha=0.9)
    
    ax.set_xlabel('Publication Year', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Citation Count', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Publication Year')
    cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1)
    
    plt.tight_layout()
    return fig


def plot_venue_trends(
    entries: List[LibraryEntry],
    top_n_venues: int = 10,
    title: str = "Publication Trends by Venue"
) -> plt.Figure:
    """Plot line chart showing publication trends by venue over time.
    
    Shows how publication counts for top venues change over time.
    
    Args:
        entries: List of library entries.
        top_n_venues: Number of top venues to display.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Aggregate by venue and year
    venue_years: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        if entry.venue and entry.year:
            venue_years[entry.venue][entry.year] += 1
    
    if not venue_years:
        ax.text(0.5, 0.5, 'No venue data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Get top venues by total count
    venue_totals = {venue: sum(counts.values()) for venue, counts in venue_years.items()}
    top_venues = sorted(venue_totals.items(), key=lambda x: x[1], reverse=True)[:top_n_venues]
    
    # Get all years
    all_years = set()
    for counts in venue_years.values():
        all_years.update(counts.keys())
    all_years = sorted(all_years)
    
    # Plot lines for each venue
    try:
        cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
    except AttributeError:
        cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
    
    for i, (venue, _) in enumerate(top_venues):
        years_list = []
        counts_list = []
        for year in all_years:
            count = venue_years[venue].get(year, 0)
            if count > 0 or len(years_list) == 0:  # Include zero years for continuity
                years_list.append(year)
                counts_list.append(count)
        
        color = cmap(i % cmap.N)
        ax.plot(years_list, counts_list, marker='o', linewidth=2, markersize=6,
               label=venue[:50], color=color, alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Number of Publications', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND - 2, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    return fig


def plot_author_productivity(
    entries: List[LibraryEntry],
    top_n_authors: int = 20,
    title: str = "Author Productivity Over Time"
) -> plt.Figure:
    """Plot bar chart of publications per author over time.
    
    Shows the most productive authors and their publication counts.
    
    Args:
        entries: List of library entries.
        top_n_authors: Number of top authors to display.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Count publications per author
    author_counts: Dict[str, int] = defaultdict(int)
    for entry in entries:
        if entry.authors:
            for author in entry.authors:
                author_counts[author] += 1
    
    if not author_counts:
        ax.text(0.5, 0.5, 'No author data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Get top authors
    top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_authors]
    authors = [a for a, _ in top_authors]
    counts = [c for _, c in top_authors]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(authors))
    ax.barh(y_pos, counts, alpha=0.8, color='#2E86AB', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(authors, fontsize=FONT_SIZE_LABELS - 2)
    ax.set_xlabel('Number of Publications', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Author', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_citation_network(
    entries: List[LibraryEntry],
    title: str = "Citation Network"
) -> plt.Figure:
    """Plot network graph of citation relationships.
    
    Note: This is a placeholder for future citation network analysis.
    Currently shows a message if citation network data is not available.
    
    Args:
        entries: List of library entries.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Check if we have citation relationships
    has_citations = any(e.citation_count is not None for e in entries)
    
    if not has_citations:
        ax.text(0.5, 0.5, 'Citation network data not available\n(Requires citation relationship data)',
               ha='center', va='center', fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Future: Implement citation network visualization
    # This would require citation relationship data (who cites whom)
    ax.text(0.5, 0.5, 'Citation network visualization\n(Feature coming soon)',
           ha='center', va='center', fontsize=FONT_SIZE_LABELS)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig


def plot_topic_distribution(
    entries: List[LibraryEntry],
    title: str = "Topic Distribution"
) -> plt.Figure:
    """Plot distribution of topics/themes across papers.
    
    Uses keywords from abstracts to identify topic distribution.
    
    Args:
        entries: List of library entries with abstracts.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract keywords from abstracts
    from infrastructure.literature.meta_analysis.aggregator import DataAggregator
    aggregator = DataAggregator()
    keyword_data = aggregator.prepare_keyword_data()
    
    if not keyword_data.keyword_counts:
        ax.text(0.5, 0.5, 'No topic/keyword data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Get top keywords
    top_keywords = sorted(
        keyword_data.keyword_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]
    
    keywords = [k for k, _ in top_keywords]
    counts = [c for _, c in top_keywords]
    
    # Create bar chart
    y_pos = np.arange(len(keywords))
    ax.barh(y_pos, counts, alpha=0.8, color='#6A4C93', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keywords, fontsize=FONT_SIZE_LABELS - 2)
    ax.set_xlabel('Frequency', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Topic/Keyword', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_word_cloud(
    entries: List[LibraryEntry],
    title: str = "Word Cloud"
) -> plt.Figure:
    """Plot word cloud visualization of most common terms.
    
    Args:
        entries: List of library entries with abstracts/text.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    try:
        from wordcloud import WordCloud
        WORDCLOUD_AVAILABLE = True
    except ImportError:
        WORDCLOUD_AVAILABLE = False
        logger.warning("wordcloud library not available, creating simplified visualization")
    
    # Collect text from abstracts
    text_corpus = []
    for entry in entries:
        if entry.abstract:
            text_corpus.append(entry.abstract)
    
    if not text_corpus:
        ax.text(0.5, 0.5, 'No text data available for word cloud', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    combined_text = ' '.join(text_corpus)
    
    if WORDCLOUD_AVAILABLE:
        # Generate word cloud
        wordcloud = WordCloud(
            width=1200, height=800,
            background_color='white',
            max_words=200,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42
        ).generate(combined_text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
    else:
        # Fallback: simple frequency bar chart
        from infrastructure.literature.meta_analysis.aggregator import DataAggregator
        aggregator = DataAggregator()
        keyword_data = aggregator.prepare_keyword_data()
        
        top_keywords = sorted(
            keyword_data.keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:30]
        
        keywords = [k for k, _ in top_keywords]
        counts = [c for _, c in top_keywords]
        
        y_pos = np.arange(len(keywords))
        ax.barh(y_pos, counts, alpha=0.8, color='#2E86AB', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keywords, fontsize=FONT_SIZE_LABELS - 3)
        ax.set_xlabel('Frequency', fontsize=FONT_SIZE_LABELS, fontweight='medium')
        ax.set_ylabel('Word', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    entries: List[LibraryEntry],
    title: str = "Metadata Correlation Matrix"
) -> plt.Figure:
    """Plot correlation matrix of metadata fields.
    
    Shows correlations between year, citations, and other numeric fields.
    
    Args:
        entries: List of library entries.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract numeric data
    data = {
        'year': [],
        'citations': [],
        'author_count': [],
    }
    
    for entry in entries:
        if entry.year:
            data['year'].append(entry.year)
            data['citations'].append(entry.citation_count if entry.citation_count else 0)
            data['author_count'].append(len(entry.authors) if entry.authors else 0)
    
    if not data['year']:
        ax.text(0.5, 0.5, 'Insufficient data for correlation matrix', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Create DataFrame-like structure for correlation
    import pandas as pd
    try:
        df = pd.DataFrame(data)
        corr_matrix = df.corr()
        
        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        labels = list(corr_matrix.columns)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=FONT_SIZE_LABELS - 2)
        ax.set_yticklabels(labels, fontsize=FONT_SIZE_LABELS - 2)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=FONT_SIZE_LABELS - 3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontsize=FONT_SIZE_LEGEND - 1)
        
    except ImportError:
        # Fallback if pandas not available
        ax.text(0.5, 0.5, 'Pandas required for correlation matrix', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
    
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_publication_heatmap(
    entries: List[LibraryEntry],
    top_n_venues: int = 15,
    title: str = "Publications by Year and Venue"
) -> plt.Figure:
    """Plot heatmap of publications by year and venue.
    
    Shows publication patterns across venues over time.
    
    Args:
        entries: List of library entries.
        top_n_venues: Number of top venues to display.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Aggregate by venue and year
    venue_years: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for entry in entries:
        if entry.venue and entry.year:
            venue_years[entry.venue][entry.year] += 1
    
    if not venue_years:
        ax.text(0.5, 0.5, 'No venue/year data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Get top venues
    venue_totals = {venue: sum(counts.values()) for venue, counts in venue_years.items()}
    top_venues = sorted(venue_totals.items(), key=lambda x: x[1], reverse=True)[:top_n_venues]
    venue_list = [v for v, _ in top_venues]
    
    # Get all years
    all_years = set()
    for counts in venue_years.values():
        all_years.update(counts.keys())
    all_years = sorted(all_years)
    
    # Create matrix
    heatmap_data = []
    for venue in venue_list:
        row = [venue_years[venue].get(year, 0) for year in all_years]
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    
    # Plot heatmap
    im = ax.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(all_years)))
    ax.set_yticks(np.arange(len(venue_list)))
    ax.set_xticklabels(all_years, fontsize=FONT_SIZE_LABELS - 3, rotation=45, ha='right')
    ax.set_yticklabels([v[:40] for v in venue_list], fontsize=FONT_SIZE_LABELS - 3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Publications', fontsize=FONT_SIZE_LEGEND - 1)
    
    ax.set_xlabel('Year', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Venue', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig


# Convenience functions that work with aggregator

def create_citation_vs_year_plot(
    aggregator: Optional[DataAggregator] = None,
    output_path: Optional[Path] = None,
    format: str = "png"
) -> Path:
    """Create citation vs year plot from aggregator.
    
    Args:
        aggregator: Optional DataAggregator instance.
        output_path: Optional output path.
        format: Output format (png, svg, pdf).
        
    Returns:
        Path to saved plot.
    """
    if aggregator is None:
        aggregator = DataAggregator()
    
    entries = aggregator.aggregate_library_data()
    
    if output_path is None:
        output_path = Path("data/output/citation_vs_year." + format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plot_citation_vs_year(entries)
    return save_plot(fig, output_path)


def create_venue_trends_plot(
    aggregator: Optional[DataAggregator] = None,
    output_path: Optional[Path] = None,
    top_n_venues: int = 10,
    format: str = "png"
) -> Path:
    """Create venue trends plot from aggregator.
    
    Args:
        aggregator: Optional DataAggregator instance.
        output_path: Optional output path.
        top_n_venues: Number of top venues to display.
        format: Output format (png, svg, pdf).
        
    Returns:
        Path to saved plot.
    """
    if aggregator is None:
        aggregator = DataAggregator()
    
    entries = aggregator.aggregate_library_data()
    
    if output_path is None:
        output_path = Path("data/output/venue_trends." + format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plot_venue_trends(entries, top_n_venues=top_n_venues)
    return save_plot(fig, output_path)


def create_author_productivity_plot(
    aggregator: Optional[DataAggregator] = None,
    output_path: Optional[Path] = None,
    top_n_authors: int = 20,
    format: str = "png"
) -> Path:
    """Create author productivity plot from aggregator.
    
    Args:
        aggregator: Optional DataAggregator instance.
        output_path: Optional output path.
        top_n_authors: Number of top authors to display.
        format: Output format (png, svg, pdf).
        
    Returns:
        Path to saved plot.
    """
    if aggregator is None:
        aggregator = DataAggregator()
    
    entries = aggregator.aggregate_library_data()
    
    if output_path is None:
        output_path = Path("data/output/author_productivity." + format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plot_author_productivity(entries, top_n_authors=top_n_authors)
    return save_plot(fig, output_path)


def create_publication_heatmap_plot(
    aggregator: Optional[DataAggregator] = None,
    output_path: Optional[Path] = None,
    top_n_venues: int = 15,
    format: str = "png"
) -> Path:
    """Create publication heatmap plot from aggregator.
    
    Args:
        aggregator: Optional DataAggregator instance.
        output_path: Optional output path.
        top_n_venues: Number of top venues to display.
        format: Output format (png, svg, pdf).
        
    Returns:
        Path to saved plot.
    """
    if aggregator is None:
        aggregator = DataAggregator()
    
    entries = aggregator.aggregate_library_data()
    
    if output_path is None:
        output_path = Path("data/output/publication_heatmap." + format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plot_publication_heatmap(entries, top_n_venues=top_n_venues)
    return save_plot(fig, output_path)

