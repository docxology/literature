"""Visualization utilities for meta-analysis.

Provides plotting functions for temporal, keyword, metadata,
and PCA visualizations with enhanced accessibility features.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from infrastructure.core.logging_utils import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

# Accessibility constants
FONT_SIZE_LABELS = 16
FONT_SIZE_TITLE = 18
FONT_SIZE_LEGEND = 16  # Increased from 14 for better readability
FONT_SIZE_TICKS = 12  # Tick label font size
FONT_SIZE_CBAR = 14  # Colorbar label font size
GRID_ALPHA = 0.3
EDGE_WIDTH = 0.8
MARKER_SIZE = 120

# Colorblind-friendly colormaps
COLORMAP_YEAR = 'plasma'  # Continuous colormap for years
COLORMAP_CATEGORICAL = 'tab10'  # For discrete categories


# Helper functions for PCA enhancements

def _compute_confidence_ellipse(
    data: np.ndarray,
    n_std: float = 2.0
) -> Tuple[float, float, float, float, float]:
    """Compute confidence ellipse parameters for 2D data.
    
    Args:
        data: 2D array of shape (n_samples, 2).
        n_std: Number of standard deviations for ellipse.
        
    Returns:
        Tuple of (center_x, center_y, width, height, angle) in degrees.
    """
    if len(data) < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    
    if cov.size == 0 or np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return (mean[0], mean[1], 0.0, 0.0, 0.0)
    
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    
    # Angle in degrees
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    
    # Width and height
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    return (mean[0], mean[1], width, height, angle)


def _plot_distance_vectors(
    ax: plt.Axes,
    pca_data: np.ndarray,
    cluster_labels: np.ndarray,
    alpha: float = 0.3,
    linewidth: float = 0.5
) -> None:
    """Plot distance vectors from cluster centers to data points.
    
    Args:
        ax: Matplotlib axes.
        pca_data: 2D PCA-transformed data.
        cluster_labels: Cluster labels for each point.
        alpha: Transparency of vectors.
        linewidth: Width of vector lines.
    """
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_data = pca_data[mask]
        
        if len(cluster_data) < 2:
            continue
        
        # Compute cluster center
        center = np.mean(cluster_data, axis=0)
        
        # Plot vectors from center to each point
        for point in cluster_data:
            ax.plot([center[0], point[0]], [center[1], point[1]],
                   'k-', alpha=alpha, linewidth=linewidth, zorder=0)


def _plot_word_importance_vectors(
    ax: plt.Axes,
    loadings_matrix: np.ndarray,
    feature_names: List[str],
    top_n_words: int = 20,
    scale_factor: float = 3.0,
    alpha: float = 0.7
) -> None:
    """Overlay word importance vectors on PCA space.
    
    Args:
        ax: Matplotlib axes.
        loadings_matrix: (n_features, n_components) loadings matrix.
        feature_names: List of feature (word) names.
        top_n_words: Number of top words to display.
        scale_factor: Scaling factor for vectors.
        alpha: Transparency of vectors.
    """
    if loadings_matrix.shape[1] < 2:
        return
    
    # Calculate overall importance
    importance_scores = np.sum(np.abs(loadings_matrix), axis=1)
    top_indices = np.argsort(importance_scores)[::-1][:top_n_words]
    top_words = [feature_names[i] for i in top_indices]
    top_loadings = loadings_matrix[top_indices, :2]
    
    # Plot vectors as arrows
    for word, loading in zip(top_words, top_loadings):
        arrow_length = np.linalg.norm(loading) * scale_factor
        if arrow_length > 0:
            ax.arrow(0, 0, loading[0] * scale_factor, loading[1] * scale_factor,
                    head_width=0.02, head_length=0.02, fc='red', ec='red',
                    alpha=alpha, linewidth=1.5, zorder=10)
            # Add word label
            ax.text(loading[0] * scale_factor * 1.1, loading[1] * scale_factor * 1.1,
                   word, fontsize=FONT_SIZE_LABELS - 4, alpha=0.8, fontweight='medium',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7,
                           edgecolor='none'), zorder=11)


def _plot_correlation_circle(
    ax: plt.Axes,
    loadings_matrix: np.ndarray,
    feature_names: List[str],
    top_n_words: int = 20,
    radius: float = 1.0,
    alpha: float = 0.6
) -> None:
    """Plot correlation circle showing variable contributions.
    
    Args:
        ax: Matplotlib axes.
        loadings_matrix: (n_features, n_components) loadings matrix.
        feature_names: List of feature (word) names.
        top_n_words: Number of top words to display.
        radius: Radius of the correlation circle.
        alpha: Transparency of circle and vectors.
    """
    if loadings_matrix.shape[1] < 2:
        return
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    ax.plot(circle_x, circle_y, 'k--', alpha=alpha, linewidth=1, zorder=5)
    
    # Draw axes
    ax.axhline(y=0, color='k', linestyle='--', alpha=alpha*0.5, linewidth=0.5, zorder=5)
    ax.axvline(x=0, color='k', linestyle='--', alpha=alpha*0.5, linewidth=0.5, zorder=5)
    
    # Calculate overall importance for word selection
    importance_scores = np.sum(np.abs(loadings_matrix), axis=1)
    top_indices = np.argsort(importance_scores)[::-1][:top_n_words]
    top_words = [feature_names[i] for i in top_indices]
    top_loadings = loadings_matrix[top_indices, :2]
    
    # Normalize loadings to unit circle
    for word, loading in zip(top_words, top_loadings):
        norm = np.linalg.norm(loading)
        if norm > 0:
            normalized = loading / norm * radius
            # Plot vector
            ax.arrow(0, 0, normalized[0], normalized[1],
                    head_width=0.03, head_length=0.03, fc='blue', ec='blue',
                    alpha=alpha, linewidth=1.2, zorder=6)
            # Add word label
            ax.text(normalized[0] * 1.1, normalized[1] * 1.1,
                   word, fontsize=FONT_SIZE_LABELS - 4, alpha=0.8, fontweight='medium',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7,
                           edgecolor='none'), zorder=7)


def plot_publications_by_year(
    years: List[int],
    counts: List[int],
    title: str = "Publications by Year"
) -> plt.Figure:
    """Plot publications by year with enhanced accessibility.
    
    Args:
        years: List of years.
        counts: List of publication counts per year.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Use colorblind-friendly colors with better contrast
    ax.bar(years, counts, alpha=0.8, color='#2E86AB', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    
    ax.set_xlabel('Year', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Number of Publications', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    
    # Rotate x-axis labels if many years
    if len(years) > 10:
        plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE_LABELS - 2)
    else:
        plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_keyword_frequency(
    keywords: List[str],
    counts: Optional[List[int]] = None,
    frequency_data: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    title: str = "Keyword Frequency",
    show_evolution: bool = False
) -> plt.Figure:
    """Plot keyword frequency with enhanced accessibility.
    
    Args:
        keywords: List of keywords.
        counts: List of keyword counts (for bar chart).
        frequency_data: Dictionary mapping keywords to (year, count) lists (for evolution).
        title: Plot title.
        show_evolution: Whether to show evolution over time.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    if show_evolution and frequency_data:
        # Plot evolution lines with distinct colors and markers
        try:
            cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
        except AttributeError:
            cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
        colors = cmap(np.linspace(0, 1, len(keywords)))
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P']
        
        for i, keyword in enumerate(keywords):
            if keyword in frequency_data:
                data = frequency_data[keyword]
                years = [d[0] for d in data]
                freqs = [d[1] for d in data]
                marker = markers[i % len(markers)]
                ax.plot(years, freqs, marker=marker, label=keyword, linewidth=2.5, 
                       markersize=8, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Year', fontsize=FONT_SIZE_LABELS, fontweight='medium')
        ax.set_ylabel('Frequency', fontsize=FONT_SIZE_LABELS, fontweight='medium')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZE_LEGEND, 
                 framealpha=0.9, title='Keywords', title_fontsize=FONT_SIZE_LEGEND + 1)
    else:
        # Bar chart with color gradient
        if counts is None:
            counts = [1] * len(keywords)
        
        y_pos = np.arange(len(keywords))
        # Use color gradient for better visual distinction
        try:
            cmap = plt.colormaps.get_cmap(COLORMAP_YEAR)
        except AttributeError:
            cmap = plt.get_cmap(COLORMAP_YEAR)
        colors = cmap(np.linspace(0.2, 0.8, len(keywords)))
        bars = ax.barh(y_pos, counts, alpha=0.8, color=colors, edgecolor='#1B4F72', 
                      linewidth=EDGE_WIDTH)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(keywords, fontsize=FONT_SIZE_LABELS - 2)
        ax.set_xlabel('Frequency', fontsize=FONT_SIZE_LABELS, fontweight='medium')
        ax.set_ylabel('Keywords', fontsize=FONT_SIZE_LABELS, fontweight='medium')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {int(count)}', va='center', fontsize=FONT_SIZE_LEGEND - 1)
    
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--', axis='x' if not show_evolution else 'both')
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_keyword_cooccurrence(
    cooccurrence_matrix: np.ndarray,
    keywords: List[str],
    title: str = "Keyword Co-occurrence"
) -> plt.Figure:
    """Plot keyword co-occurrence heatmap.
    
    Args:
        cooccurrence_matrix: Matrix of co-occurrence counts.
        keywords: List of keywords.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(cooccurrence_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(keywords)))
    ax.set_yticks(np.arange(len(keywords)))
    ax.set_xticklabels(keywords, rotation=45, ha='right')
    ax.set_yticklabels(keywords)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Co-occurrence Count')
    
    plt.tight_layout()
    return fig


def plot_venue_distribution(
    venues: List[str],
    counts: List[int],
    title: str = "Venue Distribution"
) -> plt.Figure:
    """Plot venue distribution with enhanced accessibility.
    
    Args:
        venues: List of venue names.
        counts: List of publication counts per venue.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    y_pos = np.arange(len(venues))
    # Use color gradient for better visual distinction
    try:
        cmap = plt.colormaps.get_cmap(COLORMAP_YEAR)
    except AttributeError:
        cmap = plt.get_cmap(COLORMAP_YEAR)
    colors = cmap(np.linspace(0.2, 0.8, len(venues)))
    bars = ax.barh(y_pos, counts, alpha=0.8, color=colors, edgecolor='#1B4F72', 
                  linewidth=EDGE_WIDTH)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(venues, fontsize=FONT_SIZE_LABELS - 2)
    ax.set_xlabel('Number of Publications', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Venues', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {int(count)}', va='center', fontsize=FONT_SIZE_LEGEND - 1)
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_author_contributions(
    authors: List[str],
    counts: List[int],
    title: str = "Author Contributions"
) -> plt.Figure:
    """Plot author contributions with enhanced accessibility.
    
    Args:
        authors: List of author names.
        counts: List of publication counts per author.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    y_pos = np.arange(len(authors))
    # Use colorblind-friendly green gradient
    try:
        cmap = plt.colormaps.get_cmap('viridis')
    except AttributeError:
        cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0.3, 0.9, len(authors)))
    bars = ax.barh(y_pos, counts, alpha=0.8, color=colors, edgecolor='#1B4F72', 
                  linewidth=EDGE_WIDTH)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(authors, fontsize=FONT_SIZE_LABELS - 2)
    ax.set_xlabel('Number of Publications', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Authors', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {int(count)}', va='center', fontsize=FONT_SIZE_LEGEND - 1)
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_citation_distribution(
    citation_counts: List[int],
    title: str = "Citation Distribution"
) -> plt.Figure:
    """Plot citation distribution histogram with enhanced accessibility.
    
    Args:
        citation_counts: List of citation counts.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use colorblind-friendly color
    n, bins, patches = ax.hist(citation_counts, bins=50, alpha=0.8, 
                               color='#6A4C93', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    
    # Color gradient for histogram bars
    try:
        cmap = plt.colormaps.get_cmap(COLORMAP_YEAR)
    except AttributeError:
        cmap = plt.get_cmap(COLORMAP_YEAR)
    colors = cmap(np.linspace(0.3, 0.9, len(patches)))
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Citation Count', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, axis='y', linestyle='--')
    
    # Add statistics with better visibility
    mean_citations = np.mean(citation_counts)
    median_citations = np.median(citation_counts)
    std_citations = np.std(citation_counts)
    
    ax.axvline(mean_citations, color='#E63946', linestyle='--', linewidth=2.5, 
              label=f'Mean: {mean_citations:.1f}', alpha=0.9)
    ax.axvline(median_citations, color='#2A9D8F', linestyle='--', linewidth=2.5, 
              label=f'Median: {median_citations:.1f}', alpha=0.9)
    ax.axvline(mean_citations + std_citations, color='#F77F00', linestyle=':', linewidth=2, 
              label=f'Mean + 1σ: {mean_citations + std_citations:.1f}', alpha=0.7)
    ax.axvline(mean_citations - std_citations, color='#F77F00', linestyle=':', linewidth=2, 
              label=f'Mean - 1σ: {mean_citations - std_citations:.1f}', alpha=0.7)
    
    ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND, framealpha=0.9, 
             title='Statistics', title_fontsize=FONT_SIZE_LEGEND + 1)
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_pca_2d(
    pca_data: np.ndarray,
    titles: List[str],
    years: List[Optional[int]],
    cluster_labels: Optional[np.ndarray] = None,
    explained_variance: Optional[np.ndarray] = None,
    title: str = "PCA Analysis (2D)",
    show_confidence_ellipses: bool = True,
    show_distance_vectors: bool = False,
    show_word_vectors: bool = False,
    show_correlation_circle: bool = False,
    loadings_matrix: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    top_n_words: int = 20
) -> plt.Figure:
    """Plot 2D PCA visualization with continuous year coloring and cluster shapes.
    
    Uses continuous colormap for years and distinct marker shapes for clusters.
    Enhanced with accessibility features and comprehensive legends.
    Supports optional confidence ellipses, distance vectors, word importance vectors,
    and correlation circles.
    
    Args:
        pca_data: 2D PCA-transformed data.
        titles: List of paper titles.
        years: List of publication years.
        cluster_labels: Optional cluster labels for marker shapes.
        explained_variance: Explained variance ratio for each component.
        title: Plot title.
        show_confidence_ellipses: Whether to draw confidence ellipses around clusters.
        show_distance_vectors: Whether to plot vectors from cluster centers to points.
        show_word_vectors: Whether to overlay word importance vectors.
        show_correlation_circle: Whether to plot correlation circle.
        loadings_matrix: (n_features, n_components) loadings matrix for word vectors.
        feature_names: List of feature (word) names for word vectors.
        top_n_words: Number of top words to display in vectors/circle.
        
    Returns:
        Matplotlib figure.
        
    Raises:
        ValueError: If array sizes don't match.
    """
    # Validate array sizes
    n_samples = len(pca_data)
    if len(titles) != n_samples:
        raise ValueError(
            f"Array size mismatch in plot_pca_2d: pca_data has {n_samples} samples, "
            f"but titles has {len(titles)} entries"
        )
    if len(years) != n_samples:
        raise ValueError(
            f"Array size mismatch in plot_pca_2d: pca_data has {n_samples} samples, "
            f"but years has {len(years)} entries"
        )
    if cluster_labels is not None and len(cluster_labels) != n_samples:
        raise ValueError(
            f"Array size mismatch in plot_pca_2d: pca_data has {n_samples} samples, "
            f"but cluster_labels has {len(cluster_labels)} entries"
        )
    
    logger.debug(f"Creating PCA 2D plot: {n_samples} samples, "
                f"{len(titles)} titles, {len(years)} years, "
                f"clusters={'yes' if cluster_labels is not None else 'no'}")
    
    fig, ax = plt.subplots(figsize=(16, 12))  # Increased for better legend spacing
    
    # Prepare year data for continuous coloring
    years_array = None
    if years and any(y is not None for y in years):
        valid_years = [y for y in years if y is not None]
        if valid_years:
            min_year = min(valid_years)
            years_array = np.array([y if y is not None else min_year for y in years])
            logger.debug(f"Year coloring: range {years_array.min()}-{years_array.max()}, "
                        f"{len(valid_years)} valid years out of {len(years)}")
    
    # Marker shapes for clusters (colorblind-friendly)
    marker_map = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P', '<', '>']
    
    if cluster_labels is not None and years_array is not None:
        # Dual encoding: color by year (continuous), shape by cluster
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            marker = marker_map[cluster_id % len(marker_map)]
            
            scatter = ax.scatter(
                pca_data[mask, 0],
                pca_data[mask, 1],
                c=years_array[mask],
                cmap=COLORMAP_YEAR,
                marker=marker,
                alpha=0.7,
                s=MARKER_SIZE,
                edgecolors='black',
                linewidth=EDGE_WIDTH,
                label=f'Cluster {int(cluster_id)}',
                vmin=years_array.min(),
                vmax=years_array.max()
            )
        
        # Add colorbar for year (positioned on right side)
        cbar = plt.colorbar(scatter, ax=ax, label='Publication Year', pad=0.08)
        cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
        cbar.set_label('Publication Year', fontsize=FONT_SIZE_CBAR, fontweight='medium')
        
        # Add legend for clusters (positioned at bottom-left to avoid overlap with colorbar)
        n_clusters = len(unique_clusters)
        ncol = 2 if n_clusters > 5 else 1  # Use 2 columns if many clusters
        ax.legend(loc='lower left', fontsize=FONT_SIZE_LEGEND, framealpha=0.9, 
                 title='Cluster Type', title_fontsize=FONT_SIZE_LEGEND + 1,
                 ncol=ncol, bbox_to_anchor=(0.0, 0.0), borderaxespad=0.5)
        
    elif cluster_labels is not None:
        # Only cluster coloring (no year data)
        unique_clusters = np.unique(cluster_labels)
        try:
            cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
        except AttributeError:
            cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
        colors = cmap(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            marker = marker_map[cluster_id % len(marker_map)]
            
            ax.scatter(
                pca_data[mask, 0],
                pca_data[mask, 1],
                c=[colors[i]],
                marker=marker,
                alpha=0.7,
                s=MARKER_SIZE,
                edgecolors='black',
                linewidth=EDGE_WIDTH,
                label=f'Cluster {int(cluster_id)}'
            )
        
        ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND, framealpha=0.9, 
                 title='Cluster Type', title_fontsize=FONT_SIZE_LEGEND + 1)
        
    elif years_array is not None:
        # Only year coloring (continuous)
        scatter = ax.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            c=years_array,
            cmap=COLORMAP_YEAR,
            alpha=0.7,
            s=MARKER_SIZE,
            edgecolors='black',
            linewidth=EDGE_WIDTH
        )
        cbar = plt.colorbar(scatter, ax=ax, label='Publication Year', pad=0.08)
        cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1)
        cbar.set_label('Publication Year', fontsize=FONT_SIZE_LEGEND, fontweight='medium')
    else:
        # Default: single color
        ax.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            alpha=0.7,
            s=MARKER_SIZE,
            color='#2E86AB',
            edgecolors='black',
            linewidth=EDGE_WIDTH
        )
    
    # Add confidence ellipses around clusters
    if show_confidence_ellipses and cluster_labels is not None:
        from matplotlib.patches import Ellipse
        unique_clusters = np.unique(cluster_labels)
        try:
            cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
        except AttributeError:
            cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            cluster_data = pca_data[mask]
            
            if len(cluster_data) >= 2:
                center_x, center_y, width, height, angle = _compute_confidence_ellipse(cluster_data, n_std=2.0)
                if width > 0 and height > 0:
                    ellipse = Ellipse(
                        (center_x, center_y), width, height, angle=angle,
                        fill=False, edgecolor=cmap(i % cmap.N), linewidth=2,
                        linestyle='--', alpha=0.6, zorder=1
                    )
                    ax.add_patch(ellipse)
    elif show_confidence_ellipses:
        logger.debug("Confidence ellipses requested but no cluster labels provided")
    
    # Add distance vectors from cluster centers
    if show_distance_vectors and cluster_labels is not None:
        _plot_distance_vectors(ax, pca_data, cluster_labels, alpha=0.3, linewidth=0.5)
    
    # Add word importance vectors
    if show_word_vectors and loadings_matrix is not None and feature_names is not None:
        _plot_word_importance_vectors(ax, loadings_matrix, feature_names, top_n_words, scale_factor=3.0)
    
    # Add correlation circle
    if show_correlation_circle and loadings_matrix is not None and feature_names is not None:
        _plot_correlation_circle(ax, loadings_matrix, feature_names, top_n_words, radius=1.0)
    
    # Enhanced axis labels with variance explained
    xlabel = f'PC1 ({explained_variance[0]*100:.1f}% variance)' if explained_variance is not None else 'PC1'
    ylabel = f'PC2 ({explained_variance[1]*100:.1f}% variance)' if explained_variance is not None else 'PC2'
    
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_pca_3d(
    pca_data: np.ndarray,
    titles: List[str],
    years: List[Optional[int]],
    cluster_labels: Optional[np.ndarray] = None,
    explained_variance: Optional[np.ndarray] = None,
    title: str = "PCA Analysis (3D)",
    show_confidence_ellipsoids: bool = True,
    show_distance_vectors: bool = False,
    show_word_vectors: bool = False,
    loadings_matrix: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    top_n_words: int = 20
) -> plt.Figure:
    """Plot 3D PCA visualization with continuous year coloring and cluster shapes.
    
    Uses continuous colormap for years and distinct marker shapes for clusters.
    Enhanced with accessibility features and comprehensive legends.
    Supports optional confidence ellipsoids, distance vectors, and word importance vectors.
    
    Args:
        pca_data: 3D PCA-transformed data.
        titles: List of paper titles.
        years: List of publication years.
        cluster_labels: Optional cluster labels for marker shapes.
        explained_variance: Explained variance ratio for each component.
        title: Plot title.
        show_confidence_ellipsoids: Whether to draw confidence ellipsoids around clusters.
        show_distance_vectors: Whether to plot vectors from cluster centers to points.
        show_word_vectors: Whether to overlay word importance vectors (projected to 3D).
        loadings_matrix: (n_features, n_components) loadings matrix for word vectors.
        feature_names: List of feature (word) names for word vectors.
        top_n_words: Number of top words to display in vectors.
        
    Returns:
        Matplotlib figure.
        
    Raises:
        ValueError: If array sizes don't match.
    """
    # Validate array sizes
    n_samples = len(pca_data)
    if len(titles) != n_samples:
        raise ValueError(
            f"Array size mismatch in plot_pca_3d: pca_data has {n_samples} samples, "
            f"but titles has {len(titles)} entries"
        )
    if len(years) != n_samples:
        raise ValueError(
            f"Array size mismatch in plot_pca_3d: pca_data has {n_samples} samples, "
            f"but years has {len(years)} entries"
        )
    if cluster_labels is not None and len(cluster_labels) != n_samples:
        raise ValueError(
            f"Array size mismatch in plot_pca_3d: pca_data has {n_samples} samples, "
            f"but cluster_labels has {len(cluster_labels)} entries"
        )
    
    logger.debug(f"Creating PCA 3D plot: {n_samples} samples, "
                f"{len(titles)} titles, {len(years)} years, "
                f"clusters={'yes' if cluster_labels is not None else 'no'}")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare year data for continuous coloring
    years_array = None
    if years and any(y is not None for y in years):
        valid_years = [y for y in years if y is not None]
        if valid_years:
            min_year = min(valid_years)
            years_array = np.array([y if y is not None else min_year for y in years])
            logger.debug(f"Year coloring: range {years_array.min()}-{years_array.max()}, "
                        f"{len(valid_years)} valid years out of {len(years)}")
    
    # Marker shapes for clusters (colorblind-friendly)
    marker_map = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P', '<', '>']
    
    if cluster_labels is not None and years_array is not None:
        # Dual encoding: color by year (continuous), shape by cluster
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            marker = marker_map[cluster_id % len(marker_map)]
            
            scatter = ax.scatter(
                pca_data[mask, 0],
                pca_data[mask, 1],
                pca_data[mask, 2],
                c=years_array[mask],
                cmap=COLORMAP_YEAR,
                marker=marker,
                alpha=0.7,
                s=MARKER_SIZE,
                edgecolors='black',
                linewidth=EDGE_WIDTH,
                label=f'Cluster {int(cluster_id)}',
                vmin=years_array.min(),
                vmax=years_array.max()
            )
        
        # Add colorbar for year
        cbar = plt.colorbar(scatter, ax=ax, label='Publication Year', shrink=0.6, pad=0.1)
        cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1)
        cbar.set_label('Publication Year', fontsize=FONT_SIZE_LEGEND, fontweight='medium')
        
        # Add legend for clusters
        ax.legend(loc='upper left', fontsize=FONT_SIZE_LEGEND, framealpha=0.9, 
                 title='Cluster Type', title_fontsize=FONT_SIZE_LEGEND + 1)
        
    elif cluster_labels is not None:
        # Only cluster coloring (no year data)
        unique_clusters = np.unique(cluster_labels)
        try:
            cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
        except AttributeError:
            cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
        colors = cmap(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            marker = marker_map[cluster_id % len(marker_map)]
            
            ax.scatter(
                pca_data[mask, 0],
                pca_data[mask, 1],
                pca_data[mask, 2],
                c=[colors[i]],
                marker=marker,
                alpha=0.7,
                s=MARKER_SIZE,
                edgecolors='black',
                linewidth=EDGE_WIDTH,
                label=f'Cluster {int(cluster_id)}'
            )
        
        ax.legend(loc='upper left', fontsize=FONT_SIZE_LEGEND, framealpha=0.9, 
                 title='Cluster Type', title_fontsize=FONT_SIZE_LEGEND + 1)
        
    elif years_array is not None:
        # Only year coloring (continuous)
        scatter = ax.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            pca_data[:, 2],
            c=years_array,
            cmap=COLORMAP_YEAR,
            alpha=0.7,
            s=MARKER_SIZE,
            edgecolors='black',
            linewidth=EDGE_WIDTH
        )
        cbar = plt.colorbar(scatter, ax=ax, label='Publication Year', shrink=0.6, pad=0.1)
        cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1)
        cbar.set_label('Publication Year', fontsize=FONT_SIZE_LEGEND, fontweight='medium')
    else:
        # Default: single color
        ax.scatter(
            pca_data[:, 0],
            pca_data[:, 1],
            pca_data[:, 2],
            alpha=0.7,
            s=MARKER_SIZE,
            color='#2E86AB',
            edgecolors='black',
            linewidth=EDGE_WIDTH
        )
    
    # Add confidence ellipsoids around clusters (3D)
    if show_confidence_ellipsoids and cluster_labels is not None:
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            unique_clusters = np.unique(cluster_labels)
            try:
                cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
            except AttributeError:
                cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = cluster_labels == cluster_id
                cluster_data = pca_data[mask]
                
                if len(cluster_data) >= 3:
                    # Compute mean and covariance
                    mean = np.mean(cluster_data, axis=0)
                    cov = np.cov(cluster_data.T)
                    
                    if cov.size > 0 and not (np.any(np.isnan(cov)) or np.any(np.isinf(cov))):
                        # Create ellipsoid (simplified: show as wireframe)
                        # For full ellipsoid, would need to generate mesh
                        # Here we show center and approximate bounds
                        u, s, vh = np.linalg.svd(cov)
                        radii = 2.0 * np.sqrt(s)  # 2 standard deviations
                        
                        # Draw approximate ellipsoid as wireframe
                        # This is a simplified visualization
                        color = cmap(i % cmap.N)
                        ax.scatter([mean[0]], [mean[1]], [mean[2]], 
                                  c=[color], s=200, marker='x', linewidth=3, alpha=0.8)
        except Exception as e:
            logger.debug(f"Could not draw 3D ellipsoids: {e}")
    elif show_confidence_ellipsoids:
        logger.debug("Confidence ellipsoids requested but no cluster labels provided")
    
    # Add distance vectors from cluster centers (3D)
    if show_distance_vectors and cluster_labels is not None:
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_data = pca_data[mask]
            
            if len(cluster_data) >= 2:
                center = np.mean(cluster_data, axis=0)
                for point in cluster_data:
                    ax.plot([center[0], point[0]], [center[1], point[1]], [center[2], point[2]],
                           'k-', alpha=0.2, linewidth=0.5)
    
    # Add word importance vectors (3D projection)
    if show_word_vectors and loadings_matrix is not None and feature_names is not None:
        if loadings_matrix.shape[1] >= 3:
            # Calculate overall importance
            importance_scores = np.sum(np.abs(loadings_matrix), axis=1)
            top_indices = np.argsort(importance_scores)[::-1][:top_n_words]
            top_words = [feature_names[i] for i in top_indices]
            top_loadings = loadings_matrix[top_indices, :3]
            
            # Plot vectors as arrows
            for word, loading in zip(top_words, top_loadings):
                arrow_length = np.linalg.norm(loading) * 3.0
                if arrow_length > 0:
                    # 3D arrow (simplified as line with marker)
                    ax.plot([0, loading[0] * 3.0], [0, loading[1] * 3.0], [0, loading[2] * 3.0],
                           'r-', alpha=0.7, linewidth=1.5)
                    ax.scatter([loading[0] * 3.0], [loading[1] * 3.0], [loading[2] * 3.0],
                              c='red', s=50, marker='>', alpha=0.8)
                    # Add word label (positioned at end of vector)
                    ax.text(loading[0] * 3.0 * 1.1, loading[1] * 3.0 * 1.1, loading[2] * 3.0 * 1.1,
                           word, fontsize=FONT_SIZE_LABELS - 4, alpha=0.8)
    
    # Enhanced axis labels with variance explained
    xlabel = f'PC1 ({explained_variance[0]*100:.1f}%)' if explained_variance is not None else 'PC1'
    ylabel = f'PC2 ({explained_variance[1]*100:.1f}%)' if explained_variance is not None else 'PC2'
    zlabel = f'PC3 ({explained_variance[2]*100:.1f}%)' if explained_variance is not None else 'PC3'
    
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS, fontweight='medium', labelpad=10)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS, fontweight='medium', labelpad=10)
    ax.set_zlabel(zlabel, fontsize=FONT_SIZE_LABELS, fontweight='medium', labelpad=10)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
    
    # Improve tick labels
    ax.tick_params(labelsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_pca_loadings_heatmap(
    loadings_matrix: np.ndarray,
    feature_names: List[str],
    n_components: int,
    top_n_words: int = 50,
    title: str = "PCA Loadings Heatmap"
) -> plt.Figure:
    """Plot PCA loadings as heatmap (words × components).
    
    Shows the loading values for top words across all components.
    
    Args:
        loadings_matrix: (n_features, n_components) loadings matrix.
        feature_names: List of feature (word) names.
        n_components: Number of principal components.
        top_n_words: Number of top words to display (by absolute loading).
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    # Calculate overall importance (sum of absolute loadings)
    importance_scores = np.sum(np.abs(loadings_matrix), axis=1)
    top_indices = np.argsort(importance_scores)[::-1][:top_n_words]
    
    # Extract top words and their loadings
    top_words = [feature_names[i] for i in top_indices]
    top_loadings = loadings_matrix[top_indices, :]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, n_components * 2), max(10, top_n_words * 0.3)))
    
    # Create heatmap
    im = ax.imshow(top_loadings, cmap='RdBu_r', aspect='auto', 
                   vmin=-np.abs(top_loadings).max(), vmax=np.abs(top_loadings).max())
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_components))
    ax.set_xticklabels([f'PC{i+1}' for i in range(n_components)], fontsize=FONT_SIZE_LABELS - 1)
    ax.set_yticks(np.arange(len(top_words)))
    ax.set_yticklabels(top_words, fontsize=FONT_SIZE_LABELS - 3)
    
    # Labels and title
    ax.set_xlabel('Principal Component', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Words', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Loading Value', pad=0.02)
    cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1)
    cbar.set_label('Loading Value', fontsize=FONT_SIZE_LEGEND, fontweight='medium')
    
    plt.tight_layout()
    return fig


def plot_pca_loadings_barplot(
    top_words: Dict[int, List[Tuple[str, float]]],
    explained_variance: np.ndarray,
    n_components: int = 5,
    top_n_words: int = 15,
    title: str = "Top Words per Principal Component"
) -> plt.Figure:
    """Plot bar charts for top words per component.
    
    Creates subplots showing top contributing words for each component.
    
    Args:
        top_words: Dictionary mapping component index to list of (word, loading) tuples.
        explained_variance: Explained variance ratio for each component.
        n_components: Number of components to plot.
        top_n_words: Number of top words per component to show.
        title: Overall plot title.
        
    Returns:
        Matplotlib figure.
    """
    # Determine grid layout
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_components == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_components > 1 else [axes]
    
    for comp_idx in range(n_components):
        ax = axes[comp_idx]
        
        if comp_idx in top_words and top_words[comp_idx]:
            words_data = top_words[comp_idx][:top_n_words]
            words = [w for w, _ in words_data]
            loadings = [l for _, l in words_data]
            
            # Color bars by loading sign
            colors = ['#2A9D8F' if l > 0 else '#E63946' for l in loadings]
            
            y_pos = np.arange(len(words))
            bars = ax.barh(y_pos, loadings, alpha=0.8, color=colors, 
                          edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
            
            # Add value labels
            for i, (bar, loading) in enumerate(zip(bars, loadings)):
                width = bar.get_width()
                ax.text(width if loading > 0 else width, bar.get_y() + bar.get_height()/2,
                       f' {loading:.3f}', va='center', 
                       fontsize=FONT_SIZE_LEGEND - 2, fontweight='medium')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=FONT_SIZE_LABELS - 3)
            ax.set_xlabel('Loading', fontsize=FONT_SIZE_LABELS - 1, fontweight='medium')
            
            var_explained = explained_variance[comp_idx] * 100
            ax.set_title(f'PC{comp_idx+1} ({var_explained:.1f}% variance)', 
                        fontsize=FONT_SIZE_LABELS, fontweight='bold')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
            ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   fontsize=FONT_SIZE_LABELS)
            ax.set_title(f'PC{comp_idx+1}', fontsize=FONT_SIZE_LABELS, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_components, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_pca_biplot(
    pca_data: np.ndarray,
    loadings_matrix: np.ndarray,
    feature_names: List[str],
    titles: List[str],
    years: List[Optional[int]],
    cluster_labels: Optional[np.ndarray] = None,
    explained_variance: Optional[np.ndarray] = None,
    top_n_words: int = 20,
    scale_factor: float = 3.0,
    title: str = "PCA Biplot (Papers and Word Vectors)"
) -> plt.Figure:
    """Plot biplot showing both papers (points) and word vectors (arrows).
    
    Papers are shown as points, word vectors as arrows indicating
    the direction and strength of word contributions to the principal components.
    
    Args:
        pca_data: 2D PCA-transformed data for papers.
        loadings_matrix: (n_features, n_components) loadings matrix.
        feature_names: List of feature (word) names.
        titles: List of paper titles.
        years: List of publication years.
        cluster_labels: Optional cluster labels for coloring papers.
        explained_variance: Explained variance ratio for each component.
        top_n_words: Number of top words to display as vectors.
        scale_factor: Scaling factor for word vectors (larger = longer arrows).
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Calculate overall importance for word selection
    importance_scores = np.sum(np.abs(loadings_matrix), axis=1)
    top_indices = np.argsort(importance_scores)[::-1][:top_n_words]
    top_words = [feature_names[i] for i in top_indices]
    top_loadings = loadings_matrix[top_indices, :2]  # Only first 2 components
    
    # Prepare year data for paper coloring
    years_array = None
    if years and any(y is not None for y in years):
        valid_years = [y for y in years if y is not None]
        if valid_years:
            min_year = min(valid_years)
            years_array = np.array([y if y is not None else min_year for y in years])
    
    # Plot papers
    marker_map = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P', '<', '>']
    
    if cluster_labels is not None and years_array is not None:
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            marker = marker_map[cluster_id % len(marker_map)]
            scatter = ax.scatter(
                pca_data[mask, 0], pca_data[mask, 1],
                c=years_array[mask], cmap=COLORMAP_YEAR,
                marker=marker, alpha=0.6, s=MARKER_SIZE * 0.8,
                edgecolors='black', linewidth=EDGE_WIDTH,
                label=f'Cluster {int(cluster_id)}',
                vmin=years_array.min(), vmax=years_array.max()
            )
        cbar = plt.colorbar(scatter, ax=ax, label='Publication Year', pad=0.08)
        cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1)
        cbar.set_label('Publication Year', fontsize=FONT_SIZE_LEGEND, fontweight='medium')
        ax.legend(loc='lower left', fontsize=FONT_SIZE_LEGEND, framealpha=0.9,
                 title='Cluster Type', title_fontsize=FONT_SIZE_LEGEND + 1, ncol=2)
    elif years_array is not None:
        scatter = ax.scatter(
            pca_data[:, 0], pca_data[:, 1],
            c=years_array, cmap=COLORMAP_YEAR,
            alpha=0.6, s=MARKER_SIZE * 0.8,
            edgecolors='black', linewidth=EDGE_WIDTH
        )
        cbar = plt.colorbar(scatter, ax=ax, label='Publication Year', pad=0.08)
        cbar.ax.tick_params(labelsize=FONT_SIZE_LEGEND - 1)
        cbar.set_label('Publication Year', fontsize=FONT_SIZE_LEGEND, fontweight='medium')
    else:
        ax.scatter(
            pca_data[:, 0], pca_data[:, 1],
            alpha=0.6, s=MARKER_SIZE * 0.8,
            color='#2E86AB', edgecolors='black', linewidth=EDGE_WIDTH
        )
    
    # Plot word vectors as arrows
    for i, (word, loading) in enumerate(zip(top_words, top_loadings)):
        # Scale the vector
        arrow_length = np.linalg.norm(loading) * scale_factor
        if arrow_length > 0:
            ax.arrow(0, 0, loading[0] * scale_factor, loading[1] * scale_factor,
                    head_width=0.02, head_length=0.02, fc='red', ec='red', alpha=0.7, linewidth=1.5)
            # Add word label
            ax.text(loading[0] * scale_factor * 1.1, loading[1] * scale_factor * 1.1,
                   word, fontsize=FONT_SIZE_LABELS - 4, alpha=0.8, fontweight='medium',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Axis labels
    xlabel = f'PC1 ({explained_variance[0]*100:.1f}% variance)' if explained_variance is not None else 'PC1'
    ylabel = f'PC2 ({explained_variance[1]*100:.1f}% variance)' if explained_variance is not None else 'PC2'
    
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_pca_word_vectors(
    loadings_matrix: np.ndarray,
    feature_names: List[str],
    explained_variance: np.ndarray,
    top_n_words: int = 30,
    title: str = "Word Vectors in Principal Component Space"
) -> plt.Figure:
    """Plot word vectors in PC space (first 2 components).
    
    Shows word contributions as vectors in the principal component space.
    
    Args:
        loadings_matrix: (n_features, n_components) loadings matrix.
        feature_names: List of feature (word) names.
        explained_variance: Explained variance ratio for each component.
        top_n_words: Number of top words to display.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate overall importance
    importance_scores = np.sum(np.abs(loadings_matrix), axis=1)
    top_indices = np.argsort(importance_scores)[::-1][:top_n_words]
    top_words = [feature_names[i] for i in top_indices]
    top_loadings = loadings_matrix[top_indices, :2]  # First 2 components
    
    # Plot vectors as arrows from origin
    for word, loading in zip(top_words, top_loadings):
        arrow_length = np.linalg.norm(loading)
        if arrow_length > 0:
            # Color by quadrant
            if loading[0] >= 0 and loading[1] >= 0:
                color = '#2A9D8F'  # Green (top-right)
            elif loading[0] < 0 and loading[1] >= 0:
                color = '#E63946'  # Red (top-left)
            elif loading[0] >= 0 and loading[1] < 0:
                color = '#F77F00'  # Orange (bottom-right)
            else:
                color = '#6A4C93'  # Purple (bottom-left)
            
            ax.arrow(0, 0, loading[0], loading[1],
                    head_width=0.02, head_length=0.02, fc=color, ec=color,
                    alpha=0.7, linewidth=1.5)
            
            # Add word label
            ax.text(loading[0] * 1.1, loading[1] * 1.1, word,
                   fontsize=FONT_SIZE_LABELS - 4, alpha=0.8, fontweight='medium',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Axis labels
    xlabel = f'PC1 ({explained_variance[0]*100:.1f}% variance)' if explained_variance is not None else 'PC1'
    ylabel = f'PC2 ({explained_variance[1]*100:.1f}% variance)' if explained_variance is not None else 'PC2'
    
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_metadata_completeness(
    completeness_stats: Dict[str, Dict[str, Any]],
    title: str = "Metadata Completeness"
) -> plt.Figure:
    """Plot metadata completeness as horizontal bar chart.
    
    Shows the fraction of papers with each metadata field available.
    Uses color gradient (green = high completeness, red = low).
    
    Args:
        completeness_stats: Dictionary from calculate_completeness_stats().
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    if not completeness_stats:
        # Create empty plot if no data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No metadata available', 
                ha='center', va='center', fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        return fig
    
    # Prepare data
    fields = []
    percentages = []
    available_counts = []
    total_counts = []
    
    # Field display names
    field_names = {
        'year': 'Year',
        'authors': 'Authors',
        'citations': 'Citations',
        'doi': 'DOI',
        'pdf': 'PDF',
        'venue': 'Venue',
        'abstract': 'Abstract'
    }
    
    for field_key in ['year', 'authors', 'citations', 'doi', 'pdf', 'venue', 'abstract']:
        if field_key in completeness_stats:
            stats = completeness_stats[field_key]
            fields.append(field_names.get(field_key, field_key.capitalize()))
            percentages.append(stats['percentage'])
            available_counts.append(stats['available'])
            total_counts.append(stats['total'])
    
    if not fields:
        # Create empty plot if no valid data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No metadata available', 
                ha='center', va='center', fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    y_pos = np.arange(len(fields))
    
    # Color gradient: green (high) to red (low)
    # Create colormap from green to yellow to red
    colors = []
    for pct in percentages:
        if pct >= 80:
            # Green for high completeness
            colors.append('#2A9D8F')
        elif pct >= 60:
            # Yellow-green
            colors.append('#E9C46A')
        elif pct >= 40:
            # Orange
            colors.append('#F77F00')
        else:
            # Red for low completeness
            colors.append('#E63946')
    
    # Create horizontal bars
    bars = ax.barh(y_pos, percentages, alpha=0.8, color=colors, 
                   edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    
    # Add percentage labels on bars
    for i, (bar, pct, avail, total) in enumerate(zip(bars, percentages, available_counts, total_counts)):
        width = bar.get_width()
        # Position label at end of bar
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {pct:.1f}% ({avail}/{total})', 
               va='center', fontsize=FONT_SIZE_LEGEND - 1, fontweight='medium')
    
    # Set y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fields, fontsize=FONT_SIZE_LABELS - 1)
    
    # Labels and title
    ax.set_xlabel('Completeness (%)', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Metadata Field', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    # Set x-axis limits (0-100%)
    ax.set_xlim(0, 105)
    
    # Add grid
    ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
    
    # Add vertical line at 100%
    ax.axvline(100, color='#1B4F72', linestyle=':', linewidth=1.5, alpha=0.5)
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_author_collaboration_network(
    author_pairs: List[Tuple[str, str, int]],
    top_n: int = 50,
    title: str = "Author Collaboration Network"
) -> plt.Figure:
    """Plot author collaboration network graph.
    
    Shows co-authorship relationships as a network graph.
    
    Args:
        author_pairs: List of (author1, author2, count) tuples.
        top_n: Number of top collaborations to display.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    try:
        import networkx as nx
        NETWORKX_AVAILABLE = True
    except ImportError:
        NETWORKX_AVAILABLE = False
        logger.warning("networkx not available, creating simplified collaboration visualization")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    if not NETWORKX_AVAILABLE or not author_pairs:
        # Fallback: simple bar chart
        if author_pairs:
            sorted_pairs = sorted(author_pairs, key=lambda x: x[2], reverse=True)[:top_n]
            pair_labels = [f"{a1} & {a2}" for a1, a2, _ in sorted_pairs]
            counts = [c for _, _, c in sorted_pairs]
            
            y_pos = np.arange(len(pair_labels))
            ax.barh(y_pos, counts, alpha=0.8, color='#2E86AB', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pair_labels, fontsize=FONT_SIZE_LABELS - 3)
            ax.set_xlabel('Number of Collaborations', fontsize=FONT_SIZE_LABELS, fontweight='medium')
            ax.set_ylabel('Author Pairs', fontsize=FONT_SIZE_LABELS, fontweight='medium')
            ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
            ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No collaboration data available', ha='center', va='center',
                   fontsize=FONT_SIZE_LABELS)
            ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        
        plt.tight_layout()
        return fig
    
    # Create network graph
    G = nx.Graph()
    
    # Add edges with weights
    sorted_pairs = sorted(author_pairs, key=lambda x: x[2], reverse=True)[:top_n]
    for author1, author2, count in sorted_pairs:
        if G.has_edge(author1, author2):
            G[author1][author2]['weight'] += count
        else:
            G.add_edge(author1, author2, weight=count)
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'No collaboration network data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Draw network
    node_sizes = [G.degree(node) * 300 for node in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#2E86AB',
                          alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='#1B4F72', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_source_distribution(
    sources: Dict[str, int],
    title: str = "Source Distribution"
) -> plt.Figure:
    """Plot source distribution as pie chart.
    
    Shows the distribution of papers across different sources.
    
    Args:
        sources: Dictionary mapping source names to counts.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    if not sources:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No source data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        return fig
    
    # Sort by count
    sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
    source_names = [s[0] for s in sorted_sources]
    counts = [s[1] for s in sorted_sources]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use colorblind-friendly colormap
    try:
        cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
    except AttributeError:
        cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
    colors = cmap(np.linspace(0, 1, len(source_names)))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        counts, labels=source_names, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': FONT_SIZE_LABELS - 2}
    )
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(FONT_SIZE_LEGEND - 1)
    
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig


def plot_topic_evolution(
    topic_data: Dict[str, List[Tuple[int, float]]],
    title: str = "Topic Evolution Over Time"
) -> plt.Figure:
    """Plot topic evolution over time.
    
    Shows how different topics (keywords/themes) evolve over years.
    
    Args:
        topic_data: Dictionary mapping topic names to list of (year, frequency) tuples.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    if not topic_data:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.text(0.5, 0.5, 'No topic evolution data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        return fig
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use colorblind-friendly colormap
    try:
        cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
    except AttributeError:
        cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
    colors = cmap(np.linspace(0, 1, len(topic_data)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P', '<', '>']
    
    for i, (topic, data) in enumerate(topic_data.items()):
        if data:
            years = [d[0] for d in data]
            frequencies = [d[1] for d in data]
            marker = markers[i % len(markers)]
            
            ax.plot(years, frequencies, marker=marker, label=topic, linewidth=2.5,
                   markersize=8, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=FONT_SIZE_LEGEND,
             framealpha=0.9, title='Topics', title_fontsize=FONT_SIZE_LEGEND + 1)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    
    plt.xticks(fontsize=FONT_SIZE_LABELS - 2)
    plt.yticks(fontsize=FONT_SIZE_LABELS - 2)
    
    plt.tight_layout()
    return fig


def plot_classification_distribution(
    classification_data: Dict[str, Any],
    title: str = "Paper Classification Distribution",
    show_domains: bool = True
) -> plt.Figure:
    """Plot paper classification distribution as pie chart.
    
    Shows the distribution of papers across classification categories
    (Core/Theory/Math, Translation/Tool, Applied) and optionally domains.
    
    Args:
        classification_data: Dictionary from prepare_classification_data() with:
            - "by_category": Dict[str, int] - counts by category
            - "applied_by_domain": Dict[str, int] - counts by domain for applied papers
            - "entries_with_classification": int - total with classification
        title: Plot title.
        show_domains: Whether to create a second subplot for applied domains.
        
    Returns:
        Matplotlib figure.
    """
    by_category = classification_data.get("by_category", {})
    applied_by_domain = classification_data.get("applied_by_domain", {})
    entries_with_classification = classification_data.get("entries_with_classification", 0)
    
    if not by_category or entries_with_classification == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No classification data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        return fig
    
    # Format category names for display
    category_labels = {
        "core_theory_math": "Core/Theory/Math",
        "translation_tool": "Translation/Tool",
        "applied": "Applied"
    }
    
    # Prepare data for pie chart
    sorted_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)
    category_names = [category_labels.get(cat, cat.replace('_', ' ').title()) for cat, _ in sorted_categories]
    counts = [count for _, count in sorted_categories]
    
    # Create figure with subplots
    if show_domains and applied_by_domain:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    else:
        fig, ax1 = plt.subplots(figsize=(12, 10))
        ax2 = None
    
    # Plot category distribution
    try:
        cmap = plt.colormaps.get_cmap(COLORMAP_CATEGORICAL)
    except AttributeError:
        cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
    colors = cmap(np.linspace(0, 1, len(category_names)))
    
    wedges, texts, autotexts = ax1.pie(
        counts, labels=category_names, autopct='%1.1f%%',
        colors=colors, startangle=90, textprops={'fontsize': FONT_SIZE_LABELS - 2}
    )
    
    # Enhance text visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(FONT_SIZE_LEGEND - 1)
    
    ax1.set_title("Classification by Category", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    # Plot domain distribution for applied papers if available
    if ax2 and applied_by_domain:
        sorted_domains = sorted(applied_by_domain.items(), key=lambda x: x[1], reverse=True)
        domain_names = [d[0] for d in sorted_domains]
        domain_counts = [d[1] for d in sorted_domains]
        
        # Limit to top 10 domains for readability
        if len(domain_names) > 10:
            domain_names = domain_names[:10]
            domain_counts = domain_counts[:10]
            domain_names.append("Other")
            domain_counts.append(sum(applied_by_domain.values()) - sum(domain_counts[:10]))
        
        domain_colors = cmap(np.linspace(0, 1, len(domain_names)))
        
        wedges2, texts2, autotexts2 = ax2.pie(
            domain_counts, labels=domain_names, autopct='%1.1f%%',
            colors=domain_colors, startangle=90, textprops={'fontsize': FONT_SIZE_LABELS - 3}
        )
        
        # Enhance text visibility
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(FONT_SIZE_LEGEND - 2)
        
        ax2.set_title("Applied Papers by Domain", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    plt.suptitle(title, fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_embedding_similarity_heatmap(
    similarity_matrix: np.ndarray,
    citation_keys: List[str],
    titles: Optional[List[str]] = None,
    title: str = "Paper Similarity Matrix",
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """Plot similarity heatmap for embeddings with hierarchical clustering dendrogram.
    
    Args:
        similarity_matrix: Similarity matrix, shape (n_documents, n_documents).
        citation_keys: Citation keys for each document.
        titles: Optional titles for each document (for display).
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    import seaborn as sns
    
    # Try to use hierarchical clustering with dendrogram
    try:
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        
        # Convert similarity to distance (1 - similarity)
        # Ensure diagonal is 0 (self-similarity = 1, so distance = 0)
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed distance matrix for linkage
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Compute linkage
        linkage_matrix = hierarchy.linkage(condensed_distances, method='average')
        
        # Create clustermap with dendrograms
        # Note: square=True is not supported when using custom linkage matrices
        g = sns.clustermap(
            similarity_matrix,
            row_linkage=linkage_matrix,
            col_linkage=linkage_matrix,
            xticklabels=citation_keys,
            yticklabels=citation_keys,
            cmap='viridis',
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"label": "Cosine Similarity"},
            figsize=figsize,
            dendrogram_ratio=(0.1, 0.1),  # Space for dendrograms
            cbar_pos=(0.02, 0.8, 0.03, 0.15)  # Position colorbar
        )
        
        # Set title
        g.fig.suptitle(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', y=0.98)
        
        # Improve font sizes
        g.ax_heatmap.set_xlabel("Papers", fontsize=FONT_SIZE_LABELS, fontweight='medium')
        g.ax_heatmap.set_ylabel("Papers", fontsize=FONT_SIZE_LABELS, fontweight='medium')
        
        # Rotate labels for readability
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=FONT_SIZE_TICKS)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=FONT_SIZE_TICKS)
        
        # Improve colorbar font size
        cbar = g.cax
        if cbar is not None:
            cbar.set_ylabel("Cosine Similarity", fontsize=FONT_SIZE_CBAR, fontweight='medium')
            cbar.tick_params(labelsize=FONT_SIZE_TICKS)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return g.fig
        
    except ImportError:
        # Fallback to regular heatmap if scipy is not available
        logger.warning("scipy not available, using unsorted heatmap without dendrogram")
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            xticklabels=citation_keys,
            yticklabels=citation_keys,
            cmap='viridis',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"label": "Cosine Similarity"},
            ax=ax
        )
        
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        ax.set_xlabel("Papers", fontsize=FONT_SIZE_LABELS, fontweight='medium')
        ax.set_ylabel("Papers", fontsize=FONT_SIZE_LABELS, fontweight='medium')
        
        # Rotate labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=FONT_SIZE_TICKS)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=FONT_SIZE_TICKS)
        
        # Improve colorbar font size
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.set_label("Cosine Similarity", fontsize=FONT_SIZE_CBAR, fontweight='medium')
            cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
        
        plt.tight_layout()
        return fig


def plot_embedding_clusters_2d(
    embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    titles: List[str],
    years: Optional[List[Optional[int]]] = None,
    title: str = "Embedding Clusters (2D)",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Plot 2D embedding clusters with coloring.
    
    Args:
        embeddings_2d: 2D reduced embeddings, shape (n_documents, 2).
        cluster_labels: Cluster labels for each document.
        titles: Titles for each document.
        years: Optional years for each document.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Generate colors for clusters
    colors = plt.cm.get_cmap(COLORMAP_CATEGORICAL)(np.linspace(0, 1, n_clusters))
    
    # Plot points by cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        cluster_data = embeddings_2d[mask]
        
        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            c=[colors[i]],
            label=f"Cluster {int(cluster_id)}",
            s=MARKER_SIZE,
            alpha=0.7,
            edgecolors='black',
            linewidths=EDGE_WIDTH
        )
    
    ax.set_xlabel("Dimension 1", fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel("Dimension 2", fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best', title='Clusters', title_fontsize=FONT_SIZE_LEGEND + 1)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.tight_layout()
    return fig


def plot_embedding_clusters_3d(
    embeddings_3d: np.ndarray,
    cluster_labels: np.ndarray,
    titles: List[str],
    years: Optional[List[Optional[int]]] = None,
    title: str = "Embedding Clusters (3D)",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """Plot 3D embedding clusters with coloring.
    
    Args:
        embeddings_3d: 3D reduced embeddings, shape (n_documents, 3).
        cluster_labels: Cluster labels for each document.
        titles: Titles for each document.
        years: Optional years for each document.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Generate colors for clusters
    colors = plt.cm.get_cmap(COLORMAP_CATEGORICAL)(np.linspace(0, 1, n_clusters))
    
    # Plot points by cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        cluster_data = embeddings_3d[mask]
        
        ax.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            cluster_data[:, 2],
            c=[colors[i]],
            label=f"Cluster {int(cluster_id)}",
            s=MARKER_SIZE,
            alpha=0.7,
            edgecolors='black',
            linewidths=EDGE_WIDTH
        )
    
    ax.set_xlabel("Dimension 1", fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel("Dimension 2", fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_zlabel("Dimension 3", fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best', title='Clusters', title_fontsize=FONT_SIZE_LEGEND + 1)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.tight_layout()
    return fig


def plot_semantic_search_results(
    query_text: str,
    results: List[Tuple[str, float, str]],
    title: str = "Semantic Search Results",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot semantic search results as a bar chart.
    
    Args:
        query_text: Original query text.
        results: List of (citation_key, similarity_score, title) tuples.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    if not results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No results found", ha='center', va='center', fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold')
        return fig
    
    citation_keys, similarities, titles = zip(*results)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    bars = ax.barh(range(len(results)), similarities, color=plt.cm.viridis(similarities))
    
    # Set labels
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{key}\n{t[:50]}..." if len(t) > 50 else f"{key}\n{t}" 
                        for key, t in zip(citation_keys, titles)],
                       fontsize=FONT_SIZE_TICKS)
    ax.set_xlabel("Similarity Score", fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(f"{title}\nQuery: {query_text[:100]}...", fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    # Add value labels on bars
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        ax.text(sim + 0.01, i, f"{sim:.3f}", va='center', fontsize=FONT_SIZE_TICKS)
    
    ax.grid(True, alpha=GRID_ALPHA, axis='x')
    ax.set_xlim(0, 1.1)
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.tight_layout()
    return fig


def plot_embedding_quality(
    embeddings: np.ndarray,
    citation_keys: Optional[List[str]] = None,
    title: str = "Embedding Quality Analysis",
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """Plot embedding quality metrics including variance per dimension and distribution.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        citation_keys: Optional citation keys for labeling.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    n_docs, n_dims = embeddings.shape
    
    # 1. Variance per dimension
    ax = axes[0, 0]
    variances = np.var(embeddings, axis=0)
    dims = np.arange(n_dims)
    ax.bar(dims, variances, alpha=0.7, color='#2E86AB', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    ax.set_xlabel('Dimension', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Variance', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title('Variance per Dimension', fontsize=FONT_SIZE_LABELS, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, axis='y', linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    # 2. Distribution of L2 norms
    ax = axes[0, 1]
    norms = np.linalg.norm(embeddings, axis=1)
    ax.hist(norms, bins=50, alpha=0.7, color='#6A4C93', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    ax.axvline(np.mean(norms), color='#E63946', linestyle='--', linewidth=2, label=f'Mean: {np.mean(norms):.3f}')
    ax.axvline(np.median(norms), color='#2A9D8F', linestyle='--', linewidth=2, label=f'Median: {np.median(norms):.3f}')
    ax.set_xlabel('L2 Norm', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Frequency', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title('Distribution of Embedding Norms', fontsize=FONT_SIZE_LABELS, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE_LEGEND, title='Statistics', title_fontsize=FONT_SIZE_LEGEND + 1)
    ax.grid(True, alpha=GRID_ALPHA, axis='y', linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    # 3. Mean per dimension
    ax = axes[1, 0]
    means = np.mean(embeddings, axis=0)
    ax.bar(dims, means, alpha=0.7, color='#F77F00', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Dimension', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Mean Value', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title('Mean Value per Dimension', fontsize=FONT_SIZE_LABELS, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, axis='y', linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    # 4. Standard deviation per dimension
    ax = axes[1, 1]
    stds = np.std(embeddings, axis=0)
    ax.bar(dims, stds, alpha=0.7, color='#E63946', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    ax.set_xlabel('Dimension', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Standard Deviation', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title('Standard Deviation per Dimension', fontsize=FONT_SIZE_LABELS, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, axis='y', linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    fig.suptitle(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_similarity_distribution(
    similarity_matrix: np.ndarray,
    title: str = "Similarity Distribution",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot histogram of similarity values (excluding diagonal).
    
    Args:
        similarity_matrix: Similarity matrix, shape (n_documents, n_documents).
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n = similarity_matrix.shape[0]
    if n > 0:
        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        values = similarity_matrix[mask].flatten()
    else:
        values = []
    
    if len(values) > 0:
        ax.hist(values, bins=50, alpha=0.7, color='#2E86AB', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
        ax.axvline(np.mean(values), color='#E63946', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(values):.3f}')
        ax.axvline(np.median(values), color='#2A9D8F', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(values):.3f}')
        ax.set_xlabel('Cosine Similarity', fontsize=FONT_SIZE_LABELS, fontweight='medium')
        ax.set_ylabel('Frequency', fontsize=FONT_SIZE_LABELS, fontweight='medium')
        ax.legend(fontsize=FONT_SIZE_LEGEND, title='Statistics', title_fontsize=FONT_SIZE_LEGEND + 1)
        ax.grid(True, alpha=GRID_ALPHA, axis='y', linestyle='--')
        ax.tick_params(labelsize=FONT_SIZE_TICKS)
    else:
        ax.text(0.5, 0.5, 'No similarity data available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
    
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_cluster_quality_metrics(
    metrics: Dict[str, any],
    title: str = "Clustering Quality Metrics",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot bar chart of clustering quality metrics.
    
    Args:
        metrics: Dictionary with clustering metrics (silhouette_score, davies_bouldin_index, etc.).
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_names = []
    metric_values = []
    metric_labels = []
    
    if metrics.get("silhouette_score") is not None:
        metric_names.append("Silhouette Score")
        metric_values.append(metrics["silhouette_score"])
        metric_labels.append(f"{metrics['silhouette_score']:.3f}")
    
    if metrics.get("davies_bouldin_index") is not None:
        metric_names.append("Davies-Bouldin Index")
        metric_values.append(metrics["davies_bouldin_index"])
        metric_labels.append(f"{metrics['davies_bouldin_index']:.3f}")
    
    if metrics.get("calinski_harabasz_score") is not None:
        metric_names.append("Calinski-Harabasz Score")
        metric_values.append(metrics["calinski_harabasz_score"])
        metric_labels.append(f"{metrics['calinski_harabasz_score']:.1f}")
    
    if metrics.get("inertia") is not None:
        metric_names.append("Inertia")
        metric_values.append(metrics["inertia"])
        metric_labels.append(f"{metrics['inertia']:.1f}")
    
    if not metric_names:
        ax.text(0.5, 0.5, 'No clustering metrics available', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Normalize values for visualization (0-1 scale)
    normalized_values = []
    for i, (name, value) in enumerate(zip(metric_names, metric_values)):
        if "Silhouette" in name:
            # Silhouette: -1 to 1, normalize to 0-1
            normalized_values.append((value + 1) / 2)
        elif "Davies" in name:
            # Davies-Bouldin: lower is better, invert and normalize
            max_val = max(metric_values) if metric_values else 1.0
            normalized_values.append(1.0 - (value / max_val) if max_val > 0 else 0.5)
        elif "Calinski" in name or "Inertia" in name:
            # Higher/lower is better, normalize by max
            max_val = max(metric_values) if metric_values else 1.0
            normalized_values.append(value / max_val if max_val > 0 else 0.5)
        else:
            normalized_values.append(0.5)
    
    y_pos = np.arange(len(metric_names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
    bars = ax.barh(y_pos, normalized_values, alpha=0.8, color=colors, 
                   edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    
    # Add value labels
    for i, (bar, label) in enumerate(zip(bars, metric_labels)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
               f' {label}', va='center', fontsize=FONT_SIZE_TICKS, fontweight='medium')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names, fontsize=FONT_SIZE_LABELS)
    ax.set_xlabel('Normalized Score', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.tight_layout()
    return fig


def plot_silhouette_analysis(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "Silhouette Analysis",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """Plot silhouette analysis showing cluster quality per sample.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        cluster_labels: Cluster labels for each document.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    try:
        from sklearn.metrics import silhouette_samples
    except ImportError:
        logger.warning("scikit-learn not available for silhouette analysis")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'scikit-learn required for silhouette analysis', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    n_clusters = len(np.unique(cluster_labels))
    if n_clusters < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Need at least 2 clusters for silhouette analysis', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Compute silhouette scores
    silhouette_scores = silhouette_samples(embeddings, cluster_labels)
    unique_labels = np.unique(cluster_labels)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_lower = 10
    colors = plt.cm.get_cmap(COLORMAP_CATEGORICAL)(np.linspace(0, 1, len(unique_labels)))
    
    for i, cluster_id in enumerate(unique_labels):
        # Get silhouette scores for this cluster
        cluster_silhouette_scores = silhouette_scores[cluster_labels == cluster_id]
        cluster_silhouette_scores.sort()
        
        size_cluster = len(cluster_silhouette_scores)
        y_upper = y_lower + size_cluster
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_scores,
                        facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        # Label cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster, str(int(cluster_id)), fontsize=FONT_SIZE_TICKS)
        
        y_lower = y_upper + 10
    
    # Add average silhouette score line
    avg_score = np.mean(silhouette_scores)
    ax.axvline(avg_score, color='red', linestyle='--', linewidth=2,
               label=f'Average Score: {avg_score:.3f}')
    
    ax.set_xlabel('Silhouette Score', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Cluster Label', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.set_xlim(-1, 1)
    ax.legend(fontsize=FONT_SIZE_LEGEND, title='Statistics', title_fontsize=FONT_SIZE_LEGEND + 1)
    ax.grid(True, alpha=GRID_ALPHA, axis='x', linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.tight_layout()
    return fig


def plot_embedding_coverage(
    embeddings: np.ndarray,
    title: str = "Embedding Space Coverage",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Visualize embedding space coverage using 2D projection.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("scikit-learn not available for coverage visualization")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'scikit-learn required for coverage visualization', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    if embeddings.shape[0] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient data for coverage visualization', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with density
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        alpha=0.6, s=MARKER_SIZE * 0.5, c='#2E86AB',
                        edgecolors='black', linewidths=EDGE_WIDTH * 0.5)
    
    # Add convex hull to show coverage
    try:
        from scipy.spatial import ConvexHull
        if len(embeddings_2d) >= 3:
            hull = ConvexHull(embeddings_2d)
            for simplex in hull.simplices:
                ax.plot(embeddings_2d[simplex, 0], embeddings_2d[simplex, 1], 
                       'r--', alpha=0.3, linewidth=1)
    except ImportError:
        pass  # scipy not available, skip convex hull
    
    explained_var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', 
                  fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', 
                  fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.tight_layout()
    return fig


def plot_embedding_outliers(
    embeddings: np.ndarray,
    outlier_indices: List[int],
    citation_keys: Optional[List[str]] = None,
    title: str = "Embedding Outliers",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """Highlight outliers in 2D/3D embedding space.
    
    Args:
        embeddings: Embedding vectors, shape (n_documents, embedding_dim).
        outlier_indices: List of outlier indices.
        citation_keys: Optional citation keys for labeling.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("scikit-learn not available for outlier visualization")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'scikit-learn required for outlier visualization', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    if embeddings.shape[0] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Insufficient data for outlier visualization', ha='center', va='center',
               fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for outliers
    outlier_mask = np.zeros(len(embeddings), dtype=bool)
    outlier_mask[outlier_indices] = True
    
    # Plot normal points
    normal_points = embeddings_2d[~outlier_mask]
    if len(normal_points) > 0:
        ax.scatter(normal_points[:, 0], normal_points[:, 1],
                  alpha=0.6, s=MARKER_SIZE * 0.5, c='#2E86AB',
                  edgecolors='black', linewidths=EDGE_WIDTH * 0.5,
                  label='Normal')
    
    # Plot outliers
    outlier_points = embeddings_2d[outlier_mask]
    if len(outlier_points) > 0:
        ax.scatter(outlier_points[:, 0], outlier_points[:, 1],
                  alpha=0.9, s=MARKER_SIZE, c='#E63946',
                  edgecolors='black', linewidths=EDGE_WIDTH,
                  marker='X', label=f'Outliers ({len(outlier_points)})')
    
    explained_var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', 
                  fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', 
                  fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.legend(fontsize=FONT_SIZE_LEGEND, title='Points', title_fontsize=FONT_SIZE_LEGEND + 1)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.tight_layout()
    return fig


def plot_dimensionality_analysis(
    dimensionality_data: Dict[str, any],
    title: str = "Dimensionality Analysis",
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """Plot explained variance and effective dimensions.
    
    Args:
        dimensionality_data: Dictionary from compute_dimensionality_analysis().
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    explained_var = dimensionality_data.get("explained_variance", [])
    cumulative_var = dimensionality_data.get("cumulative_variance", [])
    
    if not explained_var:
        for ax in axes:
            ax.text(0.5, 0.5, 'No dimensionality data available', ha='center', va='center',
                   fontsize=FONT_SIZE_LABELS)
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    components = np.arange(1, len(explained_var) + 1)
    
    # Plot 1: Explained variance per component
    ax = axes[0]
    ax.bar(components, explained_var, alpha=0.7, color='#2E86AB', 
           edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    ax.set_xlabel('Principal Component', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Explained Variance Ratio', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title('Explained Variance per Component', fontsize=FONT_SIZE_LABELS, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, axis='y', linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    # Plot 2: Cumulative explained variance
    ax = axes[1]
    ax.plot(components, cumulative_var, marker='o', linewidth=2.5, markersize=8,
           color='#E63946', alpha=0.8)
    ax.axhline(0.95, color='#2A9D8F', linestyle='--', linewidth=2, label='95% threshold')
    ax.axhline(0.99, color='#F77F00', linestyle='--', linewidth=2, label='99% threshold')
    
    n_comp_95 = dimensionality_data.get("n_components_95", len(components))
    n_comp_99 = dimensionality_data.get("n_components_99", len(components))
    
    if n_comp_95 <= len(components):
        ax.axvline(n_comp_95, color='#2A9D8F', linestyle=':', linewidth=1.5, alpha=0.7)
    if n_comp_99 <= len(components):
        ax.axvline(n_comp_99, color='#F77F00', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Number of Components', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Cumulative Explained Variance', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title('Cumulative Explained Variance', fontsize=FONT_SIZE_LABELS, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=FONT_SIZE_LEGEND, title='Thresholds', title_fontsize=FONT_SIZE_LEGEND + 1)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    fig.suptitle(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_cluster_size_distribution(
    cluster_labels: np.ndarray,
    title: str = "Cluster Size Distribution",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot distribution of cluster sizes.
    
    Args:
        cluster_labels: Cluster labels for each document.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    # Sort by cluster ID
    sorted_indices = np.argsort(unique_labels)
    cluster_ids = unique_labels[sorted_indices]
    cluster_sizes = counts[sorted_indices]
    
    bars = ax.bar(range(len(cluster_ids)), cluster_sizes, alpha=0.8, 
                 color='#2E86AB', edgecolor='#1B4F72', linewidth=EDGE_WIDTH)
    
    # Add value labels
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(size)}', ha='center', va='bottom',
               fontsize=FONT_SIZE_TICKS, fontweight='medium')
    
    ax.set_xlabel('Cluster ID', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_ylabel('Number of Documents', fontsize=FONT_SIZE_LABELS, fontweight='medium')
    ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([f'Cluster {int(cid)}' for cid in cluster_ids],
                       fontsize=FONT_SIZE_TICKS)
    ax.grid(True, alpha=GRID_ALPHA, axis='y', linestyle='--')
    ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.tight_layout()
    return fig


def plot_similarity_network(
    similarity_matrix: np.ndarray,
    citation_keys: List[str],
    threshold: float = 0.7,
    title: str = "Similarity Network",
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """Plot network graph of high-similarity pairs.
    
    Args:
        similarity_matrix: Similarity matrix, shape (n_documents, n_documents).
        citation_keys: Citation keys for each document.
        threshold: Similarity threshold for edges.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available for similarity network")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'networkx required for similarity network visualization', 
               ha='center', va='center', fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for key in citation_keys:
        G.add_node(key)
    
    # Add edges for high similarity pairs
    n = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                G.add_edge(citation_keys[i], citation_keys[j], weight=similarity)
    
    if len(G.edges()) == 0:
        ax.text(0.5, 0.5, f'No edges found with similarity >= {threshold}', 
               ha='center', va='center', fontsize=FONT_SIZE_LABELS)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Draw network
    node_sizes = [G.degree(node) * 200 + 100 for node in G.nodes()]
    edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#2E86AB',
                          alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, 
                           edge_color='#1B4F72', ax=ax)
    
    # Add labels (only for nodes with high degree to avoid clutter)
    high_degree_nodes = [n for n in G.nodes() if G.degree(n) >= 2]
    labels = {n: n[:20] + '...' if len(n) > 20 else n for n in high_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=FONT_SIZE_TICKS, font_weight='bold', ax=ax)
    
    ax.set_title(f"{title}\n(Threshold: {threshold})", 
                fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def save_plot(fig: plt.Figure, output_path: Path) -> Path:
    """Save plot to file.
    
    Args:
        fig: Matplotlib figure.
        output_path: Output file path.
        
    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved plot to {output_path}")
    return output_path


