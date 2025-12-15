"""Graphical abstract generation for meta-analysis.

Creates composite visualizations that combine all meta-analysis plots
into single-page and multi-page graphical abstracts.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.meta_analysis.aggregator import DataAggregator
from infrastructure.literature.meta_analysis.visualizations import (
    plot_pca_2d,
    plot_pca_3d,
    plot_keyword_frequency,
    plot_metadata_completeness,
    plot_publications_by_year,
    save_plot,
    FONT_SIZE_TITLE,
)

logger = get_logger(__name__)


def create_single_page_abstract(
    aggregator: Optional[DataAggregator] = None,
    keywords: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    format: str = "png"
) -> Path:
    """Create single-page graphical abstract with all visualizations.
    
    Creates a composite figure with subplots arranged in a grid:
    - PCA 2D (top-left)
    - PCA 3D (top-right)
    - Keyword frequency (middle-left)
    - Metadata completeness (middle-right)
    - Publication timeline (bottom-left)
    
    Args:
        aggregator: Optional DataAggregator instance.
        keywords: Optional list of search keywords for title.
        output_path: Optional output path.
        format: Output format (png, svg, pdf).
        
    Returns:
        Path to saved plot.
    """
    if aggregator is None:
        aggregator = DataAggregator()
    
    if output_path is None:
        output_path = Path("data/output/graphical_abstract_single_page." + format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    entries = aggregator.aggregate_library_data()
    temporal_data = aggregator.prepare_temporal_data()
    keyword_data = aggregator.prepare_keyword_data()
    metadata_data = aggregator.prepare_metadata_data()
    corpus = aggregator.prepare_text_corpus()
    
    # Calculate completeness stats
    from infrastructure.literature.meta_analysis.metadata import calculate_completeness_stats
    completeness_stats = calculate_completeness_stats(aggregator)
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3, 
                  left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    # Overall title
    title_text = "Meta-Analysis Graphical Abstract"
    if keywords:
        title_text += f"\nKeywords: {', '.join(keywords)}"
    title_text += f"\nTotal Papers: {len(entries)} | Date: {datetime.now().strftime('%Y-%m-%d')}"
    fig.suptitle(title_text, fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.98)
    
    try:
        # 1. PCA 2D (top-left)
        from infrastructure.literature.meta_analysis.pca import (
            extract_text_features,
            compute_pca,
            cluster_papers,
        )
        logger.debug("Extracting features for graphical abstract PCA 2D...")
        feature_matrix, feature_names, valid_indices = extract_text_features(corpus)
        
        # Filter corpus arrays to match valid indices
        filtered_titles = [corpus.titles[i] for i in valid_indices if i < len(corpus.titles)]
        filtered_years = [corpus.years[i] if i < len(corpus.years) else None for i in valid_indices]
        
        # Validate alignment
        if len(feature_matrix) != len(filtered_titles) or len(feature_matrix) != len(filtered_years):
            raise ValueError(
                f"Graphical abstract array size mismatch: feature_matrix={len(feature_matrix)}, "
                f"titles={len(filtered_titles)}, years={len(filtered_years)}"
            )
        
        pca_data_2d, pca_model_2d = compute_pca(feature_matrix, n_components=2)
        cluster_labels = cluster_papers(pca_data_2d, n_clusters=5)
        
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Prepare year data for coloring
        valid_years = [y for y in filtered_years if y is not None]
        min_year = min(valid_years) if valid_years else 2000
        years_array = [y if y else min_year for y in filtered_years]
        
        # Recreate in subplot with aligned arrays
        scatter = ax1.scatter(
            pca_data_2d[:, 0], pca_data_2d[:, 1],
            c=years_array,
            cmap='plasma', alpha=0.7, s=80, edgecolors='black', linewidth=0.5
        )
        ax1.set_xlabel(f'PC1 ({pca_model_2d.explained_variance_ratio_[0]*100:.1f}%)', 
                      fontsize=10, fontweight='medium')
        ax1.set_ylabel(f'PC2 ({pca_model_2d.explained_variance_ratio_[1]*100:.1f}%)', 
                      fontsize=10, fontweight='medium')
        ax1.set_title("PCA Analysis (2D)", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
    except Exception as e:
        logger.warning(f"Failed to create PCA 2D subplot: {e}")
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, 'PCA 2D\n(Not available)', ha='center', va='center')
        ax1.set_title("PCA Analysis (2D)", fontsize=12, fontweight='bold')
    
    try:
        # 2. PCA 3D (top-right) - simplified 2D projection
        ax2 = fig.add_subplot(gs[0, 1])
        pca_data_3d, pca_model_3d = compute_pca(feature_matrix, n_components=3)
        scatter = ax2.scatter(
            pca_data_3d[:, 0], pca_data_3d[:, 1],
            c=years_array,  # Use same aligned years array
            cmap='plasma', alpha=0.7, s=80, edgecolors='black', linewidth=0.5
        )
        ax2.set_xlabel(f'PC1 ({pca_model_3d.explained_variance_ratio_[0]*100:.1f}%)', 
                      fontsize=10, fontweight='medium')
        ax2.set_ylabel(f'PC2 ({pca_model_3d.explained_variance_ratio_[1]*100:.1f}%)', 
                      fontsize=10, fontweight='medium')
        ax2.set_title("PCA Analysis (3D Projection)", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
    except Exception as e:
        logger.warning(f"Failed to create PCA 3D subplot: {e}")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, 'PCA 3D\n(Not available)', ha='center', va='center')
        ax2.set_title("PCA Analysis (3D)", fontsize=12, fontweight='bold')
    
    try:
        # 3. Keyword frequency (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        top_keywords = sorted(
            keyword_data.keyword_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15]
        keywords_list = [k for k, _ in top_keywords]
        counts_list = [c for _, c in top_keywords]
        
        y_pos = np.arange(len(keywords_list))
        ax3.barh(y_pos, counts_list, alpha=0.8, color='#2E86AB', edgecolor='#1B4F72', linewidth=0.5)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(keywords_list, fontsize=8)
        ax3.set_xlabel('Frequency', fontsize=10, fontweight='medium')
        ax3.set_ylabel('Keywords', fontsize=10, fontweight='medium')
        ax3.set_title("Top Keywords", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x', linestyle='--')
    except Exception as e:
        logger.warning(f"Failed to create keyword frequency subplot: {e}")
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.text(0.5, 0.5, 'Keywords\n(Not available)', ha='center', va='center')
        ax3.set_title("Top Keywords", fontsize=12, fontweight='bold')
    
    try:
        # 4. Metadata completeness (middle-right)
        ax4 = fig.add_subplot(gs[1, 1])
        if completeness_stats:
            fields = []
            percentages = []
            for field_key in ['year', 'authors', 'citations', 'doi', 'pdf', 'venue', 'abstract']:
                if field_key in completeness_stats:
                    field_names = {
                        'year': 'Year', 'authors': 'Authors', 'citations': 'Citations',
                        'doi': 'DOI', 'pdf': 'PDF', 'venue': 'Venue', 'abstract': 'Abstract'
                    }
                    fields.append(field_names.get(field_key, field_key.capitalize()))
                    percentages.append(completeness_stats[field_key]['percentage'])
            
            y_pos = np.arange(len(fields))
            colors = ['#2A9D8F' if p >= 80 else '#E9C46A' if p >= 60 else '#F77F00' if p >= 40 else '#E63946' 
                     for p in percentages]
            ax4.barh(y_pos, percentages, alpha=0.8, color=colors, edgecolor='#1B4F72', linewidth=0.5)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(fields, fontsize=9)
            ax4.set_xlabel('Completeness (%)', fontsize=10, fontweight='medium')
            ax4.set_ylabel('Metadata Field', fontsize=10, fontweight='medium')
            ax4.set_title("Metadata Completeness", fontsize=12, fontweight='bold')
            ax4.set_xlim(0, 105)
            ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
        else:
            ax4.text(0.5, 0.5, 'Metadata\n(Not available)', ha='center', va='center')
            ax4.set_title("Metadata Completeness", fontsize=12, fontweight='bold')
    except Exception as e:
        logger.warning(f"Failed to create metadata completeness subplot: {e}")
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.text(0.5, 0.5, 'Metadata\n(Not available)', ha='center', va='center')
        ax4.set_title("Metadata Completeness", fontsize=12, fontweight='bold')
    
    try:
        # 5. Publication timeline (bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        if temporal_data.years:
            ax5.bar(temporal_data.years, temporal_data.counts, alpha=0.8, 
                   color='#2E86AB', edgecolor='#1B4F72', linewidth=0.5)
            ax5.set_xlabel('Year', fontsize=10, fontweight='medium')
            ax5.set_ylabel('Publications', fontsize=10, fontweight='medium')
            ax5.set_title("Publications by Year", fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, linestyle='--')
            if len(temporal_data.years) > 10:
                plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax5.text(0.5, 0.5, 'Timeline\n(Not available)', ha='center', va='center')
            ax5.set_title("Publications by Year", fontsize=12, fontweight='bold')
    except Exception as e:
        logger.warning(f"Failed to create publication timeline subplot: {e}")
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.text(0.5, 0.5, 'Timeline\n(Not available)', ha='center', va='center')
        ax5.set_title("Publications by Year", fontsize=12, fontweight='bold')
    
    return save_plot(fig, output_path)


def create_multi_page_abstract(
    aggregator: Optional[DataAggregator] = None,
    keywords: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    format: str = "pdf"
) -> Path:
    """Create multi-page graphical abstract PDF.
    
    Creates a PDF with one visualization per page.
    
    Args:
        aggregator: Optional DataAggregator instance.
        keywords: Optional list of search keywords for title.
        output_path: Optional output path.
        format: Output format (pdf recommended).
        
    Returns:
        Path to saved PDF.
    """
    if aggregator is None:
        aggregator = DataAggregator()
    
    if output_path is None:
        output_path = Path("data/output/graphical_abstract_multi_page." + format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Prepare data
    entries = aggregator.aggregate_library_data()
    temporal_data = aggregator.prepare_temporal_data()
    keyword_data = aggregator.prepare_keyword_data()
    metadata_data = aggregator.prepare_metadata_data()
    corpus = aggregator.prepare_text_corpus()
    
    from infrastructure.literature.meta_analysis.metadata import calculate_completeness_stats
    completeness_stats = calculate_completeness_stats(aggregator)
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        title_text = "Meta-Analysis Graphical Abstract"
        if keywords:
            title_text += f"\n\nKeywords: {', '.join(keywords)}"
        title_text += f"\n\nTotal Papers: {len(entries)}"
        title_text += f"\nDate: {datetime.now().strftime('%Y-%m-%d')}"
        ax.text(0.5, 0.5, title_text, ha='center', va='center',
               fontsize=FONT_SIZE_TITLE + 4, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Individual visualizations
        try:
            from infrastructure.literature.meta_analysis.pca import (
                create_pca_2d_plot,
                create_pca_3d_plot,
            )
            from infrastructure.literature.meta_analysis.keywords import create_keyword_frequency_plot
            from infrastructure.literature.meta_analysis.metadata import create_metadata_completeness_plot
            from infrastructure.literature.meta_analysis.temporal import create_publication_timeline_plot
            
            if not PIL_AVAILABLE:
                logger.warning("PIL/Pillow not available, skipping multi-page abstract image pages")
                return output_path
            
            # PCA 2D
            pca_2d_path = create_pca_2d_plot(aggregator=aggregator, n_clusters=5, format="png")
            img = Image.open(pca_2d_path)
            pdf_fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(pdf_fig, bbox_inches='tight')
            plt.close(pdf_fig)
            
            # PCA 3D
            pca_3d_path = create_pca_3d_plot(aggregator=aggregator, n_clusters=5, format="png")
            img = Image.open(pca_3d_path)
            pdf_fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(pdf_fig, bbox_inches='tight')
            plt.close(pdf_fig)
            
            # Keyword frequency
            keyword_data = aggregator.prepare_keyword_data()
            keyword_path = create_keyword_frequency_plot(keyword_data, format="png")
            img = Image.open(keyword_path)
            pdf_fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(pdf_fig, bbox_inches='tight')
            plt.close(pdf_fig)
            
            # Metadata completeness
            completeness_path = create_metadata_completeness_plot(aggregator=aggregator, format="png")
            img = Image.open(completeness_path)
            pdf_fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(pdf_fig, bbox_inches='tight')
            plt.close(pdf_fig)
            
            # Publication timeline
            timeline_path = create_publication_timeline_plot(aggregator=aggregator, format="png")
            img = Image.open(timeline_path)
            pdf_fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(pdf_fig, bbox_inches='tight')
            plt.close(pdf_fig)
            
        except Exception as e:
            logger.warning(f"Failed to add some pages to multi-page abstract: {e}")
            if not PIL_AVAILABLE:
                logger.warning("PIL/Pillow not available, cannot create multi-page abstract from images")
    
    logger.info(f"Created multi-page graphical abstract: {output_path}")
    return output_path


def create_graphical_abstract(
    aggregator: Optional[DataAggregator] = None,
    keywords: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    format: str = "png"
) -> Path:
    """Create single-page abstract with all visualizations including loadings.
    
    Similar to create_single_page_abstract but includes PCA loadings visualizations.
    
    Args:
        aggregator: Optional DataAggregator instance.
        keywords: Optional list of search keywords for title.
        output_path: Optional output path.
        format: Output format (png, svg, pdf).
        
    Returns:
        Path to saved plot.
    """
    # For now, use the single-page abstract
    # Can be extended to include loadings visualizations in a larger grid
    return create_single_page_abstract(aggregator, keywords, output_path, format)


def create_composite_panel(
    aggregator: Optional[DataAggregator] = None,
    keywords: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    format: str = "png",
    max_panels: int = 20
) -> Path:
    """Create auto-sized composite panel with all available visualizations.
    
    Automatically determines optimal grid size based on available visualizations
    and creates a large composite image (e.g., 5x4, 6x4, etc.) with all plots.
    
    Args:
        aggregator: Optional DataAggregator instance.
        keywords: Optional list of search keywords for title.
        output_path: Optional output path.
        format: Output format (png, svg, pdf).
        max_panels: Maximum number of panels to include.
        
    Returns:
        Path to saved plot.
    """
    if aggregator is None:
        aggregator = DataAggregator()
    
    if output_path is None:
        output_path = Path("data/output/composite_panel." + format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    entries = aggregator.aggregate_library_data()
    temporal_data = aggregator.prepare_temporal_data()
    keyword_data = aggregator.prepare_keyword_data()
    metadata_data = aggregator.prepare_metadata_data()
    corpus = aggregator.prepare_text_corpus()
    
    from infrastructure.literature.meta_analysis.metadata import calculate_completeness_stats
    completeness_stats = calculate_completeness_stats(aggregator)
    
    # Define available visualizations with their data requirements
    available_viz = []
    
    # 1. Publications by year
    if temporal_data.years:
        available_viz.append(("Publications by Year", "temporal"))
    
    # 2. Venue distribution
    if metadata_data.venues:
        available_viz.append(("Venue Distribution", "metadata"))
    
    # 4. Author contributions
    if metadata_data.authors:
        available_viz.append(("Author Contributions", "metadata"))
    
    # 5. Keyword frequency
    if keyword_data.keyword_counts:
        available_viz.append(("Keyword Frequency", "keyword"))
    
    # 6. Keyword evolution
    if keyword_data.keyword_frequency_over_time:
        available_viz.append(("Keyword Evolution", "keyword"))
    
    # 7. Metadata completeness
    if completeness_stats:
        available_viz.append(("Metadata Completeness", "metadata"))
    
    # 8. PCA 2D
    try:
        from infrastructure.literature.meta_analysis.pca import (
            extract_text_features,
            compute_pca,
            cluster_papers,
        )
        if corpus.texts or corpus.abstracts:
            feature_matrix, feature_names, valid_indices = extract_text_features(corpus)
            if len(feature_matrix) >= 2:
                available_viz.append(("PCA 2D", "pca"))
    except Exception as e:
        logger.debug(f"PCA 2D not available: {e}")
    
    # 9. PCA 3D
    try:
        if corpus.texts or corpus.abstracts:
            feature_matrix, feature_names, valid_indices = extract_text_features(corpus)
            if len(feature_matrix) >= 3:
                available_viz.append(("PCA 3D", "pca"))
    except Exception:
        pass
    
    # 10. Citation vs Year (advanced)
    if any(e.year and e.citation_count is not None for e in entries):
        available_viz.append(("Citation vs Year", "advanced"))
    
    # 11. Venue trends (advanced)
    if any(e.venue and e.year for e in entries):
        available_viz.append(("Venue Trends", "advanced"))
    
    # 12. Author productivity (advanced)
    if metadata_data.authors:
        available_viz.append(("Author Productivity", "advanced"))
    
    # 13. Topic distribution (advanced)
    if keyword_data.keyword_counts:
        available_viz.append(("Topic Distribution", "advanced"))
    
    # 14. Publication heatmap (advanced)
    if any(e.venue and e.year for e in entries):
        available_viz.append(("Publication Heatmap", "advanced"))
    
    # Limit to max_panels
    available_viz = available_viz[:max_panels]
    
    if not available_viz:
        logger.warning("No visualizations available for composite panel")
        # Create empty figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No visualization data available', ha='center', va='center',
               fontsize=FONT_SIZE_TITLE)
        ax.axis('off')
        return save_plot(fig, output_path)
    
    # Calculate optimal grid size
    n_viz = len(available_viz)
    # Try to make it roughly square or slightly wider
    cols = int(np.ceil(np.sqrt(n_viz * 1.2)))  # Slightly wider
    rows = int(np.ceil(n_viz / cols))
    
    # Ensure reasonable limits
    cols = min(cols, 6)
    rows = min(rows, 5)
    
    logger.info(f"Creating composite panel: {n_viz} visualizations in {rows}x{cols} grid")
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(cols * 4, rows * 3.5))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.4,
                  left=0.05, right=0.95, top=0.93, bottom=0.05)
    
    # Overall title
    title_text = "Meta-Analysis Composite Panel"
    if keywords:
        title_text += f"\nKeywords: {', '.join(keywords)}"
    title_text += f"\nTotal Papers: {len(entries)} | Date: {datetime.now().strftime('%Y-%m-%d')}"
    fig.suptitle(title_text, fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.98)
    
    # Import additional visualizations
    from infrastructure.literature.meta_analysis.additional_visualizations import (
        plot_citation_vs_year,
        plot_venue_trends,
        plot_author_productivity,
        plot_topic_distribution,
        plot_publication_heatmap,
    )
    
    # Generate each visualization
    for idx, (viz_name, viz_type) in enumerate(available_viz):
        row = idx // cols
        col = idx % cols
        
        if row >= rows:
            break
        
        ax = fig.add_subplot(gs[row, col])
        
        try:
            if viz_name == "Publications by Year":
                ax.bar(temporal_data.years, temporal_data.counts, alpha=0.8,
                      color='#2E86AB', edgecolor='#1B4F72', linewidth=0.5)
                ax.set_xlabel('Year', fontsize=9)
                ax.set_ylabel('Publications', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
            
            elif viz_name == "Venue Distribution":
                top_venues = sorted(metadata_data.venues.items(), key=lambda x: x[1], reverse=True)[:10]
                venues = [v[:30] for v, _ in top_venues]
                counts = [c for _, c in top_venues]
                y_pos = np.arange(len(venues))
                ax.barh(y_pos, counts, alpha=0.8, color='#2E86AB', edgecolor='#1B4F72', linewidth=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(venues, fontsize=8)
                ax.set_xlabel('Count', fontsize=9)
                ax.set_ylabel('Venue', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            
            elif viz_name == "Author Contributions":
                top_authors = sorted(metadata_data.authors.items(), key=lambda x: x[1], reverse=True)[:15]
                authors = [a[:25] for a, _ in top_authors]
                counts = [c for _, c in top_authors]
                y_pos = np.arange(len(authors))
                ax.barh(y_pos, counts, alpha=0.8, color='#2A9D8F', edgecolor='#1B4F72', linewidth=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(authors, fontsize=7)
                ax.set_xlabel('Publications', fontsize=9)
                ax.set_ylabel('Author', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            
            elif viz_name == "Keyword Frequency":
                top_keywords = sorted(keyword_data.keyword_counts.items(),
                                    key=lambda x: x[1], reverse=True)[:15]
                keywords = [k for k, _ in top_keywords]
                counts = [c for _, c in top_keywords]
                y_pos = np.arange(len(keywords))
                ax.barh(y_pos, counts, alpha=0.8, color='#2E86AB', edgecolor='#1B4F72', linewidth=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(keywords, fontsize=7)
                ax.set_xlabel('Frequency', fontsize=9)
                ax.set_ylabel('Keyword', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            
            elif viz_name == "Metadata Completeness":
                if completeness_stats:
                    fields = []
                    percentages = []
                    for field_key in ['year', 'authors', 'citations', 'doi', 'pdf', 'venue', 'abstract']:
                        if field_key in completeness_stats:
                            field_names = {
                                'year': 'Year', 'authors': 'Authors', 'citations': 'Citations',
                                'doi': 'DOI', 'pdf': 'PDF', 'venue': 'Venue', 'abstract': 'Abstract'
                            }
                            fields.append(field_names.get(field_key, field_key.capitalize()))
                            percentages.append(completeness_stats[field_key]['percentage'])
                    
                    y_pos = np.arange(len(fields))
                    colors = ['#2A9D8F' if p >= 80 else '#E9C46A' if p >= 60 else '#F77F00' if p >= 40 else '#E63946'
                             for p in percentages]
                    ax.barh(y_pos, percentages, alpha=0.8, color=colors, edgecolor='#1B4F72', linewidth=0.5)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(fields, fontsize=8)
                    ax.set_xlabel('Completeness (%)', fontsize=9)
                    ax.set_ylabel('Field', fontsize=9)
                    ax.set_title(viz_name, fontsize=11, fontweight='bold')
                    ax.set_xlim(0, 105)
                    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            
            elif viz_name == "PCA 2D":
                feature_matrix, feature_names, valid_indices = extract_text_features(corpus)
                filtered_years = [corpus.years[i] if i < len(corpus.years) else None for i in valid_indices]
                pca_data, pca_model = compute_pca(feature_matrix, n_components=2)
                cluster_labels = cluster_papers(pca_data, n_clusters=5) if len(pca_data) >= 5 else None
                
                valid_years = [y for y in filtered_years if y is not None]
                min_year = min(valid_years) if valid_years else 2000
                years_array = np.array([y if y else min_year for y in filtered_years])
                
                scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=years_array,
                                   cmap='plasma', alpha=0.7, s=40, edgecolors='black', linewidth=0.3)
                ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=8)
                ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=8)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
            
            elif viz_name == "Citation vs Year":
                fig_viz = plot_citation_vs_year(entries)
                # Extract axes from figure and copy to subplot
                ax_viz = fig_viz.axes[0]
                for line in ax_viz.lines:
                    ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                           linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
                           marker=line.get_marker(), markersize=line.get_markersize() * 0.5)
                for collection in ax_viz.collections:
                    if hasattr(collection, 'get_offsets'):
                        offsets = collection.get_offsets()
                        colors = collection.get_array()
                        ax.scatter(offsets[:, 0], offsets[:, 1], c=colors, s=30, alpha=0.6)
                ax.set_xlabel('Year', fontsize=9)
                ax.set_ylabel('Citations', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.close(fig_viz)
            
            elif viz_name == "Venue Trends":
                fig_viz = plot_venue_trends(entries, top_n_venues=5)
                ax_viz = fig_viz.axes[0]
                for line in ax_viz.lines:
                    ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                           linestyle=line.get_linestyle(), linewidth=line.get_linewidth() * 0.7,
                           marker=line.get_marker(), markersize=line.get_markersize() * 0.7,
                           label=line.get_label()[:20])
                ax.set_xlabel('Year', fontsize=9)
                ax.set_ylabel('Publications', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.legend(fontsize=6, framealpha=0.9, ncol=1)
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.close(fig_viz)
            
            elif viz_name == "Author Productivity":
                fig_viz = plot_author_productivity(entries, top_n_authors=10)
                ax_viz = fig_viz.axes[0]
                for patch in ax_viz.patches:
                    rect = patch.get_bbox()
                    ax.barh(rect.y0, rect.width, height=rect.height, alpha=0.8,
                           color=patch.get_facecolor(), edgecolor=patch.get_edgecolor())
                ax.set_yticks(ax_viz.get_yticks())
                ax.set_yticklabels([label.get_text()[:20] for label in ax_viz.get_yticklabels()], fontsize=7)
                ax.set_xlabel('Publications', fontsize=9)
                ax.set_ylabel('Author', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                plt.close(fig_viz)
            
            elif viz_name == "Topic Distribution":
                fig_viz = plot_topic_distribution(entries)
                ax_viz = fig_viz.axes[0]
                for patch in ax_viz.patches:
                    rect = patch.get_bbox()
                    ax.barh(rect.y0, rect.width, height=rect.height, alpha=0.8,
                           color=patch.get_facecolor(), edgecolor=patch.get_edgecolor())
                ax.set_yticks(ax_viz.get_yticks())
                ax.set_yticklabels([label.get_text()[:20] for label in ax_viz.get_yticklabels()], fontsize=7)
                ax.set_xlabel('Frequency', fontsize=9)
                ax.set_ylabel('Topic', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                plt.close(fig_viz)
            
            elif viz_name == "Publication Heatmap":
                fig_viz = plot_publication_heatmap(entries, top_n_venues=10)
                ax_viz = fig_viz.axes[0]
                im = ax_viz.images[0]
                ax.imshow(im.get_array(), aspect='auto', cmap=im.get_cmap(),
                         extent=im.get_extent(), alpha=im.get_alpha())
                ax.set_xticks(ax_viz.get_xticks())
                ax.set_xticklabels([label.get_text() for label in ax_viz.get_xticklabels()], fontsize=7, rotation=45)
                ax.set_yticks(ax_viz.get_yticks())
                ax.set_yticklabels([label.get_text()[:25] for label in ax_viz.get_yticklabels()], fontsize=7)
                ax.set_xlabel('Year', fontsize=9)
                ax.set_ylabel('Venue', fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                plt.close(fig_viz)
            
            else:
                ax.text(0.5, 0.5, f'{viz_name}\n(Not available)', ha='center', va='center',
                       fontsize=9)
                ax.set_title(viz_name, fontsize=11, fontweight='bold')
                ax.axis('off')
        
        except Exception as e:
            logger.warning(f"Failed to create {viz_name} subplot: {e}")
            ax.text(0.5, 0.5, f'{viz_name}\n(Error)', ha='center', va='center',
                   fontsize=9, color='red')
            ax.set_title(viz_name, fontsize=11, fontweight='bold')
            ax.axis('off')
    
    # Fill remaining empty subplots
    for idx in range(len(available_viz), rows * cols):
        row = idx // cols
        col = idx % cols
        if row < rows:
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
    
    return save_plot(fig, output_path)


