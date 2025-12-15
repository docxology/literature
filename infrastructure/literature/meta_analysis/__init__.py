"""Meta-analysis and visualization module.

Provides tools for analyzing publication trends, keyword evolution,
metadata visualization, and PCA analysis of paper texts.
"""
from infrastructure.literature.meta_analysis.aggregator import (
    DataAggregator,
    TemporalData,
    KeywordData,
    MetadataData,
    TextCorpus,
)
from infrastructure.literature.meta_analysis.temporal import (
    get_publication_trends,
    filter_by_year_range,
    analyze_publication_rate,
    create_publication_timeline_plot,
)
from infrastructure.literature.meta_analysis.keywords import (
    extract_keywords_over_time,
    detect_emerging_keywords,
    create_keyword_frequency_plot,
    create_keyword_evolution_plot,
)
from infrastructure.literature.meta_analysis.metadata import (
    create_venue_distribution_plot,
    create_author_contributions_plot,
    create_metadata_completeness_plot,
    create_classification_distribution_plot,
    calculate_completeness_stats,
    get_metadata_summary,
)
from infrastructure.literature.meta_analysis.pca import (
    extract_text_features,
    compute_pca,
    cluster_papers,
    create_pca_2d_plot,
    create_pca_3d_plot,
    export_pca_loadings,
)
from infrastructure.literature.meta_analysis.pca_loadings import (
    extract_pca_loadings,
    get_top_words_per_component,
    export_loadings_csv,
    export_loadings_json,
    export_loadings_summary_markdown,
    export_word_importance_rankings,
    export_all_loadings,
    create_loadings_visualizations,
)
from infrastructure.literature.meta_analysis.summary import (
    generate_summary_data,
    generate_text_summary,
    export_summary_json,
    generate_all_summaries,
)
from infrastructure.literature.meta_analysis.graphical_abstract import (
    create_single_page_abstract,
    create_multi_page_abstract,
    create_graphical_abstract,
    create_composite_panel,
)
from infrastructure.literature.meta_analysis.visualizations import (
    plot_pca_loadings_heatmap,
    plot_pca_loadings_barplot,
    plot_pca_biplot,
    plot_pca_word_vectors,
    plot_metadata_completeness,
    plot_publications_by_year,
    plot_pca_2d,
    plot_pca_3d,
    plot_embedding_similarity_heatmap,
    plot_embedding_clusters_2d,
    plot_embedding_clusters_3d,
    plot_semantic_search_results,
    plot_embedding_quality,
    plot_similarity_distribution,
    plot_cluster_quality_metrics,
    plot_silhouette_analysis,
    plot_embedding_coverage,
    plot_embedding_outliers,
    plot_dimensionality_analysis,
    plot_cluster_size_distribution,
    plot_similarity_network,
)
from infrastructure.literature.meta_analysis.embeddings import (
    EmbeddingData,
    SimilarityResults,
    generate_document_embeddings,
    compute_similarity_matrix,
    cluster_embeddings,
    find_similar_papers,
    reduce_dimensions,
    export_embeddings,
    export_similarity_matrix,
    export_clusters,
    export_embedding_statistics,
    export_validation_report,
    export_clustering_metrics,
)
from infrastructure.literature.meta_analysis.embedding_validation import (
    validate_embedding_quality,
    validate_embedding_completeness,
    validate_embedding_dimensions,
    validate_similarity_matrix,
    detect_embedding_outliers,
    validate_all,
)
from infrastructure.literature.meta_analysis.embedding_statistics import (
    compute_embedding_statistics,
    compute_similarity_statistics,
    compute_clustering_metrics,
    compute_dimensionality_analysis,
    compute_outlier_statistics,
    compute_all_statistics,
)
from infrastructure.literature.meta_analysis.additional_visualizations import (
    plot_citation_vs_year,
    plot_venue_trends,
    plot_author_productivity,
    plot_citation_network,
    plot_topic_distribution,
    plot_word_cloud,
    plot_correlation_matrix,
    plot_publication_heatmap,
    create_citation_vs_year_plot,
    create_venue_trends_plot,
    create_author_productivity_plot,
    create_publication_heatmap_plot,
)

__all__ = [
    # Aggregator
    "DataAggregator",
    "TemporalData",
    "KeywordData",
    "MetadataData",
    "TextCorpus",
    # Temporal analysis
    "get_publication_trends",
    "filter_by_year_range",
    "analyze_publication_rate",
    "create_publication_timeline_plot",
    # Keyword analysis
    "extract_keywords_over_time",
    "detect_emerging_keywords",
    "create_keyword_frequency_plot",
    "create_keyword_evolution_plot",
    # Metadata analysis
    "create_venue_distribution_plot",
    "create_author_contributions_plot",
    "create_metadata_completeness_plot",
    "create_classification_distribution_plot",
    "calculate_completeness_stats",
    "get_metadata_summary",
    # PCA analysis
    "extract_text_features",
    "compute_pca",
    "cluster_papers",
    "create_pca_2d_plot",
    "create_pca_3d_plot",
    "export_pca_loadings",
    # PCA loadings
    "extract_pca_loadings",
    "get_top_words_per_component",
    "export_loadings_csv",
    "export_loadings_json",
    "export_loadings_summary_markdown",
    "export_word_importance_rankings",
    "export_all_loadings",
    "create_loadings_visualizations",
    # Summary reports
    "generate_summary_data",
    "generate_text_summary",
    "export_summary_json",
    "generate_all_summaries",
    # Graphical abstracts
    "create_single_page_abstract",
    "create_multi_page_abstract",
    "create_graphical_abstract",
    "create_composite_panel",
    # Visualization functions
    "plot_pca_loadings_heatmap",
    "plot_pca_loadings_barplot",
    "plot_pca_biplot",
    "plot_pca_word_vectors",
    "plot_metadata_completeness",
    "plot_publications_by_year",
    "plot_pca_2d",
    "plot_pca_3d",
    # Additional visualizations
    "plot_citation_vs_year",
    "plot_venue_trends",
    "plot_author_productivity",
    "plot_citation_network",
    "plot_topic_distribution",
    "plot_word_cloud",
    "plot_correlation_matrix",
    "plot_publication_heatmap",
    "create_citation_vs_year_plot",
    "create_venue_trends_plot",
    "create_author_productivity_plot",
    "create_publication_heatmap_plot",
    # Embedding analysis
    "EmbeddingData",
    "SimilarityResults",
    "generate_document_embeddings",
    "compute_similarity_matrix",
    "cluster_embeddings",
    "find_similar_papers",
    "reduce_dimensions",
    "export_embeddings",
    "export_similarity_matrix",
    "export_clusters",
    # Embedding visualizations
    "plot_embedding_similarity_heatmap",
    "plot_embedding_clusters_2d",
    "plot_embedding_clusters_3d",
    "plot_semantic_search_results",
    "plot_embedding_quality",
    "plot_similarity_distribution",
    "plot_cluster_quality_metrics",
    "plot_silhouette_analysis",
    "plot_embedding_coverage",
    "plot_embedding_outliers",
    "plot_dimensionality_analysis",
    "plot_cluster_size_distribution",
    "plot_similarity_network",
    # Embedding validation
    "validate_embedding_quality",
    "validate_embedding_completeness",
    "validate_embedding_dimensions",
    "validate_similarity_matrix",
    "detect_embedding_outliers",
    "validate_all",
    # Embedding statistics
    "compute_embedding_statistics",
    "compute_similarity_statistics",
    "compute_clustering_metrics",
    "compute_dimensionality_analysis",
    "compute_outlier_statistics",
    "compute_all_statistics",
    # Embedding exports
    "export_embedding_statistics",
    "export_validation_report",
    "export_clustering_metrics",
]


