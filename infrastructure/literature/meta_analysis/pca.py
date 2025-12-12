"""PCA analysis of extracted texts.

Performs principal component analysis on paper texts to identify
clusters and relationships between papers.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

from infrastructure.core.logging_utils import get_logger
from infrastructure.literature.meta_analysis.aggregator import (
    DataAggregator,
    TextCorpus,
)
from infrastructure.literature.meta_analysis.visualizations import (
    plot_pca_2d,
    plot_pca_3d,
    save_plot,
)

logger = get_logger(__name__)

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. PCA functionality will be limited.")


def extract_text_features(
    corpus: TextCorpus,
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.95
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract text features using TF-IDF.
    
    Args:
        corpus: Text corpus from aggregator.
        max_features: Maximum number of features.
        min_df: Minimum document frequency.
        max_df: Maximum document frequency.
        
    Returns:
        Tuple of (feature_matrix, feature_names, valid_indices).
        feature_matrix: (n_valid_docs, n_features) feature matrix.
        feature_names: List of feature names.
        valid_indices: List of original corpus indices that were kept (for alignment).
        
    Raises:
        ValueError: If corpus is empty or has insufficient data.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for text feature extraction")
    
    # Validate corpus
    if not corpus.texts and not corpus.abstracts:
        raise ValueError("Corpus is empty: no texts or abstracts available")
    
    total_docs = len(corpus.texts) if corpus.texts else len(corpus.abstracts)
    if total_docs < 2:
        raise ValueError(f"Insufficient documents for feature extraction: need at least 2, got {total_docs}")
    
    logger.debug(f"Starting feature extraction from {total_docs} documents")
    
    # Combine titles and abstracts for better representation
    texts = []
    valid_indices = []
    
    for idx in range(total_docs):
        title = corpus.titles[idx] if idx < len(corpus.titles) else ""
        abstract = corpus.abstracts[idx] if idx < len(corpus.abstracts) else ""
        combined = f"{title} {abstract}".strip()
        
        if combined:  # Only keep non-empty texts
            texts.append(combined)
            valid_indices.append(idx)
    
    logger.debug(f"Filtered to {len(texts)} non-empty texts from {total_docs} total documents")
    
    if len(texts) < 2:
        raise ValueError(f"Insufficient non-empty texts for feature extraction: need at least 2, got {len(texts)}")
    
    # Adjust min_df if we have very few documents
    adjusted_min_df = min(min_df, max(1, len(texts) - 1))
    logger.debug(f"Using min_df={adjusted_min_df} (requested {min_df}, adjusted for {len(texts)} documents)")
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=adjusted_min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        feature_matrix = vectorizer.fit_transform(texts).toarray()
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        if feature_matrix.shape[0] == 0 or feature_matrix.shape[1] == 0:
            raise ValueError("Feature extraction produced empty matrix")
        
        logger.info(f"Extracted {feature_matrix.shape[1]} features from {len(texts)} documents "
                   f"(filtered from {total_docs} total, {total_docs - len(texts)} empty removed)")
        
        return feature_matrix, feature_names, valid_indices
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        import traceback
        logger.debug(f"Feature extraction traceback: {traceback.format_exc()}")
        raise ValueError(f"Failed to extract features: {e}") from e


def compute_pca(
    feature_matrix: np.ndarray,
    n_components: int = 2
) -> Tuple[np.ndarray, PCA]:
    """Compute PCA on feature matrix.
    
    Args:
        feature_matrix: Feature matrix from extract_text_features.
        n_components: Number of principal components.
        
    Returns:
        Tuple of (transformed_data, pca_model).
        
    Raises:
        ValueError: If feature matrix is empty or has insufficient dimensions.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for PCA")
    
    # Validate input
    if feature_matrix.size == 0:
        raise ValueError("Feature matrix is empty")
    
    n_samples, n_features = feature_matrix.shape
    
    if n_samples < 2:
        raise ValueError(f"Insufficient samples for PCA: need at least 2, got {n_samples}")
    
    if n_features < n_components:
        logger.warning(
            f"Number of features ({n_features}) is less than requested components ({n_components}). "
            f"Reducing components to {n_features}."
        )
        n_components = min(n_components, n_features)
    
    # Suppress numpy warnings for empty slices
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom')
        
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(feature_matrix)
    
    logger.debug(f"PCA computed: {n_samples} samples, {n_components} components, "
                f"explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    return transformed, pca


def cluster_papers(
    pca_data: np.ndarray,
    n_clusters: int = 5
) -> np.ndarray:
    """Cluster papers using K-means on PCA space.
    
    Args:
        pca_data: PCA-transformed data.
        n_clusters: Number of clusters.
        
    Returns:
        Cluster labels for each paper.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for clustering")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_data)
    
    return labels


def create_pca_2d_plot(
    corpus: Optional[TextCorpus] = None,
    output_path: Optional[Path] = None,
    aggregator: Optional[DataAggregator] = None,
    n_components: int = 2,
    n_clusters: Optional[int] = None,
    format: str = "png"
) -> Path:
    """Create 2D PCA visualization.
    
    Args:
        corpus: Optional text corpus (created if not provided).
        output_path: Optional output path.
        aggregator: Optional DataAggregator instance.
        n_components: Number of PCA components (must be 2 for 2D plot).
        n_clusters: Optional number of clusters for coloring.
        format: Output format (png, svg, pdf).
        
    Returns:
        Path to saved plot.
        
    Raises:
        ValueError: If insufficient data for PCA.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for PCA visualization")
    
    if aggregator is None:
        aggregator = DataAggregator()
    
    if corpus is None:
        corpus = aggregator.prepare_text_corpus()
    
    # Validate corpus
    if not corpus.texts and not corpus.abstracts:
        raise ValueError("Cannot create PCA plot: corpus is empty (no texts or abstracts)")
    
    total_docs = len([t for t in corpus.texts if t.strip()]) + len([a for a in corpus.abstracts if a.strip()])
    if total_docs < 2:
        raise ValueError(f"Cannot create PCA plot: need at least 2 documents, got {total_docs}")
    
    try:
        # Extract features
        logger.debug("Extracting text features for PCA 2D plot...")
        feature_matrix, feature_names, valid_indices = extract_text_features(corpus)
        
        # Filter corpus arrays to match valid indices
        filtered_titles = [corpus.titles[i] for i in valid_indices if i < len(corpus.titles)]
        filtered_years = [corpus.years[i] if i < len(corpus.years) else None for i in valid_indices]
        
        logger.debug(f"Aligned arrays: {len(feature_matrix)} samples, {len(filtered_titles)} titles, {len(filtered_years)} years")
        
        # Validate alignment
        if len(feature_matrix) != len(filtered_titles) or len(feature_matrix) != len(filtered_years):
            raise ValueError(
                f"Array size mismatch: feature_matrix={len(feature_matrix)}, "
                f"titles={len(filtered_titles)}, years={len(filtered_years)}"
            )
        
        # Compute PCA
        logger.debug("Computing PCA with 2 components...")
        pca_data, pca_model = compute_pca(feature_matrix, n_components=2)
        logger.info(f"PCA computed: {len(pca_data)} samples, "
                   f"explained variance: {pca_model.explained_variance_ratio_.sum():.2%}")
        
        # Optional clustering
        cluster_labels = None
        if n_clusters is not None and len(pca_data) >= n_clusters:
            logger.debug(f"Computing {n_clusters} clusters...")
            cluster_labels = cluster_papers(pca_data, n_clusters=n_clusters)
            logger.debug(f"Clustering complete: {len(np.unique(cluster_labels))} unique clusters")
        elif n_clusters is not None:
            logger.warning(f"Cannot cluster: need at least {n_clusters} samples, got {len(pca_data)}")
        
        if output_path is None:
            output_path = Path("data/output/pca_2d." + format)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create plot with aligned arrays
        logger.debug("Creating PCA 2D plot...")
        fig = plot_pca_2d(
            pca_data=pca_data,
            titles=filtered_titles,
            years=filtered_years,
            cluster_labels=cluster_labels,
            explained_variance=pca_model.explained_variance_ratio_,
            title="PCA Analysis of Papers (2D)"
        )
        
        return save_plot(fig, output_path)
    except ValueError as e:
        logger.error(f"PCA 2D plot creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating PCA 2D plot: {e}")
        raise ValueError(f"Failed to create PCA 2D plot: {e}") from e


def create_pca_3d_plot(
    corpus: Optional[TextCorpus] = None,
    output_path: Optional[Path] = None,
    aggregator: Optional[DataAggregator] = None,
    n_clusters: Optional[int] = None,
    format: str = "png"
) -> Path:
    """Create 3D PCA visualization.
    
    Args:
        corpus: Optional text corpus (created if not provided).
        output_path: Optional output path.
        aggregator: Optional DataAggregator instance.
        n_clusters: Optional number of clusters for coloring.
        format: Output format (png, svg, pdf).
        
    Returns:
        Path to saved plot.
        
    Raises:
        ValueError: If insufficient data for PCA.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for PCA visualization")
    
    if aggregator is None:
        aggregator = DataAggregator()
    
    if corpus is None:
        corpus = aggregator.prepare_text_corpus()
    
    # Validate corpus
    if not corpus.texts and not corpus.abstracts:
        raise ValueError("Cannot create PCA plot: corpus is empty (no texts or abstracts)")
    
    total_docs = len([t for t in corpus.texts if t.strip()]) + len([a for a in corpus.abstracts if a.strip()])
    if total_docs < 2:
        raise ValueError(f"Cannot create PCA plot: need at least 2 documents, got {total_docs}")
    
    try:
        # Extract features
        logger.debug("Extracting text features for PCA 3D plot...")
        feature_matrix, feature_names, valid_indices = extract_text_features(corpus)
        
        # Filter corpus arrays to match valid indices
        filtered_titles = [corpus.titles[i] for i in valid_indices if i < len(corpus.titles)]
        filtered_years = [corpus.years[i] if i < len(corpus.years) else None for i in valid_indices]
        
        logger.debug(f"Aligned arrays: {len(feature_matrix)} samples, {len(filtered_titles)} titles, {len(filtered_years)} years")
        
        # Validate alignment
        if len(feature_matrix) != len(filtered_titles) or len(feature_matrix) != len(filtered_years):
            raise ValueError(
                f"Array size mismatch: feature_matrix={len(feature_matrix)}, "
                f"titles={len(filtered_titles)}, years={len(filtered_years)}"
            )
        
        # Compute PCA (3D requires at least 3 features, but we'll handle gracefully)
        logger.debug("Computing PCA with 3 components...")
        pca_data, pca_model = compute_pca(feature_matrix, n_components=3)
        logger.info(f"PCA computed: {len(pca_data)} samples, "
                   f"explained variance: {pca_model.explained_variance_ratio_.sum():.2%}")
        
        # Optional clustering
        cluster_labels = None
        if n_clusters is not None and len(pca_data) >= n_clusters:
            logger.debug(f"Computing {n_clusters} clusters...")
            cluster_labels = cluster_papers(pca_data, n_clusters=n_clusters)
            logger.debug(f"Clustering complete: {len(np.unique(cluster_labels))} unique clusters")
        elif n_clusters is not None:
            logger.warning(f"Cannot cluster: need at least {n_clusters} samples, got {len(pca_data)}")
        
        if output_path is None:
            output_path = Path("data/output/pca_3d." + format)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create plot with aligned arrays
        logger.debug("Creating PCA 3D plot...")
        fig = plot_pca_3d(
            pca_data=pca_data,
            titles=filtered_titles,
            years=filtered_years,
            cluster_labels=cluster_labels,
            explained_variance=pca_model.explained_variance_ratio_,
            title="PCA Analysis of Papers (3D)"
        )
        
        return save_plot(fig, output_path)
    except ValueError as e:
        logger.error(f"PCA 3D plot creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating PCA 3D plot: {e}")
        raise ValueError(f"Failed to create PCA 3D plot: {e}") from e


def export_pca_loadings(
    corpus: Optional[TextCorpus] = None,
    aggregator: Optional[DataAggregator] = None,
    n_components: int = 5,
    top_n_words: int = 20,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Export PCA loadings in multiple formats.
    
    Convenience function that wraps pca_loadings.export_all_loadings.
    
    Args:
        corpus: Optional text corpus (created if not provided).
        aggregator: Optional DataAggregator instance.
        n_components: Number of principal components.
        top_n_words: Number of top words per component.
        output_dir: Output directory (defaults to data/output).
        
    Returns:
        Dictionary mapping format name to output path.
    """
    from infrastructure.literature.meta_analysis.pca_loadings import export_all_loadings
    
    return export_all_loadings(
        corpus=corpus,
        aggregator=aggregator,
        n_components=n_components,
        top_n_words=top_n_words,
        output_dir=output_dir
    )


