"""Meta-analysis operation functions for literature workflow."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from infrastructure.core.logging_utils import get_logger, log_header, log_success
from infrastructure.literature.workflow.workflow import LiteratureWorkflow
from infrastructure.literature.workflow.operations.cleanup import find_orphaned_pdfs

logger = get_logger(__name__)

def run_meta_analysis(
    workflow: LiteratureWorkflow,
    interactive: bool = True,
    include_embeddings: bool = False,
) -> int:
    """Execute meta-analysis on existing library data.
    
    Analyzes existing citations, PDFs, and extracted text in the library.
    Does not perform search, download, or extraction - those are handled separately.
    Logs warnings about missing data but proceeds with available data.
    
    Args:
        workflow: Configured LiteratureWorkflow instance.
        interactive: Whether in interactive mode.
        include_embeddings: Whether to include Ollama embedding analysis (default: False).
            Requires Ollama server running and ≥2 papers with extracted text.
        
    Returns:
        Exit code (0=success, 1=failure).
    """
    log_header("META-ANALYSIS")
    
    logger.info("Analyzing existing library data (citations, PDFs, extracted text)")
    if include_embeddings:
        logger.info("Embedding analysis will be included (requires Ollama)")
    logger.info("")
    
    # Ensure output directory exists
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Get library entries for analysis
    library_entries = workflow.literature_search.library_index.list_entries()
    
    # Find orphaned PDFs and include them in analysis
    orphaned_entries = find_orphaned_pdfs(library_entries)
    
    # Combine library entries with orphaned entries for comprehensive analysis
    all_entries = library_entries + orphaned_entries
    
    if not all_entries:
        logger.warning("No papers found in library. Cannot perform meta-analysis.")
        logger.info("Use option 3.1 to search for papers, or option 4.1-4.3 to download PDFs and extract text.")
        return 1
    
    logger.info(f"Analyzing {len(all_entries)} papers ({len(library_entries)} from library, {len(orphaned_entries)} orphaned PDFs)...")
    
    # Initialize aggregator with all entries (library + orphaned)
    from infrastructure.literature.meta_analysis.aggregator import DataAggregator
    aggregator = DataAggregator(workflow.literature_search.config, default_entries=all_entries)
    
    # Validate data quality and log metrics
    logger.info("Assessing data quality...")
    quality_metrics = aggregator.validate_data_quality()
    
    # Log data quality report
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)
    logger.info(f"Total papers: {quality_metrics['total']}")
    logger.info(f"  • Papers with year: {quality_metrics['has_year']} ({quality_metrics['year_coverage']:.1f}%)")
    logger.info(f"  • Papers with authors: {quality_metrics['has_authors']} ({quality_metrics['author_coverage']:.1f}%)")
    logger.info(f"  • Papers with abstract: {quality_metrics['has_abstract']} ({quality_metrics['abstract_coverage']:.1f}%)")
    logger.info(f"  • Papers with DOI: {quality_metrics['has_doi']} ({quality_metrics['doi_coverage']:.1f}%)")
    logger.info(f"  • Papers with PDF: {quality_metrics['has_pdf']} ({quality_metrics['pdf_coverage']:.1f}%)")
    logger.info(f"  • Papers with extracted text: {quality_metrics['has_extracted_text']} ({quality_metrics['extracted_text_coverage']:.1f}%)")
    logger.info("")
    
    # Log warnings about missing data
    missing_pdfs = quality_metrics['total'] - quality_metrics['has_pdf']
    missing_extracted = quality_metrics['has_pdf'] - quality_metrics['has_extracted_text']
    
    if missing_pdfs > 0:
        logger.warning(f"⚠ {missing_pdfs} paper(s) missing PDFs. Use option 4.1 to download PDFs.")
    if missing_extracted > 0:
        logger.warning(f"⚠ {missing_extracted} paper(s) with PDFs but no extracted text. Use option 4.2 or 4.3 to extract text.")
    if quality_metrics['has_extracted_text'] < 2 and include_embeddings:
        logger.warning(f"⚠ Embedding analysis requires ≥2 papers with extracted text (found {quality_metrics['has_extracted_text']}).")
    
    logger.info("")
    logger.info("Proceeding with meta-analysis on available data...")
    logger.info("")
    
    meta_analysis_start = time.time()
    
    # Perform meta-analysis operations
    outputs_generated = []
    analysis_steps = []
    
    try:
        # PCA Analysis
        if not quality_metrics['sufficient_for_pca']:
            logger.warning(f"PCA analysis skipped: insufficient data (need at least 2 papers with extracted text, got {quality_metrics['has_extracted_text']})")
        else:
            step_start = time.time()
            logger.info("Generating PCA visualizations...")
            from infrastructure.literature.meta_analysis.pca import (
                create_pca_2d_plot,
                create_pca_3d_plot,
            )
            
            pca_2d_path = create_pca_2d_plot(aggregator=aggregator, n_clusters=5, format="png")
            outputs_generated.append(("PCA 2D", pca_2d_path))
            step_time = time.time() - step_start
            logger.info(f"✓ Generated: {pca_2d_path.name} ({step_time:.2f}s)")
            analysis_steps.append(("PCA 2D", step_time))
            
            step_start = time.time()
            pca_3d_path = create_pca_3d_plot(aggregator=aggregator, n_clusters=5, format="png")
            outputs_generated.append(("PCA 3D", pca_3d_path))
            step_time = time.time() - step_start
            logger.info(f"✓ Generated: {pca_3d_path.name} ({step_time:.2f}s)")
            analysis_steps.append(("PCA 3D", step_time))
        
    except ImportError as e:
        logger.warning(f"PCA analysis skipped (scikit-learn not available): {e}")
    except ValueError as e:
        logger.error(f"PCA analysis skipped: {e}")
        logger.debug(f"PCA analysis ValueError details: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"PCA analysis failed: {e}")
        import traceback
        logger.error(f"PCA analysis error details: {traceback.format_exc()}")
        logger.debug(f"Full PCA analysis traceback: {traceback.format_exc()}")
    
    try:
        # Keyword Analysis
        if not quality_metrics['sufficient_for_keywords']:
            logger.warning(f"Keyword analysis skipped: insufficient data (need at least 1 abstract, got {quality_metrics['has_abstract']})")
        else:
            step_start = time.time()
            logger.info("Generating keyword analysis...")
            from infrastructure.literature.meta_analysis.keywords import (
                create_keyword_frequency_plot,
                create_keyword_evolution_plot,
            )
            
            keyword_data = aggregator.prepare_keyword_data()
            keyword_freq_path = create_keyword_frequency_plot(
                keyword_data, top_n=20, format="png"
            )
            outputs_generated.append(("Keyword Frequency", keyword_freq_path))
            step_time = time.time() - step_start
            logger.info(f"✓ Generated: {keyword_freq_path.name} ({step_time:.2f}s)")
            analysis_steps.append(("Keyword Frequency", step_time))
            
            # Get top keywords for evolution plot
            sorted_keywords = sorted(
                keyword_data.keyword_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            top_keywords = [k for k, _ in sorted_keywords]
            
            if top_keywords:
                step_start = time.time()
                keyword_evol_path = create_keyword_evolution_plot(
                    keyword_data, keywords=top_keywords, format="png"
                )
                outputs_generated.append(("Keyword Evolution", keyword_evol_path))
                step_time = time.time() - step_start
                logger.info(f"✓ Generated: {keyword_evol_path.name} ({step_time:.2f}s)")
                analysis_steps.append(("Keyword Evolution", step_time))
            else:
                logger.warning("No keywords found for evolution plot")
        
    except Exception as e:
        logger.warning(f"Keyword analysis failed: {e}")
        import traceback
        logger.debug(f"Keyword analysis error details: {traceback.format_exc()}")
    
    try:
        # Author Analysis
        if quality_metrics['has_authors'] == 0:
            logger.warning("Author analysis skipped: no authors found in data")
        else:
            step_start = time.time()
            logger.info("Generating author analysis...")
            from infrastructure.literature.meta_analysis.metadata import (
                create_author_contributions_plot,
            )
            
            author_path = create_author_contributions_plot(
                top_n=20, aggregator=aggregator, format="png"
            )
            outputs_generated.append(("Author Contributions", author_path))
            step_time = time.time() - step_start
            logger.info(f"✓ Generated: {author_path.name} ({step_time:.2f}s)")
            analysis_steps.append(("Author Contributions", step_time))
        
    except Exception as e:
        logger.warning(f"Author analysis failed: {e}")
        import traceback
        logger.warning(f"  Error type: {type(e).__name__}")
        logger.warning(f"  Total entries: {len(all_entries)}")
        logger.warning(f"  Entries with authors: {quality_metrics.get('has_authors', 0)}")
        logger.debug(f"Author analysis error details: {traceback.format_exc()}")
    
    try:
        # Metadata Visualizations
        step_start = time.time()
        logger.info("Generating metadata visualizations...")
        from infrastructure.literature.meta_analysis.metadata import (
            create_venue_distribution_plot,
        )
        
        venue_path = create_venue_distribution_plot(
            top_n=15, aggregator=aggregator, format="png"
        )
        outputs_generated.append(("Venue Distribution", venue_path))
        step_time = time.time() - step_start
        logger.info(f"✓ Generated: {venue_path.name} ({step_time:.2f}s)")
        analysis_steps.append(("Venue Distribution", step_time))
        
    except Exception as e:
        logger.warning(f"Metadata visualization failed: {e}")
        import traceback
        logger.warning(f"  Error type: {type(e).__name__}")
        logger.warning(f"  Total entries: {len(all_entries)}")
        # Try to get more context about the data
        try:
            metadata_sample = aggregator.prepare_metadata_data()
            logger.warning(f"  Venues found: {len(metadata_sample.venues)}")
            logger.warning(f"  Authors found: {len(metadata_sample.authors)}")
            logger.warning(f"  Sources found: {len(metadata_sample.sources)}")
        except Exception as inner_e:
            logger.warning(f"  Could not gather metadata context: {inner_e}")
        logger.debug(f"Metadata visualization error details: {traceback.format_exc()}")
    
    try:
        # Temporal Analysis
        if not quality_metrics['sufficient_for_temporal']:
            logger.warning(f"Temporal analysis skipped: insufficient data (need at least 1 paper with year, got {quality_metrics['has_year']})")
        else:
            step_start = time.time()
            logger.info("Generating temporal analysis...")
            from infrastructure.literature.meta_analysis.temporal import (
                create_publication_timeline_plot,
            )
            
            timeline_path = create_publication_timeline_plot(
                aggregator=aggregator, format="png"
            )
            outputs_generated.append(("Publication Timeline", timeline_path))
            step_time = time.time() - step_start
            logger.info(f"✓ Generated: {timeline_path.name} ({step_time:.2f}s)")
            analysis_steps.append(("Publication Timeline", step_time))
        
    except Exception as e:
        logger.warning(f"Temporal analysis failed: {e}")
        import traceback
        logger.debug(f"Temporal analysis error details: {traceback.format_exc()}")
    
    try:
        # PCA Loadings Export
        if not quality_metrics['sufficient_for_pca']:
            logger.warning("PCA loadings export skipped: insufficient data for PCA")
        else:
            step_start = time.time()
            logger.info("Exporting PCA loadings...")
            from infrastructure.literature.meta_analysis.pca import export_pca_loadings
            
            loadings_outputs = export_pca_loadings(
                aggregator=aggregator,
                n_components=5,
                top_n_words=20,
                output_dir=Path("data/output")
            )
            
            step_time = time.time() - step_start
            for format_name, path in loadings_outputs.items():
                outputs_generated.append((f"PCA Loadings ({format_name})", path))
                logger.info(f"✓ Generated: {path.name}")
            analysis_steps.append(("PCA Loadings Export", step_time))
        
    except ImportError as e:
        logger.warning(f"PCA loadings export skipped (scikit-learn not available): {e}")
    except ValueError as e:
        logger.warning(f"PCA loadings export skipped: {e}")
    except Exception as e:
        logger.warning(f"PCA loadings export failed: {e}")
        import traceback
        logger.debug(f"PCA loadings export error details: {traceback.format_exc()}")
    
    try:
        # PCA Loadings Visualizations
        if not quality_metrics['sufficient_for_pca']:
            logger.warning("PCA loadings visualizations skipped: insufficient data for PCA")
        else:
            step_start = time.time()
            logger.info("Generating PCA loadings visualizations...")
            from infrastructure.literature.meta_analysis.pca_loadings import create_loadings_visualizations
            
            loadings_viz_outputs = create_loadings_visualizations(
                aggregator=aggregator,
                n_components=5,
                top_n_words=20,
                output_dir=Path("data/output"),
                format="png"
            )
            
            step_time = time.time() - step_start
            for viz_name, path in loadings_viz_outputs.items():
                outputs_generated.append((f"PCA Loadings ({viz_name})", path))
                logger.info(f"✓ Generated: {path.name}")
            analysis_steps.append(("PCA Loadings Visualizations", step_time))
        
    except ImportError as e:
        logger.warning(f"PCA loadings visualizations skipped (scikit-learn not available): {e}")
    except ValueError as e:
        logger.warning(f"PCA loadings visualizations skipped: {e}")
    except Exception as e:
        logger.warning(f"PCA loadings visualizations failed: {e}")
        import traceback
        logger.debug(f"PCA loadings visualizations error details: {traceback.format_exc()}")
    
    try:
        # Metadata Completeness Visualization
        step_start = time.time()
        logger.info("Generating metadata completeness visualization...")
        from infrastructure.literature.meta_analysis.metadata import create_metadata_completeness_plot
        
        completeness_path = create_metadata_completeness_plot(
            aggregator=aggregator, format="png"
        )
        outputs_generated.append(("Metadata Completeness", completeness_path))
        step_time = time.time() - step_start
        logger.info(f"✓ Generated: {completeness_path.name} ({step_time:.2f}s)")
        analysis_steps.append(("Metadata Completeness", step_time))
        
    except Exception as e:
        logger.warning(f"Metadata completeness visualization failed: {e}")
        import traceback
        logger.warning(f"  Error type: {type(e).__name__}")
        logger.warning(f"  Total entries: {len(all_entries)}")
        logger.debug(f"Metadata completeness error details: {traceback.format_exc()}")
    
    try:
        # Graphical Abstracts
        step_start = time.time()
        logger.info("Generating graphical abstracts...")
        from infrastructure.literature.meta_analysis.graphical_abstract import (
            create_single_page_abstract,
            create_multi_page_abstract,
        )
        
        # Single-page abstract
        single_page_path = create_single_page_abstract(
            aggregator=aggregator,
            keywords=None,  # No keywords since we're analyzing existing library
            format="png"
        )
        outputs_generated.append(("Graphical Abstract (Single Page)", single_page_path))
        step_time = time.time() - step_start
        logger.info(f"✓ Generated: {single_page_path.name} ({step_time:.2f}s)")
        analysis_steps.append(("Graphical Abstract (Single Page)", step_time))
        
        # Multi-page abstract
        step_start = time.time()
        multi_page_path = create_multi_page_abstract(
            aggregator=aggregator,
            keywords=None,  # No keywords since we're analyzing existing library
            format="pdf"
        )
        outputs_generated.append(("Graphical Abstract (Multi-Page)", multi_page_path))
        step_time = time.time() - step_start
        logger.info(f"✓ Generated: {multi_page_path.name} ({step_time:.2f}s)")
        analysis_steps.append(("Graphical Abstract (Multi-Page)", step_time))
        
    except Exception as e:
        logger.error(f"Graphical abstract generation failed: {e}")
        import traceback
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Total entries: {len(all_entries)}")
        # Try to get more context about the data
        try:
            metadata_sample = aggregator.prepare_metadata_data()
            logger.error(f"  Venues found: {len(metadata_sample.venues)}")
            logger.error(f"  This error may be due to data type issues in metadata (e.g., venue as list)")
        except Exception as inner_e:
            logger.error(f"  Could not gather metadata context: {inner_e}")
        logger.error(f"Graphical abstract error details: {traceback.format_exc()}")
        logger.debug(f"Full graphical abstract traceback: {traceback.format_exc()}")
    
    try:
        # Summary Reports
        step_start = time.time()
        logger.info("Generating summary reports...")
        from infrastructure.literature.meta_analysis.summary import generate_all_summaries
        
        summary_outputs = generate_all_summaries(
            aggregator=aggregator,
            output_dir=Path("data/output"),
            n_pca_components=5
        )
        
        step_time = time.time() - step_start
        for format_name, path in summary_outputs.items():
            outputs_generated.append((f"Summary ({format_name})", path))
            logger.info(f"✓ Generated: {path.name}")
        analysis_steps.append(("Summary Reports", step_time))
        
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        import traceback
        logger.warning(f"  Error type: {type(e).__name__}")
        logger.warning(f"  Total entries: {len(all_entries)}")
        logger.warning(f"  This error may be due to data type issues in metadata")
        logger.debug(f"Summary generation error details: {traceback.format_exc()}")
    
    # Embedding Analysis (only if requested)
    if include_embeddings:
        try:
            if quality_metrics['has_extracted_text'] < 2:
                logger.warning(f"Embedding analysis skipped: insufficient data (need at least 2 papers with extracted text, got {quality_metrics['has_extracted_text']})")
            else:
                step_start = time.time()
                logger.info("Generating embedding analysis...")
                from infrastructure.literature.llm.embeddings import (
                    generate_document_embeddings,
                    compute_similarity_matrix,
                    cluster_embeddings,
                    reduce_dimensions,
                    export_embeddings,
                    export_similarity_matrix,
                    export_clusters,
                    EmbeddingData,
                    SimilarityResults,
                )
                from infrastructure.literature.meta_analysis.visualizations import (
                    plot_embedding_similarity_heatmap,
                    plot_embedding_clusters_2d,
                    plot_embedding_clusters_3d,
                    save_plot,
                )
                
                # Prepare corpus
                corpus = aggregator.prepare_text_corpus()
                
                # Import validation and statistics modules
                from infrastructure.literature.meta_analysis.embedding_validation import validate_all
                from infrastructure.literature.meta_analysis.embedding_statistics import compute_all_statistics
                from infrastructure.literature.meta_analysis.visualizations import (
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
                from infrastructure.literature.llm.embeddings import (
                    export_embedding_statistics,
                    export_validation_report,
                    export_clustering_metrics,
                )
                
                # Generate embeddings
                logger.info("Checking embedding cache and generating missing embeddings...")
                embedding_data = generate_document_embeddings(
                    corpus,
                    config=workflow.literature_search.config,
                    show_progress=True
                )
                
                # Export embeddings
                embeddings_path = Path("data/output/embeddings.json")
                export_embeddings(embedding_data, embeddings_path)
                outputs_generated.append(("Embeddings", embeddings_path))
                logger.info(f"✓ Exported embeddings: {embeddings_path.name}")
                
                # Validate embeddings
                logger.info("Validating embeddings...")
                validation_results = validate_all(
                    embedding_data,
                    expected_dim=embedding_data.embedding_dimension
                )
                
                # Log validation warnings
                if validation_results.get("has_warnings"):
                    logger.warning("Embedding validation warnings:")
                    for warning in validation_results.get("all_warnings", []):
                        logger.warning(f"  • {warning}")
                
                if validation_results.get("has_errors"):
                    logger.error("Embedding validation errors detected!")
                
                # Export validation report
                validation_path = Path("data/output/embedding_validation_report.json")
                export_validation_report(validation_results, validation_path)
                outputs_generated.append(("Validation Report", validation_path))
                logger.info(f"✓ Exported validation report: {validation_path.name}")
                
                # Compute similarity matrix
                logger.info("Computing similarity matrix...")
                similarity_matrix = compute_similarity_matrix(embedding_data.embeddings)
                similarity_results = SimilarityResults(
                    similarity_matrix=similarity_matrix,
                    citation_keys=embedding_data.citation_keys,
                    titles=embedding_data.titles
                )
                
                # Validate similarity matrix
                from infrastructure.literature.meta_analysis.embedding_validation import validate_similarity_matrix
                sim_validation = validate_similarity_matrix(
                    similarity_matrix,
                    citation_keys=embedding_data.citation_keys,
                    embeddings=embedding_data.embeddings
                )
                validation_results["similarity"] = sim_validation
                
                if sim_validation.get("warnings"):
                    logger.warning("Similarity matrix validation warnings:")
                    for warning in sim_validation["warnings"]:
                        logger.warning(f"  • {warning}")
                
                # Export similarity matrix
                similarity_path = Path("data/output/embedding_similarity_matrix.csv")
                export_similarity_matrix(similarity_results, similarity_path)
                outputs_generated.append(("Similarity Matrix", similarity_path))
                logger.info(f"✓ Exported similarity matrix: {similarity_path.name}")
                
                # Compute statistics
                logger.info("Computing embedding statistics...")
                statistics = compute_all_statistics(
                    embedding_data,
                    similarity_matrix=similarity_matrix
                )
                
                # Export statistics
                stats_json_path = Path("data/output/embedding_statistics.json")
                export_embedding_statistics(statistics, stats_json_path, format="json")
                outputs_generated.append(("Embedding Statistics (JSON)", stats_json_path))
                logger.info(f"✓ Exported statistics: {stats_json_path.name}")
                
                stats_csv_path = Path("data/output/embedding_statistics.csv")
                export_embedding_statistics(statistics, stats_csv_path, format="csv")
                outputs_generated.append(("Embedding Statistics (CSV)", stats_csv_path))
                logger.info(f"✓ Exported statistics CSV: {stats_csv_path.name}")
                
                # Log key statistics
                if "embedding_stats" in statistics:
                    emb_stats = statistics["embedding_stats"]
                    logger.info(f"  • Documents: {emb_stats.get('n_documents')}, Dimension: {emb_stats.get('embedding_dim')}")
                    logger.info(f"  • Mean norm: {emb_stats.get('mean_norm', 0):.3f}, Std norm: {emb_stats.get('std_norm', 0):.3f}")
                
                if "similarity_stats" in statistics:
                    sim_stats = statistics["similarity_stats"]
                    logger.info(f"  • Similarity mean: {sim_stats.get('mean', 0):.3f}, median: {sim_stats.get('median', 0):.3f}")
                
                if "dimensionality_analysis" in statistics:
                    dim_analysis = statistics["dimensionality_analysis"]
                    logger.info(f"  • Effective dimensions: {dim_analysis.get('effective_dim')} (95% variance)")
                
                # Generate quality visualizations
                logger.info("Creating embedding quality visualizations...")
                
                # Embedding quality plot
                fig = plot_embedding_quality(
                    embedding_data.embeddings,
                    citation_keys=embedding_data.citation_keys
                )
                quality_path = save_plot(fig, Path("data/output/embedding_quality.png"))
                outputs_generated.append(("Embedding Quality", quality_path))
                logger.info(f"✓ Generated: {quality_path.name}")
                
                # Similarity distribution
                fig = plot_similarity_distribution(similarity_matrix)
                sim_dist_path = save_plot(fig, Path("data/output/similarity_distribution.png"))
                outputs_generated.append(("Similarity Distribution", sim_dist_path))
                logger.info(f"✓ Generated: {sim_dist_path.name}")
                
                # Embedding coverage
                fig = plot_embedding_coverage(embedding_data.embeddings)
                coverage_path = save_plot(fig, Path("data/output/embedding_coverage.png"))
                outputs_generated.append(("Embedding Coverage", coverage_path))
                logger.info(f"✓ Generated: {coverage_path.name}")
                
                # Dimensionality analysis
                if "dimensionality_analysis" in statistics:
                    fig = plot_dimensionality_analysis(statistics["dimensionality_analysis"])
                    dim_analysis_path = save_plot(fig, Path("data/output/dimensionality_analysis.png"))
                    outputs_generated.append(("Dimensionality Analysis", dim_analysis_path))
                    logger.info(f"✓ Generated: {dim_analysis_path.name}")
                
                # Outlier visualization
                outlier_indices = validation_results.get("outliers", {}).get("outlier_indices", [])
                if outlier_indices:
                    fig = plot_embedding_outliers(
                        embedding_data.embeddings,
                        outlier_indices,
                        citation_keys=embedding_data.citation_keys
                    )
                    outliers_path = save_plot(fig, Path("data/output/embedding_outliers.png"))
                    outputs_generated.append(("Embedding Outliers", outliers_path))
                    logger.info(f"✓ Generated: {outliers_path.name}")
                
                # Create similarity heatmap
                logger.info("Creating similarity heatmap...")
                fig = plot_embedding_similarity_heatmap(
                    similarity_matrix=similarity_matrix,
                    citation_keys=embedding_data.citation_keys,
                    titles=embedding_data.titles,
                    title="Paper Similarity Matrix (Embeddings)"
                )
                heatmap_path = save_plot(fig, Path("data/output/embedding_similarity_heatmap.png"))
                outputs_generated.append(("Similarity Heatmap", heatmap_path))
                logger.info(f"✓ Generated: {heatmap_path.name}")
                
                # Similarity network (if threshold is reasonable)
                if len(embedding_data.citation_keys) <= 50:  # Only for smaller collections
                    try:
                        fig = plot_similarity_network(
                            similarity_matrix,
                            embedding_data.citation_keys,
                            threshold=0.7
                        )
                        network_path = save_plot(fig, Path("data/output/similarity_network.png"))
                        outputs_generated.append(("Similarity Network", network_path))
                        logger.info(f"✓ Generated: {network_path.name}")
                    except Exception as e:
                        logger.debug(f"Could not generate similarity network: {e}")
                
                # Clustering
                n_clusters = min(5, len(embedding_data.citation_keys))
                cluster_labels = None
                if n_clusters >= 2:
                    logger.info(f"Computing {n_clusters} clusters...")
                    cluster_labels = cluster_embeddings(embedding_data.embeddings, n_clusters=n_clusters)
                    
                    # Compute clustering metrics
                    from infrastructure.literature.meta_analysis.embedding_statistics import compute_clustering_metrics
                    clustering_metrics = compute_clustering_metrics(
                        embedding_data.embeddings,
                        cluster_labels
                    )
                    
                    # Add to statistics
                    statistics["clustering_metrics"] = clustering_metrics
                    
                    # Export clustering metrics
                    clust_metrics_json_path = Path("data/output/clustering_metrics.json")
                    export_clustering_metrics(clustering_metrics, clust_metrics_json_path, format="json")
                    outputs_generated.append(("Clustering Metrics (JSON)", clust_metrics_json_path))
                    logger.info(f"✓ Exported clustering metrics: {clust_metrics_json_path.name}")
                    
                    clust_metrics_csv_path = Path("data/output/clustering_metrics.csv")
                    export_clustering_metrics(clustering_metrics, clust_metrics_csv_path, format="csv")
                    outputs_generated.append(("Clustering Metrics (CSV)", clust_metrics_csv_path))
                    logger.info(f"✓ Exported clustering metrics CSV: {clust_metrics_csv_path.name}")
                    
                    # Log clustering metrics
                    if clustering_metrics.get("silhouette_score") is not None:
                        logger.info(f"  • Silhouette score: {clustering_metrics['silhouette_score']:.3f}")
                    if clustering_metrics.get("davies_bouldin_index") is not None:
                        logger.info(f"  • Davies-Bouldin index: {clustering_metrics['davies_bouldin_index']:.3f}")
                    
                    # Cluster quality visualizations
                    logger.info("Creating cluster quality visualizations...")
                    
                    # Cluster quality metrics bar chart
                    fig = plot_cluster_quality_metrics(clustering_metrics)
                    cluster_quality_path = save_plot(fig, Path("data/output/cluster_quality_metrics.png"))
                    outputs_generated.append(("Cluster Quality Metrics", cluster_quality_path))
                    logger.info(f"✓ Generated: {cluster_quality_path.name}")
                    
                    # Silhouette analysis
                    fig = plot_silhouette_analysis(
                        embedding_data.embeddings,
                        cluster_labels
                    )
                    silhouette_path = save_plot(fig, Path("data/output/silhouette_analysis.png"))
                    outputs_generated.append(("Silhouette Analysis", silhouette_path))
                    logger.info(f"✓ Generated: {silhouette_path.name}")
                    
                    # Cluster size distribution
                    fig = plot_cluster_size_distribution(cluster_labels)
                    cluster_sizes_path = save_plot(fig, Path("data/output/cluster_size_distribution.png"))
                    outputs_generated.append(("Cluster Size Distribution", cluster_sizes_path))
                    logger.info(f"✓ Generated: {cluster_sizes_path.name}")
                    
                    # Export clusters
                    clusters_path = Path("data/output/embedding_clusters.json")
                    export_clusters(embedding_data.citation_keys, cluster_labels, clusters_path)
                    outputs_generated.append(("Cluster Assignments", clusters_path))
                    logger.info(f"✓ Exported clusters: {clusters_path.name}")
                    
                    # 2D visualization
                    logger.info("Creating 2D cluster visualization...")
                    embeddings_2d = reduce_dimensions(embedding_data.embeddings, n_components=2, method="umap")
                    fig = plot_embedding_clusters_2d(
                        embeddings_2d=embeddings_2d,
                        cluster_labels=cluster_labels,
                        titles=embedding_data.titles,
                        years=embedding_data.years,
                        title="Embedding Clusters (2D)"
                    )
                    cluster_2d_path = save_plot(fig, Path("data/output/embedding_clusters_2d.png"))
                    outputs_generated.append(("Embedding Clusters (2D)", cluster_2d_path))
                    logger.info(f"✓ Generated: {cluster_2d_path.name}")
                    
                    # 3D visualization (if enough samples)
                    if len(embedding_data.citation_keys) >= 3:
                        logger.info("Creating 3D cluster visualization...")
                        embeddings_3d = reduce_dimensions(embedding_data.embeddings, n_components=3, method="umap")
                        fig = plot_embedding_clusters_3d(
                            embeddings_3d=embeddings_3d,
                            cluster_labels=cluster_labels,
                            titles=embedding_data.titles,
                            years=embedding_data.years,
                            title="Embedding Clusters (3D)"
                        )
                        cluster_3d_path = save_plot(fig, Path("data/output/embedding_clusters_3d.png"))
                        outputs_generated.append(("Embedding Clusters (3D)", cluster_3d_path))
                        logger.info(f"✓ Generated: {cluster_3d_path.name}")
                else:
                    logger.warning(f"Cannot cluster: need at least 2 samples, got {len(embedding_data.citation_keys)}")
                
                # Re-export statistics with clustering metrics if available
                if cluster_labels is not None:
                    stats_json_path = Path("data/output/embedding_statistics.json")
                    export_embedding_statistics(statistics, stats_json_path, format="json")
                    stats_csv_path = Path("data/output/embedding_statistics.csv")
                    export_embedding_statistics(statistics, stats_csv_path, format="csv")
                
                step_time = time.time() - step_start
                analysis_steps.append(("Embedding Analysis", step_time))
                
        except ImportError as e:
            logger.warning(f"Embedding analysis skipped (missing dependencies): {e}")
        except RuntimeError as e:
            logger.warning(f"Embedding analysis skipped: {e}")
        except Exception as e:
            logger.warning(f"Embedding analysis failed: {e}")
            import traceback
            logger.debug(f"Embedding analysis error details: {traceback.format_exc()}")
    
    # Calculate total time
    total_meta_time = time.time() - meta_analysis_start
    
    # Display summary
    logger.info(f"\n{'=' * 60}")
    logger.info("META-ANALYSIS COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Papers analyzed: {quality_metrics['total']}")
    logger.info(f"Outputs generated: {len(outputs_generated)}")
    logger.info(f"Total analysis time: {total_meta_time:.2f}s")
    
    if analysis_steps:
        logger.info("\nAnalysis step timing:")
        for step_name, step_time in analysis_steps:
            logger.info(f"  • {step_name}: {step_time:.2f}s")
    
    logger.info("\nGenerated visualizations:")
    for name, path in outputs_generated:
        logger.info(f"  • {name}: {path}")
    
    log_success(f"Meta-analysis pipeline complete in {total_meta_time:.0f}s")
    return 0
