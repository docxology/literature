"""Meta-analysis operation functions for literature workflow."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from infrastructure.core.logging_utils import get_logger, log_header, log_success
from infrastructure.literature.workflow.workflow import LiteratureWorkflow
from infrastructure.literature.workflow.operations.search import (
    get_keywords_input,
    get_limit_input,
    get_clear_options_input,
)
from infrastructure.literature.workflow.operations.utils import get_source_selection_input
from infrastructure.literature.workflow.operations.download import (
    library_entry_to_search_result,
    failed_download_to_search_result,
    find_papers_needing_pdf,
)
from infrastructure.literature.workflow.operations.cleanup import find_orphaned_pdfs

logger = get_logger(__name__)

def run_meta_analysis(
    workflow: LiteratureWorkflow,
    keywords: Optional[List[str]] = None,
    limit: Optional[int] = None,
    clear_pdfs: bool = False,
    clear_library: bool = False,
    interactive: bool = True,
    sources: Optional[List[str]] = None,
    retry_failed: bool = False,
) -> int:
    """Execute literature search workflow with meta-analysis.
    
    Runs search → download → extract → meta-analysis pipeline.
    Performs PCA analysis, keyword analysis, author analysis, and visualizations.
    
    Args:
        workflow: Configured LiteratureWorkflow instance.
        keywords: Optional keywords list (prompts if not provided).
        limit: Optional limit per keyword (prompts if not provided).
        clear_pdfs: Clear PDFs before download (default: False).
        clear_library: Perform total clear before operations (default: False).
        interactive: Whether in interactive mode.
        sources: Optional list of sources to use (prompts if not provided and interactive=True).
        
    Returns:
        Exit code (0=success, 1=failure).
    """
    log_header("LITERATURE SEARCH AND META-ANALYSIS PIPELINE")
    
    # Handle clear operations
    from infrastructure.literature.library.clear import clear_pdfs as clear_pdfs_func, clear_library as clear_library_func
    
    if clear_library:
        result = clear_library_func(confirm=True, interactive=interactive)
        if not result["success"]:
            logger.info("Library clear cancelled")
            return 1
        logger.info(f"Total clear completed: {result['message']}")
        # Reload the workflow's library index to reflect the cleared state
        workflow.literature_search.library_index.reload()
        clear_pdfs = False
    elif clear_pdfs:
        result = clear_pdfs_func(confirm=True, interactive=interactive)
        if not result["success"]:
            logger.info("PDF clear cancelled")
            return 1
        logger.info(f"Cleared PDFs: {result['message']}")
    
    # Get source selection if not provided
    if sources is None and interactive:
        sources = get_source_selection_input(workflow, default_enabled=True)
        if not sources:
            logger.error("No sources selected. Exiting.")
            return 1
    elif sources is None:
        # Non-interactive mode: use all available searchable sources
        enabled_sources = list(workflow.literature_search.sources.keys())
        sources = [s for s in enabled_sources 
                   if hasattr(workflow.literature_search.sources[s], 'search')]
    
    # Format sources display
    if not sources:
        sources_display = "no sources"
    elif len(sources) <= 8:
        sources_display = ', '.join(sources)
    else:
        sources_display = f"{', '.join(sources[:5])}, and {len(sources) - 5} more"

    logger.info("\nThis will:")
    logger.info(f"  1. Search {sources_display} for papers")
    logger.info("  2. Download PDFs and add to BibTeX library")
    logger.info("  3. Extract text from PDFs")
    logger.info("  4. Perform meta-analysis (PCA, keywords, authors, visualizations)")
    logger.info("")
    
    # Get limit if not provided
    if limit is None:
        limit = get_limit_input()
    
    # Get keywords if not provided
    if keywords is None:
        keywords = get_keywords_input()
        if not keywords:
            logger.info("No keywords provided. Exiting.")
            return 1
    
    # Get clear options if in interactive mode
    if interactive and not (clear_pdfs or clear_library):
        clear_pdfs, _, clear_library = get_clear_options_input()
        if clear_library:
            result = clear_library_func(confirm=True, interactive=interactive)
            if not result["success"]:
                logger.info("Library clear cancelled")
                return 1
            logger.info(f"Total clear completed: {result['message']}")
            # Reload the workflow's library index to reflect the cleared state
            workflow.literature_search.library_index.reload()
            clear_pdfs = False
        elif clear_pdfs:
            result = clear_pdfs_func(confirm=True, interactive=interactive)
            if not result["success"]:
                logger.info("PDF clear cancelled")
                return 1
            logger.info(f"Cleared PDFs: {result['message']}")
    
    # Check for failed downloads and prompt for retry
    if interactive and not retry_failed and workflow.failed_tracker.has_failures():
        retriable_count = workflow.failed_tracker.count_retriable()
        total_failed = workflow.failed_tracker.count_failures()
        if retriable_count > 0:
            logger.info(f"\nFound {retriable_count} retriable failed downloads (out of {total_failed} total)")
            retry_choice = input("Retry previously failed downloads? [y/N]: ").strip().lower()
            retry_failed = retry_choice in ('y', 'yes')
    
    # Step 1: Search
    log_header("STEP 1: SEARCHING FOR PAPERS")
    logger.info(f"Search keywords: {', '.join(keywords)}")
    logger.info(f"Results per keyword: {limit}")
    logger.info(f"Sources: {', '.join(sources)}")
    
    try:
        search_results = workflow._search_papers(keywords, limit, sources=sources)
        papers_found = len(search_results)
        
        if not search_results:
            logger.warning("No papers found for the given keywords")
            return 1
        
        # Add all results to library
        added_count = 0
        already_existed_count = 0
        
        for result in search_results:
            try:
                citation_key = workflow.literature_search.add_to_library(result)
                added_count += 1
                logger.info(f"Added: {citation_key}")
            except Exception:
                already_existed_count += 1
                logger.debug(f"Already exists: {result.title[:50]}...")
        
        logger.info(f"Papers found: {papers_found}")
        logger.info(f"Papers added to bibliography: {added_count}")
        if already_existed_count > 0:
            logger.info(f"Papers already in bibliography: {already_existed_count}")
        
        # Step 2: Download PDFs
        log_header("STEP 2: DOWNLOADING PDFs")
        library_entries = workflow.literature_search.library_index.list_entries()
        papers_needing_pdf = find_papers_needing_pdf(
            library_entries,
            failed_tracker=workflow.failed_tracker,
            retry_failed=retry_failed
        )
        
        # Add failed downloads if retry requested
        failed_results = []
        if retry_failed and workflow.failed_tracker.has_failures():
            failed_downloads = workflow.failed_tracker.get_retriable_failed()
            logger.info(f"Found {len(failed_downloads)} retriable failed downloads")
            
            for citation_key, failure_data in failed_downloads.items():
                # Check if PDF already exists (might have been downloaded manually)
                expected_pdf = Path("data/pdfs") / f"{citation_key}.pdf"
                if not expected_pdf.exists():
                    search_result = failed_download_to_search_result(failure_data)
                    failed_results.append((citation_key, search_result))
                else:
                    # Remove from tracker if PDF now exists
                    workflow.failed_tracker.remove_successful(citation_key)
            
            if failed_results:
                logger.info(f"Retrying {len(failed_results)} previously failed downloads")
        
        total_to_download = len(papers_needing_pdf) + len(failed_results)
        
        if total_to_download > 0:
            logger.info(f"Downloading {total_to_download} PDFs...")
            downloaded_count = 0
            failed_count = 0
            
            # Process regular library entries
            for i, entry in enumerate(papers_needing_pdf, 1):
                logger.info(f"[{i}/{total_to_download}] Processing: {entry.title[:60]}...")
                search_result = library_entry_to_search_result(entry)
                download_result = workflow.literature_search.download_paper_with_result(search_result)
                
                if download_result.success and download_result.pdf_path:
                    downloaded_count += 1
                    # Enhanced logging with full path, file size, and source
                    abs_path = download_result.pdf_path.resolve()
                    file_size = abs_path.stat().st_size if abs_path.exists() else 0
                    source = entry.source or "unknown"
                    logger.info(f"✓ Downloaded: {abs_path} ({file_size:,} bytes) [Source: {source}]")
                elif download_result.failure_reason == "no_pdf_url":
                    source = entry.source or "unknown"
                    logger.warning(f"✗ No PDF URL available: {entry.title[:50]}... [Source: {source}]")
                else:
                    # Save to failed tracker
                    workflow.failed_tracker.save_failed(
                        entry.citation_key, download_result,
                        title=entry.title, source=entry.source
                    )
                    
                    failed_count += 1
                    # Enhanced failure logging with full context
                    error_details = f"{download_result.failure_reason or 'unknown'}: {download_result.failure_message or 'No error message'}"
                    urls_attempted = download_result.attempted_urls or []
                    expected_path = Path("data/pdfs") / f"{entry.citation_key}.pdf"
                    source = entry.source or "unknown"
                    
                    logger.error(f"✗ Failed: {entry.citation_key}")
                    logger.error(f"  Title: {entry.title[:80]}...")
                    logger.error(f"  Error: {error_details}")
                    logger.error(f"  Expected path: {expected_path.resolve()}")
                    logger.error(f"  Source: {source}")
                    if urls_attempted:
                        logger.error(f"  URLs attempted: {len(urls_attempted)}")
                        for j, url in enumerate(urls_attempted[:3], 1):  # Show first 3
                            logger.error(f"    {j}. {url[:100]}...")
                        if len(urls_attempted) > 3:
                            logger.error(f"    ... and {len(urls_attempted) - 3} more")
            
            # Process failed downloads retry
            for i, (citation_key, search_result) in enumerate(failed_results, len(papers_needing_pdf) + 1):
                logger.info(f"[{i}/{total_to_download}] Retrying: {search_result.title[:60]}...")
                download_result = workflow.literature_search.download_paper_with_result(search_result)
                
                if download_result.success and download_result.pdf_path:
                    downloaded_count += 1
                    abs_path = download_result.pdf_path.resolve()
                    file_size = abs_path.stat().st_size if abs_path.exists() else 0
                    source = search_result.source or "unknown"
                    logger.info(f"✓ Retry successful: {abs_path} ({file_size:,} bytes) [Source: {source}]")
                else:
                    # Save to failed tracker
                    workflow.failed_tracker.save_failed(
                        citation_key, download_result,
                        title=search_result.title, source=search_result.source
                    )
                    
                    failed_count += 1
                    error_details = f"{download_result.failure_reason or 'unknown'}: {download_result.failure_message or 'No error message'}"
                    expected_path = Path("data/pdfs") / f"{citation_key}.pdf"
                    source = search_result.source or "unknown"
                    
                    logger.error(f"✗ Retry failed: {citation_key}")
                    logger.error(f"  Title: {search_result.title[:80]}...")
                    logger.error(f"  Error: {error_details}")
                    logger.error(f"  Expected path: {expected_path.resolve()}")
                    logger.error(f"  Source: {source}")
            
            logger.info(f"PDFs downloaded: {downloaded_count}")
            if failed_count > 0:
                logger.warning(f"Download failures: {failed_count}")
        else:
            logger.info("All papers already have PDFs")
        
        # Step 3: Extract text
        log_header("STEP 3: EXTRACTING TEXT FROM PDFs")
        from infrastructure.literature.summarization.orchestrator import run_extract_text
        extract_exit_code = run_extract_text(workflow)
        if extract_exit_code != 0:
            logger.warning("Some text extractions failed, continuing with meta-analysis...")
        
        # Step 4: Meta-analysis
        log_header("STEP 4: PERFORMING META-ANALYSIS")
        meta_analysis_start = time.time()
        
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
            logger.warning("No papers found (library is empty and no orphaned PDFs). Cannot perform meta-analysis.")
            return 1
        
        logger.info(f"Analyzing {len(all_entries)} papers ({len(library_entries)} from library, {len(orphaned_entries)} orphaned PDFs)...")
        
        # Initialize aggregator with all entries (library + orphaned)
        from infrastructure.literature.meta_analysis.aggregator import DataAggregator
        aggregator = DataAggregator(workflow.literature_search.config, default_entries=all_entries)
        
        # Validate data quality and log metrics
        logger.info("Validating data quality...")
        quality_metrics = aggregator.validate_data_quality()
        logger.info(f"Data quality metrics:")
        logger.info(f"  Total papers: {quality_metrics['total']}")
        logger.info(f"  Year coverage: {quality_metrics['year_coverage']:.1f}% ({quality_metrics['has_year']}/{quality_metrics['total']})")
        logger.info(f"  Author coverage: {quality_metrics['author_coverage']:.1f}% ({quality_metrics['has_authors']}/{quality_metrics['total']})")
        logger.info(f"  Abstract coverage: {quality_metrics['abstract_coverage']:.1f}% ({quality_metrics['has_abstract']}/{quality_metrics['total']})")
        logger.info(f"  DOI coverage: {quality_metrics['doi_coverage']:.1f}% ({quality_metrics['has_doi']}/{quality_metrics['total']})")
        logger.info(f"  PDF coverage: {quality_metrics['pdf_coverage']:.1f}% ({quality_metrics['has_pdf']}/{quality_metrics['total']})")
        logger.info(f"  Extracted text coverage: {quality_metrics['extracted_text_coverage']:.1f}% ({quality_metrics['has_extracted_text']}/{quality_metrics['total']})")
        logger.info(f"  Sufficient for PCA: {quality_metrics['sufficient_for_pca']}")
        logger.info(f"  Sufficient for keywords: {quality_metrics['sufficient_for_keywords']}")
        logger.info(f"  Sufficient for temporal: {quality_metrics['sufficient_for_temporal']}")
        
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
                create_citation_distribution_plot,
            )
            
            venue_path = create_venue_distribution_plot(
                top_n=15, aggregator=aggregator, format="png"
            )
            outputs_generated.append(("Venue Distribution", venue_path))
            step_time = time.time() - step_start
            logger.info(f"✓ Generated: {venue_path.name} ({step_time:.2f}s)")
            analysis_steps.append(("Venue Distribution", step_time))
            
            step_start = time.time()
            citation_path = create_citation_distribution_plot(
                aggregator=aggregator, format="png"
            )
            outputs_generated.append(("Citation Distribution", citation_path))
            step_time = time.time() - step_start
            logger.info(f"✓ Generated: {citation_path.name} ({step_time:.2f}s)")
            analysis_steps.append(("Citation Distribution", step_time))
            
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
                keywords=keywords,
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
                keywords=keywords,
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
            logger.error(f"  Keywords: {', '.join(keywords) if keywords else 'None'}")
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
        
        # Calculate total time
        total_meta_time = time.time() - meta_analysis_start
        
        # Display summary
        logger.info(f"\n{'=' * 60}")
        logger.info("META-ANALYSIS COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Keywords searched: {', '.join(keywords)}")
        logger.info(f"Papers found: {papers_found}")
        logger.info(f"Papers added: {added_count}")
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
        
    except Exception as e:
        logger.error(f"Meta-analysis pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
