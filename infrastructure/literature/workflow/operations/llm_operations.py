"""LLM operation functions for literature workflow."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from infrastructure.core.logging_utils import get_logger, log_header, log_success
from infrastructure.literature.workflow.workflow import LiteratureWorkflow
from infrastructure.literature.workflow.operations.utils import display_file_locations

logger = get_logger(__name__)

def run_llm_operation(workflow: LiteratureWorkflow, operation: str, paper_config_path: Optional[str] = None) -> int:
    """Execute advanced LLM operation on selected papers.

    Args:
        workflow: Configured LiteratureWorkflow instance.
        operation: Type of operation ("review", "communication", "compare", "gaps", "network").
        paper_config_path: Path to paper selection config file.

    Returns:
        Exit code (0=success, 1=failure).
    """
    import time
    from infrastructure.literature.llm.selector import PaperSelector
    from infrastructure.literature.llm.operations import LiteratureLLMOperations
    
    # Map operation names to display names
    operation_names = {
        "review": "Literature Review Synthesis",
        "communication": "Science Communication Narrative",
        "compare": "Comparative Analysis",
        "gaps": "Research Gap Identification",
        "network": "Citation Network Analysis"
    }

    operation_display = operation_names.get(operation, operation.title())
    log_header(f"ADVANCED LLM OPERATION: {operation_display.upper()}")

    # Initialize LLM operations
    llm_operations = LiteratureLLMOperations(workflow.summarizer.llm_client)

    # Load paper selection config
    config_path = Path(paper_config_path) if paper_config_path else Path("literature/paper_selection.yaml")

    try:
        selector = PaperSelector.from_config(config_path)
        logger.info(f"Loaded paper selection config from {config_path}")
    except FileNotFoundError:
        logger.warning(f"Paper selection config not found: {config_path}")
        logger.info("Using all papers in library (create literature/paper_selection.yaml to filter)")

        # Create a selector that selects all papers
        from infrastructure.literature.llm.selector import PaperSelectionConfig
        selector = PaperSelector(PaperSelectionConfig())

    # Get library entries and apply selection
    library_entries = workflow.literature_search.library_index.list_entries()

    if not library_entries:
        logger.warning("Library is empty. Nothing to analyze.")
        logger.warning("Add papers first using --search-only.")
        return 1

    selected_papers = selector.select_papers(library_entries)

    if not selected_papers:
        logger.warning("No papers match the selection criteria.")
        logger.warning(f"No papers match the selection criteria in {config_path}")
        logger.warning("Check your paper_selection.yaml configuration.")
        return 1

    # Display selection summary
    selection_stats = selector.get_selection_summary(selected_papers, len(library_entries))
    logger.info(f"\nSelected {selection_stats['selected_papers']} papers from {selection_stats['total_papers']} total")
    logger.info("Papers to analyze:")
    for i, paper in enumerate(selected_papers, 1):
        year = paper.year or "n/d"
        authors = paper.authors[0] if paper.authors else "Unknown"
        if len(paper.authors or []) > 1:
            authors += " et al."
        logger.info(f"  {i}. {paper.citation_key} - {authors} ({year}): {paper.title[:60]}...")

    # Execute the operation
    log_header(f"EXECUTING {operation_display.upper()}")

    try:
        start_time = time.time()

        if operation == "review":
            result = llm_operations.generate_literature_review(selected_papers, focus="general")
        elif operation == "communication":
            result = llm_operations.generate_science_communication(selected_papers)
        elif operation == "compare":
            result = llm_operations.generate_comparative_analysis(selected_papers, aspect="methods")
        elif operation == "gaps":
            result = llm_operations.generate_research_gaps(selected_papers, domain="general")
        elif operation == "network":
            result = llm_operations.analyze_citation_network(selected_papers)

        # Save result
        output_dir = Path("literature/llm_outputs") / f"{operation}_outputs"
        output_path = llm_operations.save_result(result, output_dir)

        total_time = time.time() - start_time

        # Display results
        logger.info(f"\n{'=' * 70}")
        logger.info(f"{operation_display.upper()} COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Papers analyzed: {result.papers_used}")
        logger.info(f"Generation time: {result.generation_time:.1f}s")
        logger.info(f"Estimated tokens: {result.tokens_estimated}")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Total operation time: {total_time:.1f}s")

        display_file_locations()

        log_success(f"{operation_display} complete!")
        return 0

    except Exception as e:
        logger.error(f"LLM operation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
