"""Repository/tool analysis for the literature library.

Importable business logic behind the thin analysis orchestrators in
``scripts/``: GitHub-URL extraction, tool/language mention detection,
co-occurrence statistics, time-series trends, and run comparison reports.
"""

from infrastructure.literature.tool_analysis.common import (
    DEFAULT_EXTRACTED_TEXT_DIR,
    DEFAULT_LIBRARY_FILE,
    DEFAULT_PLOTS_DIR,
    clean_url,
    code_patterns,
    extract_year_from_citation_key,
    find_code_mentions,
    find_github_urls,
    find_items_in_content,
    github_pattern,
    languages,
)
from infrastructure.literature.tool_analysis.cooccurrence import run as run_tool_cooccurrence
from infrastructure.literature.tool_analysis.language_trends import run as run_language_trends
from infrastructure.literature.tool_analysis.over_time import run as run_tools_and_languages_over_time
from infrastructure.literature.tool_analysis.repo_extractor import run as run_repo_extractor
from infrastructure.literature.tool_analysis.run_comparison import run as run_analysis_comparison
from infrastructure.literature.tool_analysis.specific_tools import run as run_specific_tools

__all__ = [
    "DEFAULT_EXTRACTED_TEXT_DIR",
    "DEFAULT_LIBRARY_FILE",
    "DEFAULT_PLOTS_DIR",
    "clean_url",
    "code_patterns",
    "extract_year_from_citation_key",
    "find_code_mentions",
    "find_github_urls",
    "find_items_in_content",
    "github_pattern",
    "languages",
    "run_analysis_comparison",
    "run_language_trends",
    "run_repo_extractor",
    "run_specific_tools",
    "run_tool_cooccurrence",
    "run_tools_and_languages_over_time",
]
