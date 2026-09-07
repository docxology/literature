# AGENTS: `scripts/` — Thin Orchestrator Scripts

## Thin-Orchestrator Contract

- Scripts carry **only** orchestration: path bootstrap
  (`sys.path.insert(0, str(Path(__file__).parent.parent))`), argparse, and a
  single delegated call into an `infrastructure/` entrypoint.
- Business, data, plot, and analysis logic lives in
  `infrastructure/literature/` (importable and tested) — **never** inline in a
  script. Shared logic lives once in `infrastructure/`; scripts must not
  copy-paste it (e.g. the GitHub-URL pattern lives only in
  `infrastructure/literature/tool_analysis/common.py`).
- Scripts that import other scripts are prohibited: promote the shared logic
  into `infrastructure/` and have both call it.
- Bash modules in this directory are sourced libraries for `run_literature.sh`
  (`bash_utils.sh`, `literature_*.sh`); they must stay delegation layers over
  `infrastructure.literature` / `infrastructure.llm` and document their
  purpose in their header block.
- Reusable script behavior must be covered by tests in `tests/` (see
  `tests/infrastructure/literature/test_tool_analysis.py` for the analysis
  scripts).

## Script Inventory

| Script | Delegates to | Inputs | Outputs |
|--------|--------------|--------|---------|
| `literature_search.py` | `infrastructure.literature.workflow.orchestrator`, `infrastructure.literature.summarization`, `infrastructure.llm` | env config, `data/library.json` | `data/references.bib`, `data/library.json`, `data/pdfs/`, `data/summaries/`, `data/extracted_text/`, `data/output/` |
| `analyze_specific_tools.py` | `tool_analysis/specific_tools.py::run` | `data/extracted_text/*.txt` | `data/plots/specific_tools_analysis.txt`, `specific_tools_by_paper.csv`, `specific_tools_frequency.csv`, `specific_tools.html` |
| `analyze_tool_cooccurrence.py` | `tool_analysis/cooccurrence.py::run` | `data/extracted_text/*.txt` | `data/plots/tool_cooccurrence_analysis.txt`, `tool_cooccurrence_matrix.csv`, `tool_combinations.csv`, `tool_cooccurrence.html` |
| `analyze_tools_and_languages_over_time.py` | `tool_analysis/over_time.py::run` | `data/extracted_text/*.txt` | `data/plots/tools_and_languages_summary.md`, `languages_over_time.csv`, `tools_over_time.csv`, `tools_and_languages_over_time.html` |
| `plot_language_trends.py` | `tool_analysis/language_trends.py::run` | `data/extracted_text/*.txt` | `data/plots/language_usage_by_year.csv`, `language_usage_long.csv`, `language_trends.html`, `language_statistics.txt` |
| `extract_github_repos.py` | `tool_analysis/repo_extractor.py::run` | `data/extracted_text/*.txt`, `data/library.json` | `data/github_repositories.md` |
| `compare_analysis_runs.py` | `tool_analysis/run_comparison.py::run` | `data/extracted_text/*.txt`, `data/plots/tools_and_languages_summary.md` | `data/plots/analysis_comparison.md` |
| `re_extract_incomplete.py` | `infrastructure.literature.pdf.re_extract::check_and_re_extract_incomplete_files` | `data/library.json`, `data/pdfs/` | updated `data/extracted_text/` |
| `bash_utils.sh` | (sourced library) color codes, logging, choice parsing | — | terminal output |
| `literature_helpers.sh` | (sourced library) timed command execution, env validation, Python invocation, Ollama checks | env config | terminal output |
| `literature_errors.sh` | (sourced library) error context extraction, safe command execution | — | terminal output |
| `literature_operations.sh` | (sourced library) pipeline/meta-analysis/individual operations; delegates to `infrastructure.literature` + `infrastructure.llm` via `literature_helpers.sh` Python invocations | env config, `data/` | library, PDFs, summaries, plots |
| `literature_menu.sh` | (sourced library) interactive menu rendering and routing | — | terminal output |
| `literature_cli.sh` | (sourced library) non-interactive CLI mode and help | argv | terminal output |
| `literature_testing.sh` | (sourced library) pytest/coverage execution and result parsing | `tests/` | terminal output, coverage |

(`tool_analysis/*` is shorthand for
`infrastructure/literature/tool_analysis/`.)

## Design Contract

- No script may implement analysis, parsing, aggregation, or plotting logic
  inline; add it to the owning `infrastructure/` module and delegate.
- Analysis entrypoints are the `run(...)` functions of
  `infrastructure/literature/tool_analysis/*.py`; scripts pass CLI flags
  through as parameters.
- Defaults preserve the historical behavior: input `data/extracted_text/`,
  outputs under `data/plots/` (repository report at
  `data/github_repositories.md`).
