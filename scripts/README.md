# Scripts Module

Thin orchestrator scripts for the literature repository. Every Python script is
orchestration only: path bootstrap, argument parsing, and a single delegated
call into an `infrastructure/` entrypoint. Testable logic lives in
`infrastructure/literature/` (analysis logic in
[`infrastructure/literature/tool_analysis/`](../infrastructure/literature/tool_analysis/)).

## Inventory

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `literature_search.py` | Search / download / extract / summarize / meta-analysis / LLM operations | `infrastructure.literature.workflow.orchestrator`, `infrastructure.literature.summarization`, `infrastructure.llm` | `python3 scripts/literature_search.py --search --keywords "machine learning"` |
| `analyze_specific_tools.py` | Library/framework/tool usage analysis across repos | `infrastructure.literature.tool_analysis.specific_tools` | `python3 scripts/analyze_specific_tools.py` |
| `analyze_tool_cooccurrence.py` | Tool co-occurrence and Jaccard similarity analysis | `infrastructure.literature.tool_analysis.cooccurrence` | `python3 scripts/analyze_tool_cooccurrence.py` |
| `analyze_tools_and_languages_over_time.py` | Time-series summary of tools and languages | `infrastructure.literature.tool_analysis.over_time` | `python3 scripts/analyze_tools_and_languages_over_time.py` |
| `plot_language_trends.py` | Language usage trend charts and CSVs | `infrastructure.literature.tool_analysis.language_trends` | `python3 scripts/plot_language_trends.py` |
| `extract_github_repos.py` | GitHub repository extraction and markdown report | `infrastructure.literature.tool_analysis.repo_extractor` | `python3 scripts/extract_github_repos.py` |
| `compare_analysis_runs.py` | Compare current extraction state with the previous run | `infrastructure.literature.tool_analysis.run_comparison` | `python3 scripts/compare_analysis_runs.py` |
| `re_extract_incomplete.py` | Re-extract incomplete HTML text files | `infrastructure.literature.pdf.re_extract` | `python3 scripts/re_extract_incomplete.py` |
| `bash_utils.sh` | Shared bash utilities (color, logging, parsing); sourced, not executed | sourced by `run_literature.sh` | source only |
| `literature_helpers.sh` | Command execution with timeout, env validation, Ollama checks | sourced by `run_literature.sh` | source only |
| `literature_errors.sh` | Error context extraction and safe command execution | sourced by `run_literature.sh` | source only |
| `literature_operations.sh` | Literature workflow operations: full pipeline, meta-analysis, individual operations (search, download, extract, summarize, cleanup, LLM); delegates to `infrastructure.literature` / `infrastructure.llm` via Python invocations | sourced by `run_literature.sh` / `literature_menu.sh` | source only |
| `literature_menu.sh` | Interactive menu rendering and choice routing | sourced by `run_literature.sh` | source only |
| `literature_cli.sh` | Non-interactive CLI mode and help | sourced by `run_literature.sh` | source only |
| `literature_testing.sh` | Test-suite execution and pytest/coverage output parsing | sourced by `run_literature.sh` | source only |

## Analysis Scripts

The six repository/tool analysis scripts share one logic home:
`infrastructure/literature/tool_analysis/`. Common extraction (GitHub-URL
pattern, URL cleaning, citation-key year extraction, language/tool pattern
tables) lives in `tool_analysis/common.py`; each script calls the `run()`
entrypoint of its module.

Common flags: `--extracted-text-dir` (default `data/extracted_text`) and
`--output-dir` (default `data/plots`); `extract_github_repos.py` also accepts
`--library-file` (default `data/library.json`).

### `analyze_specific_tools.py`

```bash
python3 scripts/analyze_specific_tools.py
```

Module logic: `infrastructure/literature/tool_analysis/specific_tools.py` —
tool-usage counting, text/CSV report, Chart.js HTML.

Outputs (`data/plots/`): `specific_tools_analysis.txt`,
`specific_tools_by_paper.csv`, `specific_tools_frequency.csv`,
`specific_tools.html`.

### `analyze_tool_cooccurrence.py`

```bash
python3 scripts/analyze_tool_cooccurrence.py
```

Module logic: `infrastructure/literature/tool_analysis/cooccurrence.py` —
pair/triplet co-occurrence, Jaccard similarity, co-occurrence matrix CSV.

Outputs (`data/plots/`): `tool_cooccurrence_analysis.txt`,
`tool_cooccurrence_matrix.csv`, `tool_combinations.csv`,
`tool_cooccurrence.html`.

### `analyze_tools_and_languages_over_time.py`

```bash
python3 scripts/analyze_tools_and_languages_over_time.py
```

Module logic: `infrastructure/literature/tool_analysis/over_time.py` —
per-year language/tool series, markdown summary with trends.

Outputs (`data/plots/`): `tools_and_languages_summary.md`,
`languages_over_time.csv`, `tools_over_time.csv`,
`tools_and_languages_over_time.html`.

### `plot_language_trends.py`

```bash
python3 scripts/plot_language_trends.py
```

Module logic: `infrastructure/literature/tool_analysis/language_trends.py` —
year/language matrix, wide/long CSVs, Chart.js HTML, summary statistics.

Outputs (`data/plots/`): `language_usage_by_year.csv`,
`language_usage_long.csv`, `language_trends.html`, `language_statistics.txt`.

### `extract_github_repos.py`

```bash
python3 scripts/extract_github_repos.py
```

Module logic: `infrastructure/literature/tool_analysis/repo_extractor.py` —
repo URL extraction, library metadata enrichment, markdown report.

Outputs: `data/github_repositories.md`.

### `compare_analysis_runs.py`

```bash
python3 scripts/compare_analysis_runs.py
```

Module logic: `infrastructure/literature/tool_analysis/run_comparison.py` —
current-state scan, previous-summary parsing, comparison markdown.

Outputs (`data/plots/`): `analysis_comparison.md`.

## See Also

- [`AGENTS.md`](AGENTS.md) - thin-orchestrator contract and inventory
- [`../infrastructure/literature/tool_analysis/README.md`](../infrastructure/literature/tool_analysis/README.md) - analysis module
- [`../infrastructure/literature/AGENTS.md`](../infrastructure/literature/AGENTS.md) - Literature module documentation
