# Tool Analysis Module

Extraction and analysis of GitHub repositories, programming languages, and
software tools mentioned in the literature library's extracted paper text.

## Components

- **common.py**: shared GitHub-URL pattern, URL cleaning, citation-key year
  extraction, language/tool pattern tables, and context-window mention search
- **repo_extractor.py**: `extract_github_info` + markdown report of all repos
- **specific_tools.py**: specific libraries/frameworks/tools analysis
- **cooccurrence.py**: tool co-occurrence statistics and Jaccard similarity
- **language_trends.py**: language usage trends (CSV + Chart.js HTML)
- **over_time.py**: tools-and-languages-over-time summary and time series
- **run_comparison.py**: comparison of the current extraction state against
  the previous `tools_and_languages_summary.md` run

## Quick Start

```python
from infrastructure.literature.tool_analysis import run_specific_tools

run_specific_tools(output_dir="data/plots")
```

## Inputs / Outputs

- Input: `data/extracted_text/*.txt` (plus `data/library.json` metadata for
  the repository extractor)
- Output: CSV/HTML/markdown reports in `data/plots/` and
  `data/github_repositories.md`

## See Also

- [`../../scripts/README.md`](../../scripts/README.md) - thin orchestrators that call `run()` entry points
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview
