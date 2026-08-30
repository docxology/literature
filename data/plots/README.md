# literature/data/plots/

Bibliometric plots and CSVs from the tool/language trend analysis scripts: language usage over time, specific-tool frequency, tool co-occurrence matrices, and their HTML dashboards (`language_trends.html`, `tools_and_languages_over_time.html`, …).

## How it's produced & consumed

- Written by `../../scripts/analyze_tools_and_languages_over_time.py`,
  `analyze_specific_tools.py`, `analyze_tool_cooccurrence.py`,
  `plot_language_trends.py` (see `tools_and_languages_summary.md` and
  `analysis_comparison.md` for the latest run's narrative).
- Inputs are the extracted-text corpus in `../extracted_text/`.

Part of the DataTools literature pipeline (local-only artifacts). Parent:
[`../README.md`](../README.md); agent-facing details in [`AGENTS.md`](AGENTS.md).
