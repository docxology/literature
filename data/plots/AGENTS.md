# AGENTS.md — `literature/data/plots`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. This `literature/`
> subtree is its own git repo — read-only for doc agents (no git writes).
> Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

Bibliometric plots and CSVs from the tool/language trend analysis scripts: language usage over time, specific-tool frequency, tool co-occurrence matrices, and their HTML dashboards (`language_trends.html`, `tools_and_languages_over_time.html`, …).

## Layout (verified by direct listing, 2026-08-29)

```
analysis_comparison.md, language_statistics.txt, language_trends.html, language_usage_by_year.csv, language_usage_long.csv, languages_over_time.csv, specific_tools.html, specific_tools_analysis.txt, specific_tools_by_paper.csv, specific_tools_frequency.csv, tool_combinations.csv, tool_cooccurrence.html, tool_cooccurrence_analysis.txt, tool_cooccurrence_matrix.csv, tools_and_languages_over_time.…  # truncated; ~100s of files
```

## Usage

- Written by `../../scripts/analyze_tools_and_languages_over_time.py`,
  `analyze_specific_tools.py`, `analyze_tool_cooccurrence.py`,
  `plot_language_trends.py` (see `tools_and_languages_summary.md` and
  `analysis_comparison.md` for the latest run's narrative).
- Inputs are the extracted-text corpus in `../extracted_text/`.

## Gotchas

- Generated pipeline artifacts: safe to delete/regenerate via the pipeline
  (menu category 4/5/6 of `../../run_literature.sh`); filenames are
  content-derived  so a full re-run reproduces them.
- Never `git add` anything under `projects/ongoing/` (and never push the
  inner `literature/` repo from a doc lane).
