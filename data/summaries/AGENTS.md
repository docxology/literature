# AGENTS.md — `literature/data/summaries`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. This `literature/`
> subtree is its own git repo — read-only for doc agents (no git writes).
> Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

LLM-generated paper summaries (Markdown), produced by the local-Ollama summarization workflow. `_debug/` holds prompt/draft intermediates from test and debug runs.

## Layout (verified by direct listing, 2026-08-29)

```
_debug, paper1_summary.md, paper2_summary.md
```

## Usage

- Written by menu category 5.1 (LLM operations: summarize, literature review,
  comparisons) of `../../run_literature.sh`; requires `ollama serve`.
- Progress tracking: `../summarization_progress.json`.
- `_debug/` contents (e.g. `test_draft_prompt.txt`) are debugging artifacts,
  not summaries — ignore unless diagnosing prompt construction.

## Gotchas

- Generated pipeline artifacts: safe to delete/regenerate via the pipeline
  (menu category 4/5/6 of `../../run_literature.sh`); filenames are
  content-derived  so a full re-run reproduces them.
- Never `git add` anything under `projects/ongoing/` (and never push the
  inner `literature/` repo from a doc lane).
