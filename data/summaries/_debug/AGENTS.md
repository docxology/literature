# AGENTS.md — `literature/data/summaries/_debug`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. This `literature/`
> subtree is its own git repo — read-only for doc agents (no git writes).
> Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

Debug intermediates from summary-generation runs: captured prompt (`test_draft_prompt.txt`) and draft outputs (`test_draft.md`, `test_context_summary.md`). Not part of the summary corpus.

## Layout (verified by direct listing, 2026-08-29)

```
test_context_summary.md, test_draft.md, test_draft_prompt.txt
```

## Usage

- Written when summary generation runs with debug output enabled; useful to
  inspect exact prompts sent to Ollama when a summary looks wrong.
- Safe to clear; nothing downstream reads these files.

## Gotchas

- Generated pipeline artifacts: safe to delete/regenerate via the pipeline
  (menu category 4/5/6 of `../../run_literature.sh`); filenames are
  content-derived  so a full re-run reproduces them.
- Never `git add` anything under `projects/ongoing/` (and never push the
  inner `literature/` repo from a doc lane).
