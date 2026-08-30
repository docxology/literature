# AGENTS.md — `literature/data/extracted_text`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. This `literature/`
> subtree is its own git repo — read-only for doc agents (no git writes).
> Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

Plaintext extracted from downloaded paper PDFs, one `.txt` per paper, named by citation key (e.g. `baltieri2020bayesian.txt`). This is the corpus consumed by LLM summarization, embeddings, and meta-analysis.

## Layout (verified by direct listing, 2026-08-29)

```
aguilera2021how.txt, alanova2025lightweight.txt, albarracin2022mapping.txt, albarracin2023designing.txt, albarracin2024modeling.txt, albarracin2025physics.txt, albarracin2025resilience.txt, allahverdyan2014active.txt, amorese2024online.txt, annabi2020autonomous.txt, annabi2021bidirectional.txt, annicchiarico2024bayesian.txt, annicchiarico2025active.txt, asadi2022quantum.txt, baioumy2020active.t…  # truncated; ~100s of files
```

## Usage

- Written by menu category 4.2 (`Extract text only`) of `../../run_literature.sh`.
- Consumers: category 5 (LLM analysis), category 6 (meta-analysis/embeddings),
  and `scripts/literature_search.py`.
- Extraction backend order and fallbacks are configured in the PDF extraction
  module (`infrastructure/literature/pdf/`); optional extras (`pdfplumber`,
  `beautifulsoup4`) improve quality (see pyproject `[project.optional-dependencies].pdf`).

## Gotchas

- Generated pipeline artifacts: safe to delete/regenerate via the pipeline
  (menu category 4/5/6 of `../../run_literature.sh`); filenames are
  content-derived (SHA-style keys are used only in `embeddings/`) so a full re-run reproduces them.
- Never `git add` anything under `projects/ongoing/` (and never push the
  inner `literature/` repo from a doc lane).
