# AGENTS.md — `literature/tests/infrastructure/validation`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. The `literature/`
> subtree is its own git repo — read-only for doc agents (no git writes).
> Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

Tests for `infrastructure/validation/pdf_validator.py` — PDF validation and text extraction (`extract_text_from_pdf`, `PDFValidationError`). Built with real generated PDFs (reportlab-style fixture PDFs), no mocks, per the suite policy.

## Layout (verified by direct listing, 2026-08-29)

```
test_pdf_validator.py
```

## How to run

From the `literature/` repo root:

```bash
pytest tests/infrastructure/validation -v
```

## Gotchas

- These tests import from `infrastructure.validation` / `infrastructure.literature.summarization`
  via a `sys.path.insert` of the repo root — run pytest from the repo root so the
  import resolves.
- Never `git add` anything under `projects/ongoing/`.
