# AGENTS.md — `literature/tests/infrastructure/literature/summarization`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. The `literature/`
> subtree is its own git repo — read-only for doc agents (no git writes).
> Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

Tests for `infrastructure/literature/summarization` output validation: `test_output_validation.py` covers `MethodsToolsValidator` (methods/tools output validation, quote validation, summary-quality checks, output-size anomaly detection); `test_regressions.py` pins known past summarization failure cases so they cannot silently recur.

## Layout (verified by direct listing, 2026-08-29)

```
test_output_validation.py, test_regressions.py
```

## How to run

From the `literature/` repo root:

```bash
pytest tests/infrastructure/literature/summarization -v
```

## Gotchas

- These tests import from `infrastructure.validation` / `infrastructure.literature.summarization`
  via a `sys.path.insert` of the repo root — run pytest from the repo root so the
  import resolves.
- Never `git add` anything under `projects/ongoing/`.
