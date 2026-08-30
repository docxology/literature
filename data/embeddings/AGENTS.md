# AGENTS.md — `literature/data/embeddings`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. This `literature/`
> subtree is its own git repo — read-only for doc agents (no git writes).
> Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

Per-paper embedding vectors stored as one JSON per paper, keyed by a content hash (e.g. `0003ff1c….json`). Each file is `{"embedding": [floats], "model": "embeddinggemma", "dimension": 768}` (verified against an on-disk file). Feeds clustering, similarity, and PCA outputs in `output/`.

## Layout (verified by direct listing, 2026-08-29)

```
.gitkeep, 0003ff1c7a3f1ced798841dd3c9db949152b5f715e37842c79cbc8ddf635497f.json, 002c15081baf4106d521585adaefcac8d22ace52dc4f7b6d78a5332178189cf4.json, 0046e5199b69ce9adaa4d12cd4f74d25f98e0ebb3a3121c1edeec5f2feabd0fa.json, 0051afab735978acba5b7175042a260dd09431eff0753206ede08b5ca420425f.json, 0066efaa80d6550fc1fb2566b54a314d79c6db32191612be59eaf576a0c6ce65.json, 006d6aca38be304bc571a766547287bf…  # truncated; ~100s of files
```

## Usage

- Produced during meta-analysis with embeddings (menu category 6.2 of
  `../../run_literature.sh`; local Ollama, model `embeddinggemma`).
- Consumed by `infrastructure/literature/llm/embeddings/` code paths and the
  clustering/PCA analysis scripts; aggregates land in `../output/embeddings.json`
  and `../output/embedding_*.{csv,json,png}`.

## Gotchas

- Generated pipeline artifacts: safe to delete/regenerate via the pipeline
  (menu category 4/5/6 of `../../run_literature.sh`); filenames are
  content-derived  so a full re-run reproduces them.
- Never `git add` anything under `projects/ongoing/` (and never push the
  inner `literature/` repo from a doc lane).
