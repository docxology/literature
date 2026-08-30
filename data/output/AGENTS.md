# AGENTS.md — `literature/data/output`

> Local-only path under `projects/ongoing/DataTools/` — matched by the
> root `.gitignore` rule `projects/*`; never commit. This `literature/`
> subtree is its own git repo — read-only for doc agents (no git writes).
> Repo-wide policy: see `/Volumes/external_drive/Git/template/projects/ongoing/AGENTS.md`.

## What this is

Aggregate analysis artifacts of the meta-analysis stage: bibliometric plots (`publications_by_year.png`, `venue_distribution.png`), embedding clustering/PCA outputs (`embedding_clusters*.png`, `pca_*.png`, CSV/JSON metrics), graphical abstracts, and summary reports (`meta_analysis_summary.md`, `pca_loadings_summary.md`).

## Layout (verified by direct listing, 2026-08-29)

```
author_contributions.png, cluster_quality_metrics.png, cluster_size_distribution.png, clustering_metrics.csv, clustering_metrics.json, dimensionality_analysis.png, embedding_clusters.json, embedding_clusters_2d.png, embedding_clusters_3d.png, embedding_coverage.png, embedding_outliers.png, embedding_quality.png, embedding_similarity_heatmap.png, embedding_similarity_matrix.csv, embedding_statis…  # truncated; ~100s of files
```

## Usage

- Written by menu categories 6.1/6.2 (meta-analysis without/with embeddings) of
  `../../run_literature.sh`; matplotlib/seaborn required (core deps in
  `../../pyproject.toml`).
- Human-readable entry points: `meta_analysis_summary.md`,
  `pca_loadings_summary.md`, `embedding_validation_report.json`.
- Regenerating overwrites these files; compare with care if you hand-tune plots.

## Gotchas

- Generated pipeline artifacts: safe to delete/regenerate via the pipeline
  (menu category 4/5/6 of `../../run_literature.sh`); filenames are
  content-derived  so a full re-run reproduces them.
- Never `git add` anything under `projects/ongoing/` (and never push the
  inner `literature/` repo from a doc lane).
