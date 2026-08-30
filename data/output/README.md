# literature/data/output/

Aggregate analysis artifacts of the meta-analysis stage: bibliometric plots (`publications_by_year.png`, `venue_distribution.png`), embedding clustering/PCA outputs (`embedding_clusters*.png`, `pca_*.png`, CSV/JSON metrics), graphical abstracts, and summary reports (`meta_analysis_summary.md`, `pca_loadings_summary.md`).

## How it's produced & consumed

- Written by menu categories 6.1/6.2 (meta-analysis without/with embeddings) of
  `../../run_literature.sh`; matplotlib/seaborn required (core deps in
  `../../pyproject.toml`).
- Human-readable entry points: `meta_analysis_summary.md`,
  `pca_loadings_summary.md`, `embedding_validation_report.json`.
- Regenerating overwrites these files; compare with care if you hand-tune plots.

Part of the DataTools literature pipeline (local-only artifacts). Parent:
[`../README.md`](../README.md); agent-facing details in [`AGENTS.md`](AGENTS.md).
