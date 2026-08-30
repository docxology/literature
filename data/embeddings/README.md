# literature/data/embeddings/

Per-paper embedding vectors stored as one JSON per paper, keyed by a content hash (e.g. `0003ff1c….json`). Each file is `{"embedding": [floats], "model": "embeddinggemma", "dimension": 768}` (verified against an on-disk file). Feeds clustering, similarity, and PCA outputs in `output/`.

## How it's produced & consumed

- Produced during meta-analysis with embeddings (menu category 6.2 of
  `../../run_literature.sh`; local Ollama, model `embeddinggemma`).
- Consumed by `infrastructure/literature/llm/embeddings/` code paths and the
  clustering/PCA analysis scripts; aggregates land in `../output/embeddings.json`
  and `../output/embedding_*.{csv,json,png}`.

Part of the DataTools literature pipeline (local-only artifacts). Parent:
[`../README.md`](../README.md); agent-facing details in [`AGENTS.md`](AGENTS.md).
