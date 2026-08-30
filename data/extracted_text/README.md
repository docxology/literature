# literature/data/extracted_text/

Plaintext extracted from downloaded paper PDFs, one `.txt` per paper, named by citation key (e.g. `baltieri2020bayesian.txt`). This is the corpus consumed by LLM summarization, embeddings, and meta-analysis.

## How it's produced & consumed

- Written by menu category 4.2 (`Extract text only`) of `../../run_literature.sh`.
- Consumers: category 5 (LLM analysis), category 6 (meta-analysis/embeddings),
  and `scripts/literature_search.py`.
- Extraction backend order and fallbacks are configured in the PDF extraction
  module (`infrastructure/literature/pdf/`); optional extras (`pdfplumber`,
  `beautifulsoup4`) improve quality (see pyproject `[project.optional-dependencies].pdf`).

Part of the DataTools literature pipeline (local-only artifacts). Parent:
[`../README.md`](../README.md); agent-facing details in [`AGENTS.md`](AGENTS.md).
