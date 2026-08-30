# literature/data/summaries/

LLM-generated paper summaries (Markdown), produced by the local-Ollama summarization workflow. `_debug/` holds prompt/draft intermediates from test and debug runs.

## How it's produced & consumed

- Written by menu category 5.1 (LLM operations: summarize, literature review,
  comparisons) of `../../run_literature.sh`; requires `ollama serve`.
- Progress tracking: `../summarization_progress.json`.
- `_debug/` contents (e.g. `test_draft_prompt.txt`) are debugging artifacts,
  not summaries — ignore unless diagnosing prompt construction.

Part of the DataTools literature pipeline (local-only artifacts). Parent:
[`../README.md`](../README.md); agent-facing details in [`AGENTS.md`](AGENTS.md).
