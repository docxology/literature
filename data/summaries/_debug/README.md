# literature/data/summaries/_debug/

Debug intermediates from summary-generation runs: captured prompt (`test_draft_prompt.txt`) and draft outputs (`test_draft.md`, `test_context_summary.md`). Not part of the summary corpus.

## How it's produced & consumed

- Written when summary generation runs with debug output enabled; useful to
  inspect exact prompts sent to Ollama when a summary looks wrong.
- Safe to clear; nothing downstream reads these files.

Part of the DataTools literature pipeline (local-only artifacts). Parent:
[`../README.md`](../README.md); agent-facing details in [`AGENTS.md`](AGENTS.md).
