# Literature toolkit contributor guidance

Read the [documentation index](docs/README.md) and the scoped `AGENTS.md`
before changing an implementation module. The toolkit runs standalone from
its repository root. The pinned Template renderer is used for the manuscript.

- `infrastructure/literature/` contains search, local library, PDF handling,
  workflow, summarization, and analysis code.
- `infrastructure/core/`, `infrastructure/llm/`, and
  `infrastructure/validation/` contain shared utilities and integrations.
- `scripts/` contains entry points and local reproduction helpers.
- `data/` holds the existing library, bibliography, and derived corpus
  artifacts. Use a separate fixture workspace to test edits and exports.
- [`docs/manuscript/`](docs/manuscript/README.md) is the authored paper source.
  Store generated fixture and render files under `output/`.

Use the [offline reproduction guide](docs/reproduction.md) to verify local
indexing, citation-key reuse, and exports. Keep fixture observations and
provider or corpus evaluations distinct. Follow the
[CLI reference](docs/reference/cli-reference.md) and
[configuration guide](docs/guides/configuration.md) when editing commands.
