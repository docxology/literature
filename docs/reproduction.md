# Offline metadata replay and manuscript build

Use a clean checkout and the [local setup](getting-started.md#installation).
From the project root, exercise the synthetic fixture and the native index tests.

```bash
uv run python scripts/replay_manuscript_fixture.py
uv run --extra dev pytest tests/infrastructure/literature/test_library_index.py
```

The fixture opens a temporary index and bibliography, reopens the index,
then stages the JSON files and BibTeX under `output/fixture/`. The count
receipt `output/reports/metadata-fixture.json` identifies the [source fixture](../tests/fixtures/bibliographic_metadata.json).
A chart and its registry appear under `output/figures/`. The fixture uses
invented records and does not read the repository corpus or contact services.

The normal library commands honor `LITERATURE_LIBRARY_INDEX`,
`LITERATURE_BIBTEX_FILE`, and `LITERATURE_DOWNLOAD_DIR`. For read-only
inspection of these generated records, use:

```bash
LITERATURE_LIBRARY_INDEX=output/fixture/library.json \
LITERATURE_BIBTEX_FILE=output/fixture/references.bib \
LITERATURE_DOWNLOAD_DIR=output/fixture/pdfs \
LITERATURE_USE_UNPAYWALL=false \
uv run python -m infrastructure.literature.core library list
```

The JSON export lists entries while the index stores them by key. The native
CLI can also report statistics or export the library. See the
[data formats reference](reference/data-formats.md). Search, downloading,
and summarization have independent dependencies and are outside this replay.

## Render with pinned Template

Set up a separate checkout following the
[pinned Template guide](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/RUN_GUIDE.md).
Verify that Pandoc and pandoc-crossref are compatible. Local verification
used Python 3.14, Pandoc 3.14, its matching cross-reference filter, and
XeLaTeX. After running the fixture, register the project from the Template root:

```bash
git switch --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
mkdir -p projects/working
ln -s /absolute/path/to/literature projects/working/literature
uv run --locked python scripts/pipeline/stage_03_render.py --project working/literature
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/literature
```

A detached project worktree directly at that slot also works. Inspect
`output/pdf/literature_combined.pdf` and `output/web/index.html` inside the
project, plus the validation and provenance receipts under `output/reports/`.
The combined outputs should contain the local-contract discussion, source
citations, and the synthetic fixture figure.
