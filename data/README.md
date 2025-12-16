# literature/ - Academic Literature Database

Central repository for academic papers, PDFs, references, and research summaries.

## Quick Start

### Search and Download Papers

```bash
# Search across sources and download PDFs
python3 -m infrastructure.literature.core.cli search "machine learning" --download --limit 5

# Interactive menu
./run_literature.sh

# Or directly via Python
python3 scripts/literature_search.py --search     # Search for papers
python3 scripts/literature_search.py --summarize  # Generate summaries
```

### View Library

```bash
# Show statistics
python3 -m infrastructure.literature.core.cli library stats

# List recent papers
python3 -m infrastructure.literature.core.cli library list --limit 10
```

## Directory Contents

- **`library.json`** - Paper metadata index (499 papers)
- **`references.bib`** - BibTeX bibliography for citations
- **`pdfs/`** - Downloaded full-text PDFs
- **`summaries/`** - AI-generated paper summaries
- **`failed_downloads.json`** - Retry queue for failed downloads

## Usage in Manuscripts

### Cite Papers

```markdown
Recent advances in transformers \cite{vaswani2017attention}
show significant improvements in NLP tasks.
```

### Bibliography Generation

Papers are automatically included in `99_references.md` during PDF compilation.

## File Formats

- **PDFs**: Named by citation key (e.g., `smith2024ml.pdf`)
- **Summaries**: Markdown format with key findings and methodology
- **Library**: JSON with complete metadata and download status
- **Bibliography**: Standard BibTeX format

## Statistics

- **Total Papers**: 499
- **PDF Downloads**: 91% success rate
- **DOI Coverage**: 87%
- **Sources**: arXiv, Semantic Scholar, CrossRef, PubMed

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation
- [`infrastructure/literature/README.md`](../infrastructure/literature/README.md) - CLI reference
