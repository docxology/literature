# Scripts Module

## Purpose

The scripts module contains orchestrator scripts that coordinate literature processing workflows. These are thin orchestrators that delegate business logic to infrastructure modules.

## Components

### Literature Search Orchestrator (`literature_search.py`)

Main orchestrator for literature search and summarization operations:

**Modes:**
- **Search mode**: Search for papers with configurable keywords and limits
- **Summarize mode**: Generate summaries for existing PDFs in library
- **Download mode**: Download PDFs for existing library entries
- **Extract text mode**: Extract text from PDFs
- **Meta-analysis mode**: Run meta-analysis on library
- **Cleanup mode**: Clean up library entries
- **LLM operations mode**: Multi-paper LLM operations (reviews, comparisons, etc.)

**Features:**
- Interactive keyword input
- Progress tracking
- Parallel summarization support
- Error handling
- Environment-based configuration

### Bash Utilities (`bash_utils.sh`)

Shared bash utilities for script orchestration:
- Logging functions
- Error handling
- Progress display
- Common operations

## Usage Examples

### Search for Papers

```bash
# Interactive search
python3 scripts/literature_search.py --search

# Search with specific keywords
python3 scripts/literature_search.py --search --keywords "machine learning,deep learning"

# Search with custom limit
python3 scripts/literature_search.py --search --limit 50 --keywords "optimization"
```

### Generate Summaries

```bash
# Summarize existing PDFs
python3 scripts/literature_search.py --summarize

# Search and summarize
python3 scripts/literature_search.py --search --summarize
```

### Download PDFs

```bash
# Download PDFs for existing entries
python3 scripts/literature_search.py --download-only
```

### Meta-Analysis

```bash
# Run standard meta-analysis on existing library (no embeddings, no Ollama required)
python3 scripts/literature_search.py --meta-analysis

# Run meta-analysis with embeddings on existing library (requires Ollama)
python3 scripts/literature_search.py --meta-analysis --with-embeddings
```

**Note**: Meta-analysis works only with existing library data. It does not search, download, or extract.
Use `--search-only`, `--download-only`, or `--extract-text` for those operations.

### Multi-paper LLM Operations

```bash
# Generate literature review
python3 scripts/literature_search.py --llm-operation review

# Science communication
python3 scripts/literature_search.py --llm-operation communication

# Comparative analysis
python3 scripts/literature_search.py --llm-operation compare
```

## Configuration

### Environment Variables

```bash
# Search settings
export LITERATURE_DEFAULT_LIMIT=25
export LITERATURE_SOURCES=arxiv,semanticscholar

# Summarization settings
export MAX_PARALLEL_SUMMARIES=1
export LLM_SUMMARIZATION_TIMEOUT=600

# Logging
export LOG_LEVEL=1  # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
```

## Output Structure

```
data/
├── references.bib              # BibTeX entries
├── library.json                # JSON index
├── summarization_progress.json # Progress tracking
├── pdfs/                       # Downloaded PDFs
├── summaries/                  # AI-generated summaries
├── extracted_text/            # Extracted PDF text
└── output/                    # Meta-analysis outputs
```

## Error Handling

The orchestrator provides:
- Graceful error handling
- Progress persistence
- Retry logic for transient failures
- Detailed error messages

## See Also

- [`README.md`](README.md) - Quick reference
- [`../infrastructure/literature/AGENTS.md`](../infrastructure/literature/AGENTS.md) - Literature module documentation
- [`literature_search.py`](literature_search.py) - Implementation


