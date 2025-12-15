# CLI Reference

Command-line interface reference.

## Literature CLI

### Search Command

```bash
python3 -m infrastructure.literature.core.cli search "query" [options]
```

**Options:**
- `--limit N` - Limit results per source
- `--sources SOURCES` - Comma-separated source list
- `--download` - Download PDFs automatically

**Examples:**
```bash
# Basic search
python3 -m infrastructure.literature.core.cli search "machine learning"

# Search with options
python3 -m infrastructure.literature.core.cli search "transformers" \
    --limit 20 \
    --sources arxiv,semanticscholar \
    --download
```

### Library Commands

#### List

```bash
python3 -m infrastructure.literature.core.cli library list [--limit N]
```

#### Stats

```bash
python3 -m infrastructure.literature.core.cli library stats
```

#### Export

```bash
python3 -m infrastructure.literature.core.cli library export [--output FILE]
```

## LLM CLI

### Query Command

```bash
python3 -m infrastructure.llm.cli query "prompt" [options]
```

**Options:**
- `--short` - Short response mode
- `--long` - Long response mode
- `--stream` - Stream output
- `--model MODEL` - Override model
- `--temperature FLOAT` - Temperature
- `--max-tokens INT` - Max tokens
- `--seed INT` - Random seed

**Examples:**
```bash
# Basic query
python3 -m infrastructure.llm.cli query "What is machine learning?"

# Short response
python3 -m infrastructure.llm.cli query --short "Define AI"

# With options
python3 -m infrastructure.llm.cli query \
    --temperature 0.0 \
    --seed 42 \
    "Test query"
```

### Check Command

```bash
python3 -m infrastructure.llm.cli check
```

### Models Command

```bash
python3 -m infrastructure.llm.cli models
```

### Template Command

```bash
# List templates
python3 -m infrastructure.llm.cli template --list

# Apply template
python3 -m infrastructure.llm.cli template TEMPLATE_NAME --input "text"
```

## Orchestrator Script

### Literature Search Script

```bash
python3 scripts/07_literature_search.py [options]
```

**Modes:**
- `--search` - Search for papers (orchestrated pipeline: search → download → summarize)
- `--search-only` - Search without downloading
- `--download-only` - Download PDFs only
- `--extract-text` - Extract text from PDFs
- `--summarize` - Generate summaries
- `--meta-analysis` - Run meta-analysis
- `--cleanup` - Clean up library
- `--llm-operation OPERATION` - Advanced LLM operations

**Options:**
- `--keywords KEYWORDS` - Comma-separated keywords for search (prompts if not provided)
- `--limit N` - Papers per keyword (default: from LITERATURE_DEFAULT_LIMIT or 25)
- `--retry-failed` - Retry previously failed downloads (default: False - failed downloads are skipped)
- `--clear-pdfs` - Clear all PDFs before download (default: False, incremental/additive)
- `--clear-summaries` - Clear all summaries before generation (default: False, incremental/additive)
- `--clear-library` - Clear library index before operations (requires confirmation, default: False)
- `--paper-config PATH` - Path to YAML config file for paper selection (default: data/paper_selection.yaml)

**Note:** Sources are configured via `LITERATURE_SOURCES` environment variable, not via CLI flag. See [Configuration Guide](../guides/configuration.md) for details.

**Failed Download Behavior:**

By default, papers with previously failed downloads are **automatically skipped** to avoid wasting time on access-restricted or unavailable papers. The system tracks failures in `data/failed_downloads.json`.

- **Default (no flag)**: Failed downloads are skipped with a message: `⊘ Skipping {citation_key}: previously failed ({reason}). Use --retry-failed to retry.`
- **With `--retry-failed`**: Previously failed downloads are retried, useful for network errors or temporary issues

This flag works with:
- `--search` - Full search and download workflow
- `--download-only` - Download existing bibliography entries
- `--meta-analysis` - Meta-analysis workflow

**Examples:**
```bash
# Search
python3 scripts/07_literature_search.py --search --keywords "machine learning"

# Summarize
python3 scripts/07_literature_search.py --summarize

# Meta-analysis
python3 scripts/07_literature_search.py --meta-analysis --keywords "optimization"
```

## See Also

- **[API Reference](api-reference.md)** - Python API documentation
- **[Configuration Guide](../guides/configuration.md)** - Configuration options

