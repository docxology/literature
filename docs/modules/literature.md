# Literature Module

Literature search and management functionality.

## Overview

The literature module provides tools for searching scientific papers, downloading PDFs, managing references, and generating LLM-powered summaries.

## Key Components

### Core (`core/`)

Main interface for literature operations:
- LiteratureSearch class
- Configuration management
- CLI interface

### Sources (`sources/`)

API adapters for academic databases:
- arXiv
- Semantic Scholar
- PubMed
- CrossRef
- OpenAlex
- DBLP
- bioRxiv/medRxiv
- Europe PMC
- Unpaywall (open access)

### PDF (`pdf/`)

PDF handling:
- Downloading with retry logic
- Text extraction
- Fallback strategies
- HTML parsing

### Library (`library/`)

Library management:
- JSON-based indexing
- BibTeX generation
- Statistics and reporting
- Cleanup operations

### Summarization (`summarization/`)

LLM-powered summarization:
- Multi-stage generation
- Quality validation
- Context extraction
- Progress tracking

### Meta-Analysis (`meta_analysis/`)

Analysis tools:
- Temporal trends (publication year analysis)
- Keyword evolution (keyword frequency over time)
- PCA analysis (text feature extraction and clustering)
- Metadata visualization (venue, author, citation distributions)
- Graphical abstract generation

### Workflow (`workflow/`)

Workflow orchestration:
- LiteratureWorkflow for multi-paper operations
- Progress tracking with resumability
- Search orchestrator
- Operation modules (search, download, cleanup, meta-analysis, LLM operations)

### Analysis (`analysis/`)

Paper analysis tools:
- PaperAnalyzer for structure/content analysis
- DomainDetector for automatic domain detection
- ContextBuilder for rich context generation

### HTML Parsers (`html_parsers/`)

Publisher-specific PDF URL extraction:
- Elsevier, Springer, IEEE, ACM, Wiley parsers
- Generic parser fallback
- Modular parser system

### Reporting (`reporting/`)

Reporting:
- LiteratureReporter for multi-format export
- JSON, CSV, HTML export formats
- Library statistics and summaries

### LLM Operations (`llm/`)

Multi-paper LLM operations:
- LiteratureLLMOperations for multi-paper synthesis
- PaperSelector for configurable filtering
- Literature review generation
- Science communication narratives
- Comparative analysis
- Research gap identification

## Usage Examples

### Search

```python
from infrastructure.literature import LiteratureSearch

searcher = LiteratureSearch()
papers = searcher.search("machine learning", limit=10)
```

### Add to Library

```python
for paper in papers:
    citation_key = searcher.add_to_library(paper)
    searcher.download_paper(paper)
```

### Summarization

```python
from infrastructure.literature.summarization import SummarizationEngine
from infrastructure.llm import LLMClient

llm_client = LLMClient()
engine = SummarizationEngine(llm_client)

result = engine.summarize_paper(
    result=search_result,
    pdf_path=Path("data/pdfs/paper.pdf")
)
```

### Workflow Operations

```python
from infrastructure.literature.workflow import LiteratureWorkflow

workflow = LiteratureWorkflow()

# Search and add papers
result = workflow.search_and_add(
    keywords=["active inference"],
    limit=10
)

# Download PDFs
download_result = workflow.download_pdfs()
```

### Meta-Analysis

```python
from infrastructure.literature.meta_analysis import (
    DataAggregator,
    create_publication_timeline_plot,
    create_keyword_frequency_plot
)

# Create aggregator
aggregator = DataAggregator()

# Publication timeline
create_publication_timeline_plot()

# Keyword analysis
keyword_data = extract_keywords_over_time()
create_keyword_frequency_plot(keyword_data, top_n=20)
```

### LLM Operations

```python
from infrastructure.literature.llm import (
    LiteratureLLMOperations,
    PaperSelector
)

# Generate literature review
operations = LiteratureLLMOperations()
result = operations.generate_literature_review(
    papers=selected_papers,
    focus="methods"
)
```

### Reporting

```python
from infrastructure.literature.reporting import LiteratureReporter

reporter = LiteratureReporter()
reporter.export_library_report(
    output_dir=Path("output"),
    formats=["json", "csv", "html"]
)
```

## See Also

- **[Literature Module Documentation](../../infrastructure/literature/AGENTS.md)** - Technical documentation
- **[Searching Papers Guide](../guides/search-papers.md)** - Search guide
- **[Summarizing Papers Guide](../guides/summarize-papers.md)** - Summarization guide
- **[LLM Operations Guide](../guides/llm-operations.md)** - Multi-paper LLM operations
- **[Meta-Analysis Guide](../guides/meta-analysis.md)** - Meta-analysis guide
- **[Modules AGENTS.md](AGENTS.md)** - Module documentation standards

