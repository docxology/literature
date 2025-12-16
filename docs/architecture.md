# System Architecture

Complete architecture overview of the Literature Search and Management System.

## Overview

The Literature Search and Management System is a **standalone repository** for academic literature search, PDF management, reference tracking, and AI-powered paper summarization. It provides a complete workflow from paper discovery to analysis and citation management.

## Architecture Principles

### Standalone Design

The system is completely independent and self-contained:
- **No dependencies** on external template or manuscript systems
- **Duplicated shared infrastructure** (`infrastructure/core/`, `infrastructure/llm/`) for independence
- **Separate bibliography** (`data/references.bib`) from any manuscript system
- **Complete test suite** for all functionality

### Thin Orchestrator Pattern

Business logic is implemented in infrastructure modules, with thin orchestrator scripts coordinating workflows:
- **Infrastructure modules**: Contain all business logic
- **Orchestrator scripts**: Coordinate module interactions
- **Clear separation**: Logic vs. orchestration

### Modular Architecture

The system is organized into logical, independent modules:
- **Core**: Foundation utilities
- **LLM**: Local LLM integration
- **Literature**: Literature search and management
- **Validation**: PDF validation and text extraction

## System Layers

```
┌─────────────────────────────────────────────────────────┐
│                    Scripts Layer                         │
│  (Thin orchestrators: literature_search.py)          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                Infrastructure Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐         │
│  │   Core   │  │   LLM    │  │  Literature  │         │
│  └──────────┘  └──────────┘  └──────────────┘         │
│  ┌──────────┐                                          │
│  │Validation│                                          │
│  └──────────┘                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                            │
│  (library.json, references.bib, PDFs, summaries)        │
└─────────────────────────────────────────────────────────┘
```

## Core Modules

### Infrastructure Core (`infrastructure/core/`)

Foundation utilities used across all modules:
- **Logging**: Unified Python logging system
- **Exceptions**: Comprehensive exception hierarchy
- **Configuration**: YAML and environment variable management
- **Progress**: Progress tracking and visual indicators
- **Checkpoint**: Pipeline checkpoint management
- **Retry**: Retry logic with exponential backoff
- **Performance**: Performance monitoring
- **Environment**: Environment setup and validation

### LLM Module (`infrastructure/llm/`)

Local LLM integration for research assistance:
- **Core Client**: LLMClient for Ollama interaction
- **Configuration**: LLMConfig and GenerationOptions
- **Context Management**: ConversationContext
- **Templates**: Research prompt templates
- **Validation**: Output validation
- **Review System**: Manuscript review generation
- **CLI**: Command-line interface
- **Prompts**: Composable prompt fragment system

### Literature Module (`infrastructure/literature/`)

Literature search and management:
- **Core**: LiteratureSearch main interface
- **Sources**: API adapters (arXiv, Semantic Scholar, PubMed, CrossRef, OpenAlex, DBLP, bioRxiv, Europe PMC, Unpaywall)
- **PDF**: PDF downloading and extraction
- **Library**: Library indexing and BibTeX generation
- **Summarization**: AI-powered paper summarization
- **Meta-Analysis**: Analysis tools and visualizations
- **Workflow**: Workflow orchestration with progress tracking
- **Analysis**: Paper analysis and domain detection
- **HTML Parsers**: Publisher-specific PDF URL extraction
- **Reporting**: Comprehensive reporting with multi-format export
- **LLM Operations**: Advanced LLM operations for multi-paper synthesis

### Validation Module (`infrastructure/validation/`)

PDF validation and text extraction:
- **PDF Text Extraction**: Multi-library support (pdfplumber, pypdf, PyPDF2)
- **Automatic Fallback**: Tries libraries in order if one fails
- **Error Handling**: Comprehensive error handling for PDF issues

## Data Flow

### Search Workflow

```
User Input (keywords)
    │
    ▼
LiteratureSearch.search()
    │
    ├─► ArxivSource.search()
    ├─► SemanticScholarSource.search()
    └─► Other sources...
    │
    ▼
Deduplication & Ranking
    │
    ▼
Add to Library (library.json, references.bib)
    │
    ▼
Download PDFs (if available)
    │
    ▼
Extract Text (optional)
    │
    ▼
Generate Summaries (optional, requires Ollama)
```

### Summarization Workflow

```
PDF File
    │
    ▼
PDF Text Extraction
    │
    ▼
Context Extraction (abstract, intro, conclusion)
    │
    ▼
Draft Generation (LLM)
    │
    ▼
Quality Validation
    │
    ├─► Pass → Save Summary
    └─► Fail → Refinement (LLM)
            │
            ▼
        Save Summary (with validation metadata)
```

## Module Dependencies

```
infrastructure/
├── core/          (no dependencies)
├── llm/
│   └── depends on: core/
├── literature/
│   ├── depends on: core/
│   ├── depends on: llm/ (for summarization)
│   └── depends on: validation/ (for PDF text extraction)
└── validation/    (no dependencies)
```

## Configuration Management

Configuration is managed through:
1. **Environment variables** (highest priority)
2. **Configuration files** (YAML)
3. **Default values** (in code)

All modules support environment-based configuration for easy deployment.

## Testing Architecture

### Test Organization

```
tests/
└── infrastructure/
    ├── core/          # Core utilities tests
    ├── llm/           # LLM integration tests
    └── literature/    # Literature search tests
```

### Test Philosophy

- **No mocks**: Real data and computations
- **Integration tests**: Marked with `@pytest.mark.requires_ollama`
- **Graceful skipping**: Tests skip when services unavailable

## Output Structure

```
data/
├── library.json              # Paper metadata index
├── references.bib            # BibTeX bibliography
├── summarization_progress.json  # Progress tracking
├── failed_downloads.json     # Failed download tracking
├── pdfs/                     # Downloaded PDFs
├── summaries/                # AI-generated summaries
├── extracted_text/           # Extracted PDF text
├── embeddings/               # Cached embedding files (JSON)
└── output/                   # Meta-analysis outputs

literature/                   # LLM operation outputs (created at repo root)
└── llm_outputs/              # Advanced LLM operation results
    ├── review_outputs/
    ├── communication_outputs/
    ├── compare_outputs/
    ├── gaps_outputs/
    └── network_outputs/
```

## Extension Points

The architecture supports extension at multiple levels:

1. **New Sources**: Add source adapters in `infrastructure/literature/sources/`
2. **New Analyzers**: Add analysis tools in `infrastructure/literature/analysis/`
3. **New Templates**: Add LLM templates in `infrastructure/llm/templates/`
4. **New HTML Parsers**: Add publisher-specific parsers in `infrastructure/literature/html_parsers/`
5. **New LLM Operations**: Add advanced LLM operations in `infrastructure/literature/llm/`
6. **New Workflows**: Add workflow orchestrators in `scripts/` (thin coordinators only)
7. **New Report Formats**: Add export formats in `infrastructure/literature/reporting/`

## See Also

- **[Getting Started](getting-started.md)** - Quick start guide
- **[Module Documentation](modules/)** - Detailed module documentation
- **[LLM Operations Guide](guides/llm-operations.md)** - Advanced multi-paper LLM operations
- **[API Reference](reference/api-reference.md)** - Complete API documentation
- **[Documentation AGENTS.md](AGENTS.md)** - Documentation structure and organization

