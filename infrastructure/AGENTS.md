# Infrastructure Layer

## Purpose

The infrastructure layer provides reusable, standalone modules for literature search, PDF management, reference tracking, and AI-powered summarization. This layer is designed to be completely independent and self-contained.

## Architecture

The infrastructure layer is organized into three main modules:

### Core Module (`core/`)

Foundation utilities used across all infrastructure modules:
- **Logging**: Unified Python logging with consistent formatting
- **Exceptions**: Comprehensive exception hierarchy with context preservation
- **Configuration**: YAML and environment variable configuration management
- **Progress**: Progress tracking and visual indicators
- **Checkpoint**: Pipeline checkpoint management
- **Retry**: Retry logic with exponential backoff
- **Performance**: Performance monitoring and resource tracking
- **Environment**: Environment setup and validation
- **Script Discovery**: Script discovery and execution utilities
- **File Operations**: File management utilities

### LLM Module (`llm/`)

Local LLM integration for research assistance:
- **Core Client**: LLMClient for interacting with Ollama
- **Configuration**: LLMConfig and GenerationOptions
- **Context Management**: ConversationContext for multi-turn conversations
- **Templates**: Research prompt templates
- **Validation**: Output validation and quality checks
- **Review System**: Manuscript review generation
- **CLI**: Command-line interface for LLM operations
- **Prompts**: Prompt fragment system for composable prompts

### Literature Module (`literature/`)

Literature search and management:
- **Core**: Main LiteratureSearch class and configuration
- **Sources**: API adapters for academic databases (arXiv, Semantic Scholar, etc.)
- **PDF**: PDF downloading, extraction, and fallback strategies
- **Library**: Library indexing, BibTeX generation, statistics
- **Summarization**: AI-powered paper summarization
- **Meta-Analysis**: Analysis tools and visualizations
- **Workflow**: Workflow orchestration and progress tracking
- **Analysis**: Paper analysis, domain detection, context building
- **HTML Parsers**: Publisher-specific PDF URL extraction
- **Reporting**: Comprehensive reporting with multiple export formats
- **LLM Operations**: Advanced LLM operations for multi-paper synthesis

## Module Organization

```
infrastructure/
├── core/              # Foundation utilities
├── llm/               # LLM integration
├── literature/        # Literature search and management
└── validation/       # Validation utilities
```

## Key Features

### Standalone Design

- **No external dependencies** on template or manuscript systems
- **Duplicated shared infrastructure** for independence
- **Complete test suite** for all functionality
- **Self-contained** - can be used independently

### Modular Architecture

- **Thin orchestrator pattern** - business logic in modules, orchestration in scripts
- **Clear separation of concerns** - each module has a specific purpose
- **Comprehensive APIs** - well-documented public interfaces
- **Extensible** - easy to add new sources, analyzers, etc.

## Usage

### Importing Infrastructure Modules

```python
# Core utilities
from infrastructure.core import get_logger, TemplateError, load_config

# LLM integration
from infrastructure.llm import LLMClient, LLMConfig

# Literature search
from infrastructure.literature import (
    LiteratureSearch,
    LiteratureConfig,
    LiteratureWorkflow
)
```

### Configuration

All modules support environment variable configuration:

```bash
# Core logging
export LOG_LEVEL=1  # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR

# LLM settings
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gemma3:4b
export LLM_TEMPERATURE=0.7

# Literature settings
export LITERATURE_DEFAULT_LIMIT=25
export LITERATURE_SOURCES=arxiv,semanticscholar
```

## Testing

Comprehensive test suite located in `tests/infrastructure/`:

```bash
# Run all infrastructure tests
pytest tests/infrastructure/

# Run specific module tests
pytest tests/infrastructure/core/
pytest tests/infrastructure/llm/
pytest tests/infrastructure/literature/
```

## See Also

- [`core/AGENTS.md`](core/AGENTS.md) - Core utilities documentation
- [`llm/AGENTS.md`](llm/AGENTS.md) - LLM integration documentation
- [`literature/AGENTS.md`](literature/AGENTS.md) - Literature search documentation
- [`../README.md`](../README.md) - Repository overview

