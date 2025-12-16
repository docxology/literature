# Infrastructure Layer Module

Documentation for the infrastructure layer.

## Overview

The infrastructure layer provides reusable, standalone modules for literature search, PDF management, reference tracking, and AI-powered summarization. It is designed to be completely independent and self-contained.

## Module Structure

```
infrastructure/
├── core/          # Foundation utilities
├── llm/           # Local LLM integration
├── literature/    # Literature search and management
└── validation/    # Validation utilities
```

## Core Module

Foundation utilities used across all infrastructure modules.

**Key Components:**
- Logging system
- Exception hierarchy
- Configuration management
- Progress tracking
- Checkpoint management
- Retry logic
- Performance monitoring

**See:** [Core Module Documentation](../infrastructure/core/AGENTS.md)

## LLM Module

Local LLM integration for research assistance.

**Key Components:**
- LLMClient for Ollama interaction
- Configuration management
- Context management
- Template system
- Output validation
- Review generation

**See:** [LLM Module Documentation](../infrastructure/llm/AGENTS.md)

## Literature Module

Literature search and management functionality.

**Key Components:**
- LiteratureSearch main interface
- Source adapters (arXiv, Semantic Scholar, etc.)
- PDF handling
- Library management
- Summarization system
- Meta-analysis tools

**See:** [Literature Module Documentation](../infrastructure/literature/AGENTS.md)

## Validation Module

PDF validation and text extraction utilities.

**Key Components:**
- PDF text extraction with multi-library support
- Automatic fallback between PDF parsing libraries
- Error handling for PDF issues

**See:** [Validation Module Documentation](../infrastructure/validation/AGENTS.md)

## Module Dependencies

The infrastructure layer follows a clear dependency hierarchy:

```
infrastructure/
├── core/          # No dependencies (foundation layer)
├── llm/           # Depends on: core/
├── literature/    # Depends on: core/, llm/ (for summarization)
└── validation/    # No dependencies (standalone utility)
```

**Dependency Rules:**
- **Core**: Foundation utilities with no external dependencies
- **LLM**: Depends only on core for logging, exceptions, configuration
- **Literature**: Depends on core and llm (for summarization features)
- **Validation**: Standalone utility with no infrastructure dependencies

## Usage

### Importing Modules

```python
# Core utilities
from infrastructure.core import get_logger, TemplateError

# LLM integration
from infrastructure.llm import LLMClient, LLMConfig

# Literature search
from infrastructure.literature import (
    LiteratureSearch,
    LiteratureConfig
)

# Validation
from infrastructure.validation import extract_text_from_pdf
```

## Module Boundaries

Each module is designed to be:
- **Self-contained**: All functionality within module boundaries
- **Well-defined APIs**: Clear public interfaces via `__init__.py` exports
- **Independent testing**: Each module has its own test suite
- **Documented**: Complete AGENTS.md and README.md for each module

## Configuration

All modules support environment variable configuration. See [Configuration Guide](../guides/configuration.md) for details.

## See Also

- **[Getting Started](../getting-started.md)** - Quick start guide
- **[Architecture Overview](../architecture.md)** - System architecture
- **[API Reference](../reference/api-reference.md)** - API documentation
- **[Core Module](core.md)** - Foundation utilities
- **[LLM Module](llm.md)** - Local LLM integration
- **[Literature Module](literature.md)** - Literature search and management
- **[Validation Module](validation.md)** - PDF validation and text extraction

