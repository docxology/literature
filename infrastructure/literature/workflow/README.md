# Workflow Module

Workflow orchestration, progress tracking, and search orchestration.

## Components

- **workflow.py**: `LiteratureWorkflow` for multi-paper operations
- **orchestrator.py**: Thin orchestrator (re-exports from operations/)
- **operations/**: Operation-specific modules (search, download, cleanup, meta-analysis, LLM operations)
- **progress.py**: Progress tracking for long-running tasks

## Quick Start

```python
from infrastructure.literature.workflow import (
    LiteratureWorkflow,
    ProgressTracker
)

workflow = LiteratureWorkflow()
result = workflow.search_and_add(keywords=["machine learning"], limit=10)
```

## Features

- Multi-paper operation orchestration
- Progress tracking with resumability
- Search workflow management
- Failed download tracking with automatic skip behavior

## See Also

- [`AGENTS.md`](AGENTS.md) - Documentation
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


