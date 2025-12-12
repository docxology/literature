# Workflow Operations Module

Operation-specific implementations for literature workflow tasks.

## Quick Start

```python
from infrastructure.literature.workflow.operations import (
    run_search_only,
    run_download_only,
    run_cleanup,
    run_meta_analysis,
    run_llm_operation,
)

# Search only
run_search_only(keywords=["machine learning"], limit=10)

# Download PDFs
run_download_only()

# Cleanup library
run_cleanup()

# Meta-analysis
run_meta_analysis(keywords=["active inference"], limit=25)

# LLM operations
run_llm_operation(operation_type="review")
```

## Modules

- **search.py** - Search operations
- **download.py** - PDF download operations
- **cleanup.py** - Library cleanup operations
- **meta_analysis.py** - Meta-analysis pipeline
- **llm_operations.py** - Advanced LLM operations
- **utils.py** - Common utilities

## See Also

- [`AGENTS.md`](AGENTS.md) - Complete documentation
- [`../AGENTS.md`](../AGENTS.md) - Workflow module documentation

