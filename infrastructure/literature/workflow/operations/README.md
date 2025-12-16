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
    run_enrich_dois,
)
from infrastructure.literature.workflow import LiteratureWorkflow

# Search only
run_search_only(keywords=["machine learning"], limit=10)

# Download PDFs (skips previously failed downloads by default)
run_download_only()

# Retry previously failed downloads
run_download_only(retry_failed=True)

# Cleanup library
run_cleanup()

# Meta-analysis
run_meta_analysis(keywords=["active inference"], limit=25)

# LLM operations
run_llm_operation(operation_type="review")

# DOI enrichment
workflow = LiteratureWorkflow()
run_enrich_dois(workflow)
```

## Modules

- **search.py** - Search operations
- **download.py** - PDF download operations
- **cleanup.py** - Library cleanup operations
- **meta_analysis.py** - Meta-analysis pipeline
- **llm_operations.py** - Advanced LLM operations
- **enrichment.py** - DOI enrichment operations
- **utils.py** - Common utilities

## See Also

- [`AGENTS.md`](AGENTS.md) - Complete documentation
- [`../AGENTS.md`](../AGENTS.md) - Workflow module documentation

