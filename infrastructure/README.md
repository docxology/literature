# Infrastructure Layer

Reusable modules for literature search, PDF management, and AI-powered summarization.

## Quick Start

```python
from infrastructure.literature import LiteratureSearch
from infrastructure.llm import LLMClient

# Search for papers
searcher = LiteratureSearch()
papers = searcher.search("machine learning", limit=10)

# Use LLM for analysis
client = LLMClient()
summary = client.query("Summarize this paper...")
```

## Modules

- **core/** - Foundation utilities (logging, config, exceptions)
- **llm/** - Local LLM integration (Ollama)
- **literature/** - Literature search and management

## Features

- Standalone and self-contained
- Comprehensive test coverage
- Environment-based configuration
- Modular architecture

## See Also

- [`AGENTS.md`](AGENTS.md) - Complete documentation
- [`../docs/`](../docs/) - Comprehensive documentation




