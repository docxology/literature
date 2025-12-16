# Literature Tests Module

## Purpose

Test suite for literature search and management functionality, including integration tests with APIs.

## Test Files

### Core Tests
- `test_core.py` - LiteratureSearch core functionality
- `test_config.py` - Configuration tests
- `test_integration.py` - Integration tests
- `test_literature_integration.py` - API integration tests

### Source Tests
- `test_api.py` - API adapter tests
- `test_unpaywall.py` - Unpaywall integration tests
- `test_sources_comprehensive.py` - Source adapter tests (all sources)

### PDF Tests
- `test_pdf_handler_simple.py` - Basic PDF handler tests
- `test_pdf_handler_comprehensive.py` - PDF tests
- `test_pdf_handler_fallbacks.py` - PDF fallback strategy tests

### Library Tests
- `test_library_index.py` - Library indexing tests
- `test_clear_operations.py` - Library cleanup tests

### Summarization Tests
- `test_summarizer.py` - Summarization engine tests
- `test_summarization_streaming.py` - Streaming summarization tests
- `test_chunker.py` - Text chunking tests
- `test_prompt_builder.py` - Prompt builder tests

### Workflow Tests
- `test_workflow.py` - Workflow orchestration tests
- `test_workflow_skip_existing.py` - Skip existing functionality tests
- `test_progress.py` - Progress tracking tests

### Analysis Tests
- `test_analysis.py` - Paper analysis tests
- `test_meta_analysis.py` - Meta-analysis tests
- `test_embeddings.py` - Embedding analysis tests (pure logic, no mocks)
- `test_embeddings_real.py` - Embedding integration tests with Ollama API

### CLI Tests
- `test_literature_cli.py` - CLI interface tests
- `test_literature_cli_simple.py` - Simple CLI tests

### LLM Operations Tests
- `test_llm_operations.py` - LLM operations and integration tests
- `test_paper_selector.py` - Paper selection tests

### HTML Parser Tests
- `test_html_parsers.py` - HTML parser tests (ACM, Elsevier, IEEE, Springer, Wiley, Generic)

### Summarization Component Tests
- `test_summarization_components.py` - Individual summarization component tests (chunker, context_extractor, etc.)

### Other Tests
- `test_html_parsing.py` - HTML parser tests
- `test_reporting.py` - Reporting tests
- `test_logging.py` - Logging tests
- `test_structured_logging.py` - Structured logging tests
- `test_pure_logic.py` - Pure logic tests

## Running Tests

```bash
# All literature tests
pytest tests/infrastructure/literature/

# Specific category
pytest tests/infrastructure/literature/test_core.py
pytest tests/infrastructure/literature/test_summarizer.py

# Integration tests
pytest tests/infrastructure/literature/test_literature_integration.py
```

## Test Coverage

Coverage of:
- Core search functionality
- Source adapters (all sources tested)
- HTML parsers (all publisher-specific parsers)
- PDF handling and fallbacks
- Library management
- Summarization system (individual components)
- Workflow orchestration
- Meta-analysis (including embedding analysis)
- Embedding analysis (similarity, clustering, semantic search)
- CLI interface
- LLM operations

## Test Philosophy

### No Mocks Policy

All tests use implementations:
- **No mocks**: Tests use data, computations, and API calls
- **Embeddings**: Integration tests use Ollama embedding API
- **File operations**: All file I/O uses filesystem operations
- **Logging**: All tests include informative logging of operations
- **Veridical testing**: Tests verify behavior with systems

### Test Categories

1. **Pure Logic Tests** (`test_embeddings.py`): Test mathematical operations, data structures, and algorithms with data (no API calls)
2. **Integration Tests** (`test_embeddings_real.py`): Test with Ollama API, marked with `@pytest.mark.requires_ollama`

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../infrastructure/literature/AGENTS.md`](../../infrastructure/literature/AGENTS.md) - Literature module documentation

