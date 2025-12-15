# Tests Module

## Purpose

The tests module provides test coverage for all infrastructure modules. Tests follow a no-mocks policy, using data and computations.

## Test Organization

```
tests/
└── infrastructure/
    ├── core/          # Core utilities tests
    ├── llm/           # LLM integration tests
    └── literature/    # Literature search tests
```

## Test Philosophy

### No Mocks Policy

**All tests use implementations - no mocks are ever used.**

This policy ensures:
- **Veridical testing**: Tests verify behavior with systems
- **Data**: All tests use data structures and computations
- **API calls**: Integration tests use Ollama API and file operations
- **Comprehensive logging**: All tests include informative logging of operations
- **Factual verification**: Tests verify system behavior, not mocked responses

**Prohibited:**
- No `MagicMock`, `mocker.patch`, or `unittest.mock`
- No fake data or simulated responses
- No mocked API calls or file operations

**Required:**
- Implementations with data
- API calls (with graceful skipping if service unavailable)
- File operations with temporary directories
- Informative logging of all test operations

### Test Categories

1. **Pure Logic Tests** - Test business logic without network access
2. **Integration Tests** - Test with real services (marked with `@pytest.mark.requires_ollama`)

## Running Tests

### All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=infrastructure --cov-report=html
```

### Module-Specific Tests

```bash
# Core tests
pytest tests/infrastructure/core/

# LLM tests
pytest tests/infrastructure/llm/

# Literature tests
pytest tests/infrastructure/literature/
```

### Integration Tests

```bash
# Skip integration tests (fast)
pytest -m "not requires_ollama"

# Only integration tests (requires Ollama)
pytest -m requires_ollama
```

## Test Coverage

### Core Module
- 17 test files (including test_logging_formatters.py, test_config_validator.py)
- Comprehensive coverage of all utilities
- Exception handling tests
- Configuration tests (loader, CLI coverage, validator)
- Logging formatter tests

### LLM Module
- 24 test files (added test_review_components.py, test_templates_comprehensive.py)
- 88%+ coverage
- Pure logic and integration tests
- Template and validation tests
- Review component tests
- Template module comprehensive tests

### Literature Module
- 35 test files (added test_html_parsers.py, test_sources_comprehensive.py, test_summarization_components.py)
- Comprehensive coverage
- Integration tests with real APIs
- Workflow and orchestration tests
- HTML parser tests
- Source adapter comprehensive tests
- Summarization component tests

### Validation Module
- 1 test file (test_pdf_validator.py)
- PDF validation and text extraction tests

## Test Infrastructure

### Shared Fixtures
- `tests/conftest.py` - Root-level shared fixtures
- `tests/infrastructure/core/conftest.py` - Core module fixtures
- `tests/infrastructure/literature/conftest.py` - Literature module fixtures
- `tests/infrastructure/llm/conftest.py` - LLM module fixtures

### Test Utilities
- `tests/utils.py` - Shared test utilities (PDF creation, data generators, assertion helpers)

## Test Documentation

- `tests/infrastructure/llm/AGENTS.md` - LLM test documentation
- `tests/infrastructure/llm/README.md` - LLM test quick reference
- `tests/infrastructure/core/AGENTS.md` - Core test documentation
- `tests/infrastructure/literature/AGENTS.md` - Literature test documentation

## See Also

- [`README.md`](README.md) - Quick reference
- [`infrastructure/core/AGENTS.md`](../infrastructure/core/AGENTS.md) - Core module documentation
- [`infrastructure/llm/AGENTS.md`](../infrastructure/llm/AGENTS.md) - LLM module documentation
- [`infrastructure/literature/AGENTS.md`](../infrastructure/literature/AGENTS.md) - Literature module documentation

