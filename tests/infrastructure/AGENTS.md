# Infrastructure Tests Module

## Purpose

The infrastructure tests module provides test coverage for all infrastructure layer modules, organized by module structure.

## Test Organization

```
tests/infrastructure/
├── core/          # Core utilities tests
├── llm/           # LLM integration tests
└── literature/   # Literature search tests
```

## Test Coverage

### Core Tests (`core/`)

15 test files covering:
- Configuration management
- Logging utilities
- Exception handling
- Progress tracking
- Checkpoint management
- Retry logic
- Performance monitoring
- Environment setup
- Script discovery
- File operations

### LLM Tests (`llm/`)

22 test files with 88%+ coverage:
- LLMClient core functionality
- Configuration and context management
- Template rendering
- Output validation
- Ollama utilities
- CLI interface
- Review generation
- Streaming support

**Test Documentation:**
- `tests/infrastructure/llm/AGENTS.md` - LLM test documentation
- `tests/infrastructure/llm/README.md` - Quick reference

### Literature Tests (`literature/`)

32 test files covering:
- Core search functionality
- Source adapters
- PDF handling
- Library management
- Summarization
- Meta-analysis
- Workflow orchestration
- CLI interface
- Integration tests

## Running Tests

```bash
# All infrastructure tests
pytest tests/infrastructure/

# Specific module
pytest tests/infrastructure/core/
pytest tests/infrastructure/llm/
pytest tests/infrastructure/literature/

# With coverage
pytest tests/infrastructure/ --cov=infrastructure --cov-report=html
```

## Test Philosophy

- **No mocks**: Real data and computations
- **Integration tests**: Marked with `@pytest.mark.requires_ollama`
- **Graceful skipping**: Tests skip when services unavailable

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Tests module overview
- [`core/`](core/) - Core tests
- [`llm/`](llm/) - LLM tests
- [`literature/`](literature/) - Literature tests



