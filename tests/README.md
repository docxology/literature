# Tests Module

Comprehensive test suite for infrastructure modules.

## Quick Start

```bash
# Run all tests
pytest

# Run specific module
pytest tests/infrastructure/core/
```

## Test Organization

- **core/** - Core utilities tests
- **llm/** - LLM integration tests
- **literature/** - Literature search tests

## Test Philosophy

- No mocks policy
- Real data and computations
- Integration tests marked with `@pytest.mark.requires_ollama`

## Ollama Configuration

Test Ollama settings are configured in `test_ollama_config.yaml`. Edit this file to adjust:

- **Model**: Change `default_model` to use a different Ollama model
- **Temperature**: Adjust `temperature` (0.0-2.0) for deterministic vs creative responses
- **Context Window**: Set `context_window` based on model capabilities
- **Timeouts**: Adjust `test_timeouts` for different test types

Example: To use a faster model for tests:
```yaml
ollama:
  default_model: "llama3.2:3b"
  temperature: 0.0  # More deterministic
```

The configuration is automatically loaded by test fixtures.

## See Also

- [`AGENTS.md`](AGENTS.md) - Complete documentation

