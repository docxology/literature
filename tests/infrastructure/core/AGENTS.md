# Core Tests Module

## Purpose

Test suite for infrastructure core utilities, covering all foundation modules.

## Test Files

### Configuration Tests
- `test_config_loader.py` - Configuration file loading
- `test_config_cli_coverage.py` - CLI configuration coverage
- `test_config_validator.py` - Configuration validation tests

### Logging Tests
- `test_logging_utils.py` - Logging utilities
- `test_logging_helpers.py` - Logging helper functions
- `test_logging_progress.py` - Progress logging
- `test_logging_formatters.py` - Logging formatters (JSONFormatter, TemplateFormatter)

### Exception Tests
- `test_exceptions.py` - Exception hierarchy

### Utility Tests
- `test_checkpoint.py` - Checkpoint management
- `test_progress.py` - Progress tracking
- `test_retry.py` - Retry logic
- `test_performance.py` - Performance monitoring
- `test_environment.py` - Environment setup
- `test_script_discovery.py` - Script discovery
- `test_file_operations.py` - File operations
- `test_credentials.py` - Credential management

## Running Tests

```bash
# All core tests
pytest tests/infrastructure/core/

# Specific test file
pytest tests/infrastructure/core/test_logging_utils.py

# With coverage
pytest tests/infrastructure/core/ --cov=infrastructure.core
```

## Test Coverage

Coverage of:
- Configuration management
- Logging system (including formatters)
- Exception handling
- Progress tracking
- Checkpoint management
- Retry logic
- Performance monitoring
- Environment validation
- Script discovery
- File operations

## Test Infrastructure

- `conftest.py` - Core module shared fixtures (config files, log files)

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../infrastructure/core/AGENTS.md`](../../infrastructure/core/AGENTS.md) - Core module documentation

