# Core Module

Foundation utilities for the infrastructure layer.

## Overview

The core module provides essential utilities used across all infrastructure modules, including logging, exception handling, configuration management, and progress tracking.

## Key Components

### Logging (`logging_utils.py`)

Unified Python logging system with consistent formatting:
- Environment-based configuration (LOG_LEVEL 0-3)
- Context managers for operation tracking
- Decorators for function call logging
- Integration with bash logging format

### Exceptions (`exceptions.py`)

Exception hierarchy:
- Base exception classes
- Module-specific exceptions
- Context preservation
- Exception chaining utilities

### Configuration (`config_loader.py`)

Configuration management:
- YAML configuration file loading
- Environment variable support
- Configuration file discovery
- Translation language configuration

### Progress (`progress.py`)

Progress tracking utilities:
- Visual progress indicators
- Sub-stage progress tracking
- Progress bar utilities

### Checkpoint (`checkpoint.py`)

Pipeline checkpoint management:
- Save/restore pipeline state
- Stage result tracking

### Retry (`retry.py`)

Retry logic with exponential backoff:
- Transient failure handling
- Retryable operation wrappers

## Usage Examples

### Logging

```python
from infrastructure.core import get_logger, log_operation

logger = get_logger(__name__)
logger.info("Starting process")

with log_operation("Data processing", logger):
    process_data()
```

### Exception Handling

```python
from infrastructure.core import TemplateError, chain_exceptions

try:
    risky_operation()
except ValueError as e:
    raise chain_exceptions(
        TemplateError("Operation failed"),
        e
    )
```

### Configuration

```python
from infrastructure.core import load_config, get_config_as_dict, find_config_file

# Load YAML config
config = load_config(Path("config.yaml"))

# Get config as dictionary
env_dict = get_config_as_dict(Path("."))

# Find config file
config_path = find_config_file(Path("."))
```

### Progress Tracking

```python
from infrastructure.core import ProgressBar, SubStageProgress

with ProgressBar(total=100, desc="Processing") as pbar:
    for i in range(100):
        pbar.update(1)
```

### Retry Logic

```python
from infrastructure.core import retry_with_backoff

@retry_with_backoff(max_attempts=3, base_delay=1.0)
def risky_operation():
    # Operation that may fail
    pass
```

### Performance Monitoring

```python
from infrastructure.core import PerformanceMonitor, get_system_resources

with PerformanceMonitor() as monitor:
    # Your code here
    pass

resources = get_system_resources()
print(f"CPU: {resources.cpu_percent}%, Memory: {resources.memory_percent}%")
```

### Checkpoint Management

```python
from infrastructure.core import CheckpointManager, StageResult

checkpoint = CheckpointManager()
if checkpoint.checkpoint_exists():
    state = checkpoint.load_checkpoint()
else:
    # Run pipeline stages
    checkpoint.save_checkpoint(stage_results)
```

### Environment Setup

```python
from infrastructure.core import (
    check_python_version,
    check_dependencies,
    setup_directories
)

check_python_version(min_version=(3, 10))
check_dependencies(["pandas", "numpy"])
setup_directories(["output", "output/figures"])
```

## Additional Components

### Credentials (`credentials.py`)
- Credential management from .env and YAML config files
- Environment variable loading
- Optional python-dotenv support

### File Operations (`file_operations.py`)
- Output directory cleanup
- Final deliverable copying

### Script Discovery (`script_discovery.py`)
- Script discovery and execution
- Analysis script finding
- Orchestrator script discovery

## See Also

- **[Core Module Documentation](../../infrastructure/core/AGENTS.md)** - Technical documentation
- **[API Reference](../reference/api-reference.md)** - API documentation
- **[Modules AGENTS.md](AGENTS.md)** - Module documentation standards

