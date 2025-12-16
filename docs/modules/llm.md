# LLM Module

Local LLM integration for research assistance.

## Overview

The LLM module provides a unified interface for interacting with local large language models (via Ollama) to assist with research tasks. It offers flexible response modes, validation, and conversation context management.

## Key Components

### LLMClient (`core/client.py`)

Main interface for querying LLMs:
- Multiple response modes (short, long, structured, raw)
- Streaming support
- Context management
- Template support

### Configuration (`core/config.py`)

Configuration management:
- LLMConfig for global settings
- GenerationOptions for per-query control
- Environment variable support

### Templates (`templates/`)

Pre-built prompt templates:
- Research templates (summarize_abstract, literature_review, code_doc, data_interpret)
- Manuscript review templates (executive_summary, quality_review, methodology_review)
- Template registry with `get_template()`

### Validation (`validation/`)

Output validation:
- JSON validation with markdown handling
- Format compliance checking
- Repetition detection (sentence, paragraph, section level)
- Structure validation against schemas
- Citation extraction
- Off-topic content detection

### Prompts System (`prompts/`)

Composable prompt fragment system:
- Prompt fragments (JSON-based)
- Prompt composer for building complex prompts
- Template compositions
- System prompt management

### Review System (`infrastructure/llm/review/`)

Manuscript review generation:
- Executive summary generation
- Quality review generation
- Methodology review generation
- Improvement suggestions
- Translation support
- Review metrics and validation

### Utilities (`utils/`)

Ollama management utilities:
- Connection checking
- Model selection and preloading
- Server management
- Model information retrieval

## Usage Examples

### Basic Usage

```python
from infrastructure.llm import LLMClient

client = LLMClient()
response = client.query("What is machine learning?")
```

### Response Modes

```python
# Short response
answer = client.query_short("Define AI")

# Long response
explanation = client.query_long("Explain neural networks in detail")

# Structured response
result = client.query_structured(
    "Analyze...",
    schema={"type": "object", "properties": {...}}
)
```

### Templates

```python
from infrastructure.llm.templates import PaperSummarization, get_template

# Using template class
template = PaperSummarization()
prompt = template.render(text=paper_text)

# Using template registry
template = get_template("summarize_abstract")
prompt = template.render(text=abstract_text)
```

### Context Management

```python
from infrastructure.llm import LLMClient, ConversationContext

client = LLMClient()

# Multi-turn conversation
response1 = client.query("What is X?")
response2 = client.query("Can you elaborate?")  # Context maintained

# Reset context
client.reset()
```

### Validation

```python
from infrastructure.llm.validation import OutputValidator

# Validate JSON output
data = OutputValidator.validate_json(response)

# Check format compliance
is_compliant, issues = OutputValidator.check_format_compliance(response)

# Validate structure
OutputValidator.validate_structure(data, schema)
```

### Review Generation

```python
from infrastructure.llm.review import generate_executive_summary

# Generate executive summary
summary = generate_executive_summary(
    manuscript_text=text,
    output_dir=Path("output")
)
```

### Prompt System

```python
from infrastructure.llm.prompts import PromptFragmentLoader, PromptComposer

# Load fragments
loader = PromptFragmentLoader()
fragment = loader.load("system_prompts", "research_assistant")

# Compose prompts
composer = PromptComposer()
prompt = composer.compose(fragments=["intro", "task", "requirements"])
```

### Utilities

```python
from infrastructure.llm.utils import (
    is_ollama_running,
    select_best_model,
    preload_model
)

# Check Ollama
if is_ollama_running():
    print("Ollama is ready")

# Select model
model = select_best_model()

# Preload model
preload_model("llama3:latest")
```

## Configuration

See [Configuration Guide](../guides/configuration.md) for environment variables and settings.

## See Also

- **[LLM Module Documentation](../../infrastructure/llm/AGENTS.md)** - Technical documentation
- **[API Reference](../reference/api-reference.md)** - API documentation
- **[Modules AGENTS.md](AGENTS.md)** - Module documentation standards

