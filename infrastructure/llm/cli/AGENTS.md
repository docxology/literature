# LLM CLI Module

## Purpose

The LLM CLI module provides a command-line interface for interacting with local LLMs via Ollama. It wraps the infrastructure.llm module functionality to provide convenient terminal access.

## Commands

### Query Command

Send queries to the LLM with various response modes:

```bash
# Standard query
python3 -m infrastructure.llm.cli query "What is machine learning?"

# Short response (< 150 tokens)
python3 -m infrastructure.llm.cli query --short "Define AI"

# Long response (> 500 tokens)
python3 -m infrastructure.llm.cli query --long "Explain neural networks in detail"

# Streaming output
python3 -m infrastructure.llm.cli query --stream "Write a poem"

# With custom options
python3 -m infrastructure.llm.cli query \
    --temperature 0.0 \
    --seed 42 \
    --max-tokens 500 \
    "Test query"
```

### Check Command

Check Ollama connection status:

```bash
python3 -m infrastructure.llm.cli check
```

### Models Command

List available Ollama models:

```bash
python3 -m infrastructure.llm.cli models
```

### Template Command

Apply research templates:

```bash
# List available templates
python3 -m infrastructure.llm.cli template --list

# Apply template
python3 -m infrastructure.llm.cli template summarize_abstract \
    --input "Abstract text here..."
```

## Options

### Query Options

- `--short` - Request short response (< 150 tokens)
- `--long` - Request detailed response (> 500 tokens)
- `--stream` - Stream response in real-time
- `--model MODEL` - Override default model
- `--temperature FLOAT` - Sampling temperature (0.0 = deterministic)
- `--max-tokens INT` - Maximum tokens to generate
- `--seed INT` - Random seed for reproducibility

### Template Options

- `--list` - List available templates
- `--input TEXT` - Input text for template

## Implementation

The CLI is a thin orchestrator that:
1. Parses command-line arguments
2. Loads configuration from environment
3. Initializes LLMClient
4. Executes the requested operation
5. Formats and displays results

## Error Handling

The CLI provides user-friendly error messages:
- Connection errors with helpful suggestions
- Model availability checks
- Graceful handling of timeouts

## See Also

- [`README.md`](README.md) - Quick reference
- [`../../llm/AGENTS.md`](../../llm/AGENTS.md) - LLM module documentation
- [`../core/client.py`](../core/client.py) - LLMClient implementation






