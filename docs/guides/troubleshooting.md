# Troubleshooting Guide

Comprehensive troubleshooting guide for the Literature Search and Management System.

## Table of Contents

- [Ollama Connection Issues](#ollama-connection-issues)
- [PDF Download Failures](#pdf-download-failures)
- [Library Issues](#library-issues)
- [Summarization Problems](#summarization-problems)
- [Configuration Issues](#configuration-issues)
- [Search and API Issues](#search-and-api-issues)
- [Performance Issues](#performance-issues)

## Ollama Connection Issues

### Ollama Not Running

**Symptoms:**
- Connection errors when trying to generate summaries
- "Connection refused" errors
- Timeout errors

**Solutions:**

```bash
# Check if Ollama is running
ollama ps

# Start Ollama server
ollama serve

# In another terminal, verify it's working
ollama run gemma3:4b "Hello"
```

**Python Check:**
```python
from infrastructure.llm import LLMClient, is_ollama_running

# Simple check
if not is_ollama_running():
    print("Ollama not running - start with: ollama serve")

# Detailed diagnosis
client = LLMClient()
is_available, error = client.check_connection_detailed()
if not is_available:
    print(f"Ollama unavailable: {error}")
    if "timeout" in error.lower():
        print("  → Ollama may be slow to respond, try increasing LLM_TIMEOUT")
    elif "connection" in error.lower():
        print("  → Ollama server may not be running")
        print("  → Start with: ollama serve")
```

### Model Not Found

**Symptoms:**
- "Model not found" errors
- Empty responses

**Solutions:**

```bash
# List available models
ollama list

# Install required model
ollama pull gemma3:4b

# Verify model is installed
ollama show gemma3:4b
```

**Check Model in Python:**
```python
from infrastructure.llm.utils import get_available_models, check_model_loaded

models = get_available_models()
print(f"Available models: {models}")

# Check if specific model is loaded
is_loaded = check_model_loaded("gemma3:4b")
if not is_loaded:
    print("Model not loaded - may need to pull or start Ollama")
```

### Timeout Errors

**Symptoms:**
- Requests timing out
- Slow responses
- "Streaming timeout after X seconds" errors during summarization

**Solutions:**

```bash
# Increase timeout in environment
export LLM_TIMEOUT=120  # 2 minutes instead of default 60s for general LLM requests
export LLM_SUMMARIZATION_TIMEOUT=900  # 15 minutes for long papers (default: 600s)
```

**Adaptive Timeout:**
The summarization system uses adaptive timeouts based on prompt size:
- Base timeout: 300 seconds
- Adds 1 second per 1000 characters of prompt
- Maximum: 1200 seconds
- Final timeout = max(configured_timeout, adaptive_timeout)

Example: A 200K character prompt gets 300 + (200000/1000) = 500s minimum timeout.

**Partial Results:**
If a timeout occurs, partial results (>100 chars) are automatically saved to:
`data/summaries/_debug/{citation_key}_partial_{stage}_{timestamp}.md`

**Programmatic:**
```python
from infrastructure.llm import LLMConfig, LLMClient

# Increase timeout
config = LLMConfig(timeout=120.0)
client = LLMClient(config)
```

**Troubleshooting Timeout Issues:**
1. Check prompt size - very large prompts (>200K chars) may need two-stage mode
2. Enable two-stage mode: `LITERATURE_TWO_STAGE_ENABLED=true`
3. Increase timeout: `LLM_SUMMARIZATION_TIMEOUT=1200` (20 minutes)
4. Check generation rate - if <10 chars/s, model may be overloaded
5. Review partial results in `data/summaries/_debug/` to see progress before timeout

### Model Preloading

To reduce first-query latency, preload models:

```python
from infrastructure.llm.utils import preload_model

# Preload with error handling
success, error = preload_model("gemma3:4b", timeout=60.0)
if not success:
    print(f"Preload failed: {error}")
    print("Model will load on first query (may be slower)")
else:
    print("Model preloaded successfully")
```

## PDF Download Failures

### Understanding Failed Downloads

**Default Behavior:**
- Failed downloads are automatically tracked in `data/failed_downloads.json`
- **By default, previously failed downloads are skipped** to avoid wasting time
- Skip messages indicate: `⊘ Skipping {citation_key}: previously failed ({reason})`

**Common Failure Reasons:**
- `access_denied` - HTTP 403 Forbidden (paywall or geo-blocking)
- `not_found` - HTTP 404 Not Found (paper removed or URL changed)
- `rate_limited` - HTTP 429 Too Many Requests (API rate limit)
- `timeout` - Request timeout (slow network or server)
- `network_error` - Connection/Socket errors (DNS, SSL issues)
- `html_response` - HTML received instead of PDF (publisher landing page)
- `html_no_pdf_link` - HTML page with no PDF links

### Retrying Failed Downloads

**Retry Previously Failed Downloads:**
```bash
# Retry failed downloads when downloading
python3 scripts/07_literature_search.py --download-only --retry-failed

# Retry failed downloads in full search workflow
python3 scripts/07_literature_search.py --search --retry-failed --keywords "machine learning"
```

**Check Failed Downloads:**
```bash
# View failed downloads file
cat data/failed_downloads.json

# Or use the library CLI
python3 -m infrastructure.literature.core.cli library stats
```

### Improving Download Success Rate

**Use Browser User-Agent:**
```bash
export LITERATURE_USE_BROWSER_USER_AGENT=true
```

**Enable Unpaywall (Open Access Fallback):**
```bash
export LITERATURE_USE_UNPAYWALL=true
export UNPAYWALL_EMAIL=your@email.com
```

**Increase Timeout for Large Files:**
```bash
export LITERATURE_PDF_DOWNLOAD_TIMEOUT=120.0  # 2 minutes
```

**Adjust Retry Settings:**
```bash
export LITERATURE_DOWNLOAD_RETRY_ATTEMPTS=3
export LITERATURE_DOWNLOAD_RETRY_DELAY=3.0
```

### Download Summary

After downloads complete, a summary is displayed:
```
PDF DOWNLOAD SUMMARY
=====================================================================
  Total papers processed: 20
  ✓ Successfully downloaded: 7 (35.0%)
    • Already existed: 1
    • Newly downloaded: 6
  ✗ Failed downloads: 13
  ⏱️  Total time: 180.5s
  📊 Average time per paper: 9.0s
```

## Library Issues

### Library Validation

**Check Library Integrity:**
```bash
# Validate library structure
python3 -m infrastructure.literature.core.cli library validate

# Show library statistics
python3 -m infrastructure.literature.core.cli library stats
```

**Common Issues:**
- Corrupted JSON structure
- Missing required fields
- Invalid citation keys

### Library Cleanup

**Remove Papers Without PDFs:**
```bash
# Clean up library (removes entries without PDFs)
python3 scripts/07_literature_search.py --cleanup

# Or via CLI
python3 -m infrastructure.literature.core.cli library cleanup --no-pdf
```

**Backup Before Cleanup:**
```bash
# Complete backup
tar -czf literature_backup_$(date +%Y%m%d).tar.gz data/

# Metadata only
cp data/library.json data/references.bib backup/
```

### Library Corruption

**Symptoms:**
- JSON parsing errors
- Missing entries
- Invalid data structure

**Solutions:**

```bash
# Backup current library
cp data/library.json data/library.json.backup

# Validate library
python3 -m infrastructure.literature.core.cli library validate

# If corrupted, restore from backup
cp data/library.json.backup data/library.json
```

## Summarization Problems

### Low Quality Summaries

**Symptoms:**
- Summaries consistently score below 0.5
- Missing sections
- Repetitive content

**Solutions:**

**Check PDF Text Extraction:**
```python
from infrastructure.literature.pdf import PDFHandler

handler = PDFHandler()
text = handler.extract_text(Path("data/pdfs/paper.pdf"))
print(f"Extracted {len(text)} characters")
```

**Verify Context Extraction:**
- Check if key sections (abstract, intro, conclusion) are being found
- Review section detection patterns
- Increase `LLM_MAX_INPUT_LENGTH` if truncation is occurring

**Adjust Validation Thresholds:**
- Review validation errors in saved summary files
- Consider adjusting validation thresholds if needed
- Note: Summaries are always saved with validation metadata for review

### Slow Summarization

**Symptoms:**
- Summarization takes too long
- Timeout errors

**Solutions:**

**Reduce PDF Character Limit:**
```bash
export LLM_MAX_INPUT_LENGTH=100000  # 100K instead of 200K
```

**Use Parallel Processing:**
```bash
export MAX_PARALLEL_SUMMARIES=2  # Process 2 papers in parallel
```

**Check LLM Connection:**
- Verify Ollama is running and responsive
- Check model performance
- Review progress events to identify bottlenecks

### Missing Sections in Summaries

**Symptoms:**
- Important sections not included
- Incomplete summaries

**Solutions:**

**Check PDF Text Extraction:**
- Verify PDF text extraction is finding all sections
- Check for missing sections in extracted text

**Increase Character Limit:**
```bash
export LLM_MAX_INPUT_LENGTH=200000  # 200K characters
```

**Review Section Detection:**
- Check if context extraction is finding all sections
- Review section detection patterns
- Verify section identification is working correctly

### Validation Errors

**Symptoms:**
- Frequent validation errors
- Summaries rejected

**Solutions:**

**Review Validation Errors:**
- Check validation error messages in saved summary files
- Review validation metadata (status, score, errors)
- Summaries are always saved with validation metadata for review

**Note on Validation:**
- Term-based validation has been removed to reduce false positives
- Validation now focuses on structural quality (repetition, length, quotes)
- Consider adjusting validation thresholds if needed

## Configuration Issues

### Configuration Not Loading

**Symptoms:**
- Default values being used instead of environment variables
- Configuration not taking effect

**Solutions:**

**Check Environment Variables:**
```bash
# Verify environment variables are set
env | grep LITERATURE_
env | grep LLM_
env | grep OLLAMA_

# Load from .env file
source .env  # If using bash
```

**Verify Configuration Loading:**
```python
from infrastructure.literature import LiteratureConfig
from infrastructure.llm import LLMConfig

# Check loaded configuration
lit_config = LiteratureConfig.from_env()
print(f"Default limit: {lit_config.default_limit}")
print(f"Sources: {lit_config.sources}")

llm_config = LLMConfig.from_env()
print(f"Model: {llm_config.default_model}")
print(f"Timeout: {llm_config.timeout}")
```

**Configuration Precedence:**
1. Environment variables (highest priority)
2. Configuration files (YAML)
3. Default values (in code)

### Invalid Configuration Values

**Symptoms:**
- Errors when initializing components
- Unexpected behavior

**Solutions:**

**Validate Configuration:**
- Check environment variable types (integers, floats, booleans)
- Verify boolean values: `true`, `1`, `yes` (case-insensitive)
- Check numeric ranges (e.g., timeouts should be positive)

**Common Issues:**
- Boolean values: Use `true`/`false`, not `True`/`False`
- Numeric values: Ensure valid numbers (no strings)
- Paths: Use relative or absolute paths correctly

## Search and API Issues

### API Rate Limiting

**Symptoms:**
- HTTP 429 errors
- Slow searches
- Request failures

**Solutions:**

**Adjust Rate Limits:**
```bash
# Increase delays between requests
export LITERATURE_ARXIV_DELAY=5.0  # 5 seconds instead of 3
export LITERATURE_SEMANTICSCHOLAR_DELAY=2.0  # 2 seconds instead of 1.5
```

**Use API Keys:**
```bash
# Semantic Scholar API key (optional, for higher rate limits)
export SEMANTICSCHOLAR_API_KEY=your-api-key
```

**Check Source Health:**
```python
from infrastructure.literature import LiteratureSearch

searcher = LiteratureSearch()
health = searcher.get_source_health_status()
print(health)
```

### Source Connection Failures

**Symptoms:**
- Sources not responding
- Network errors
- Timeout errors

**Solutions:**

**Check Source Health:**
```python
from infrastructure.literature import LiteratureSearch

searcher = LiteratureSearch()
health = searcher.check_all_sources_health()
print(health)
```

**Increase Timeout:**
```bash
export LITERATURE_TIMEOUT=60.0  # 60 seconds instead of 30
```

**Retry Settings:**
```bash
export LITERATURE_RETRY_ATTEMPTS=5  # More retries
export LITERATURE_RETRY_DELAY=10.0  # Longer delay
```

### No Results Returned

**Symptoms:**
- Empty search results
- No papers found

**Solutions:**

**Check Query:**
- Verify search query is correct
- Try different keywords
- Check if sources are enabled

**Verify Sources:**
```bash
# Check enabled sources
export LITERATURE_SOURCES=arxiv,semanticscholar
```

**Check Source Availability:**
```python
from infrastructure.literature import LiteratureSearch

searcher = LiteratureSearch()
health = searcher.check_all_sources_health()
# Check which sources are healthy
```

## Performance Issues

### Slow Operations

**Symptoms:**
- Long wait times
- Timeout errors

**Solutions:**

**Parallel Processing:**
```bash
# Increase parallel downloads
export LITERATURE_MAX_PARALLEL_DOWNLOADS=8

# Increase parallel summaries
export MAX_PARALLEL_SUMMARIES=2
```

**Optimize Configuration:**
```bash
# Reduce limits if processing too many papers
export LITERATURE_DEFAULT_LIMIT=10

# Increase timeouts for large operations
export LLM_TIMEOUT=120
export LITERATURE_PDF_DOWNLOAD_TIMEOUT=120
```

**Check System Resources:**
- Monitor CPU and memory usage
- Check disk space
- Verify network connection speed

### Memory Issues

**Symptoms:**
- Out of memory errors
- System slowdown

**Solutions:**

**Reduce Parallel Operations:**
```bash
export LITERATURE_MAX_PARALLEL_DOWNLOADS=2
export MAX_PARALLEL_SUMMARIES=1
```

**Limit PDF Size:**
```bash
export LLM_MAX_INPUT_LENGTH=100000  # 100K instead of 200K
```

**Process in Batches:**
- Process papers in smaller batches
- Clear memory between batches

## Getting Help

### Debug Mode

Enable debug logging for detailed diagnostics:

```bash
export LOG_LEVEL=0  # DEBUG level
```

### Check Logs

Review log output for detailed error messages and diagnostics.

### Common Error Messages

**"Connection refused"**
- Ollama not running: `ollama serve`
- Wrong host/port: Check `OLLAMA_HOST`

**"Model not found"**
- Install model: `ollama pull gemma3:4b`
- Check model name: Verify `OLLAMA_MODEL`

**"Timeout"**
- Increase timeout: `LLM_TIMEOUT`, `LITERATURE_TIMEOUT`
- Check network connection
- Verify server is responsive

**"Access denied" (403)**
- Enable browser User-Agent: `LITERATURE_USE_BROWSER_USER_AGENT=true`
- Try Unpaywall: `LITERATURE_USE_UNPAYWALL=true`
- Paper may be behind paywall

## See Also

- **[Configuration Guide](configuration.md)** - Detailed configuration options
- **[Getting Started](../getting-started.md)** - Quick start guide
- **[API Reference](../reference/api-reference.md)** - Complete API documentation


