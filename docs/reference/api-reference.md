# API Reference

Complete API documentation for the Literature Search and Management System.

## LiteratureSearch

Main interface for literature search operations.

### Methods

#### `search(query, limit=10, sources=None, return_stats=False)`

Search for papers across enabled sources.

**Parameters:**
- `query` (str): Search query string
- `limit` (int): Maximum results per source (default: 10)
- `sources` (List[str], optional): List of sources to use
- `return_stats` (bool): If True, return tuple of (results, statistics)

**Returns:**
- `List[SearchResult]` or `Tuple[List[SearchResult], SearchStatistics]`

**Example:**
```python
papers = searcher.search("machine learning", limit=10)
papers, stats = searcher.search("AI", return_stats=True)
```

#### `download_paper(result)`

Download PDF for a search result.

**Parameters:**
- `result` (SearchResult): Search result with pdf_url

**Returns:**
- `Optional[Path]`: Path to downloaded file, or None

#### `download_paper_with_result(result)`

Download PDF with detailed result tracking.

**Parameters:**
- `result` (SearchResult): Search result with pdf_url

**Returns:**
- `DownloadResult`: Detailed download result

#### `add_to_library(result)`

Add paper to library (BibTeX and JSON index).

**Parameters:**
- `result` (SearchResult): Search result to add

**Returns:**
- `str`: Citation key for the paper

#### `get_library_stats()`

Get library statistics.

**Returns:**
- `Dict[str, Any]`: Library statistics

#### `get_library_entries()`

Get all entries in the library.

**Returns:**
- `List[Dict[str, Any]]`: List of library entries

#### `get_source_health_status()`

Get health status for all sources.

**Returns:**
- `Dict[str, Any]`: Source health status

#### `check_all_sources_health()`

Check health of all sources.

**Returns:**
- `Dict[str, bool]`: Source health status (True/False)

#### `export_library(path=None, format="json")`

Export library to a file.

**Parameters:**
- `path` (Path, optional): Output path
- `format` (str): Export format ('json' supported)

**Returns:**
- `Path`: Path to exported file

#### `remove_paper(citation_key)`

Remove a paper from the library.

**Parameters:**
- `citation_key` (str): Citation key of paper to remove

**Returns:**
- `bool`: True if removed, False if not found

#### `cleanup_library(remove_missing_pdfs=True)`

Clean up library by removing entries without PDFs.

**Parameters:**
- `remove_missing_pdfs` (bool): Whether to remove entries without PDFs

**Returns:**
- `Dict[str, int]`: Cleanup statistics

## LLMClient

Main interface for LLM operations.

### Methods

#### `query(prompt, options=None, reset_context=False)`

Standard conversational query.

**Parameters:**
- `prompt` (str): Query prompt
- `options` (GenerationOptions, optional): Generation options
- `reset_context` (bool): Reset conversation context

**Returns:**
- `str`: LLM response

#### `query_short(prompt, options=None)`

Brief response (< 150 tokens).

**Parameters:**
- `prompt` (str): Query prompt
- `options` (GenerationOptions, optional): Generation options

**Returns:**
- `str`: Short response

#### `query_long(prompt, options=None)`

Comprehensive response (> 500 tokens).

**Parameters:**
- `prompt` (str): Query prompt
- `options` (GenerationOptions, optional): Generation options

**Returns:**
- `str`: Long response

#### `query_structured(prompt, schema, options=None, use_native_json=False)`

JSON-formatted structured response.

**Parameters:**
- `prompt` (str): Query prompt
- `schema` (Dict): JSON schema
- `options` (GenerationOptions, optional): Generation options
- `use_native_json` (bool): Use Ollama format="json"

**Returns:**
- `Dict[str, Any]`: Parsed JSON response

#### `check_connection()`

Check Ollama connection.

**Returns:**
- `bool`: True if connected

#### `check_connection_detailed()`

Check connection with detailed status.

**Returns:**
- `Tuple[bool, Optional[str]]`: (is_available, error_message)

## SummarizationEngine

Paper summarization engine.

### Methods

#### `summarize_paper(result, pdf_path, max_retries=2, progress_callback=None)`

Generate summary for a paper.

**Parameters:**
- `result` (SearchResult): Paper metadata
- `pdf_path` (Path): Path to PDF file
- `max_retries` (int): Maximum retry attempts
- `progress_callback` (Callable, optional): Progress callback

**Returns:**
- `SummarizationResult`: Summary result

#### `save_summary(result, summary_result, output_dir, pdf_path=None)`

Save summary to file.

**Parameters:**
- `result` (SearchResult): Paper metadata
- `summary_result` (SummarizationResult): Summary result
- `output_dir` (Path): Output directory
- `pdf_path` (Path, optional): PDF path for metadata

**Returns:**
- `Path`: Path to saved summary

## See Also

- **[Module Documentation](../modules/)** - Module-specific documentation
- **[CLI Reference](cli-reference.md)** - Command-line interface



