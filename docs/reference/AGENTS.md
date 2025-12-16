# Reference Documentation Directory - Guide

## Purpose

The `reference/` directory contains reference documentation for APIs, CLI interfaces, and data formats. This documentation provides detailed information for developers and users.

## Reference Documentation Organization

### Available Reference Documentation

1. **api-reference.md** - Python API documentation
   - LiteratureSearch API
   - LLMClient API
   - SummarizationEngine API
   - All public methods and classes
   - Parameters, return values, exceptions

2. **cli-reference.md** - Command-line interface reference
   - Literature CLI commands
   - LLM CLI commands
   - Orchestrator script options
   - All flags and arguments
   - Usage examples

3. **data-formats.md** - Data structure and file format documentation
   - Library index format (library.json)
   - Bibliography format (references.bib)
   - Summary file format
   - Progress tracking format
   - Meta-analysis output formats

## Reference Documentation Standards

### Structure

Each reference document should follow this structure:

1. **Overview** - Purpose and scope
2. **API/CLI/Format Reference** - Documentation
3. **Examples** - Usage examples
4. **See Also** - Links to related documentation

### Writing Principles

**Thorough:**
- Document all public APIs, CLI commands, and data formats
- Include all parameters, return values, exceptions
- Provide information, not summaries
- Cover edge cases and error conditions

**Accurate:**
- All information must match actual implementation exactly
- All examples must be tested and working
- All parameter types must be correct
- All return values must be documented

**Clear:**
- Use clear, precise language
- Structure information logically
- Use consistent formatting
- Include examples for clarity

## API Reference Standards

### Documentation Requirements

For each API method/class:
- **Purpose**: What it does
- **Parameters**: All parameters with types and descriptions
- **Return Values**: Return type and description
- **Exceptions**: All possible exceptions
- **Examples**: Working code examples
- **Notes**: Important behaviors, side effects, limitations

### Example Format

```markdown
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
```

## CLI Reference Standards

### Documentation Requirements

For each CLI command:
- **Command**: Exact command syntax
- **Options**: All flags and arguments
- **Description**: What the command does
- **Examples**: Usage examples
- **Notes**: Important behaviors, requirements

### Example Format

```markdown
### Search Command

```bash
python3 -m infrastructure.literature.core.cli search "query" [options]
```

**Options:**
- `--limit N` - Limit results per source
- `--sources SOURCES` - Comma-separated source list
- `--download` - Download PDFs automatically

**Examples:**
```bash
python3 -m infrastructure.literature.core.cli search "machine learning"
```
```

## Data Format Reference Standards

### Documentation Requirements

For each data format:
- **Structure**: Structure with all fields
- **Field Descriptions**: All fields with types and descriptions
- **Examples**: Example data
- **Validation**: Format requirements and constraints
- **Versioning**: Format version information

### Example Format

```markdown
## Library Index (`library.json`)

JSON-based index of all papers in the library.

### Structure

```json
{
  "version": "1.0",
  "updated": "2025-12-02T04:42:16.615302",
  "count": 456,
  "entries": {
    "citation_key": {
      "citation_key": "smith2024machine",
      "title": "Paper Title",
      ...
    }
  }
}
```

### Fields

- `citation_key` (str): Unique identifier
- `title` (str): Paper title
- ...
```

## Cross-Referencing Patterns

### Links to Guides

When referencing usage:
```markdown
See [Searching Papers Guide](../guides/search-papers.md) for usage examples.
```

When referencing configuration:
```markdown
See [Configuration Guide](../guides/configuration.md) for configuration options.
```

### Links to Module Documentation

When referencing module details:
```markdown
See [Literature Module Documentation](../modules/literature.md) for module overview.
```

When referencing implementation:
```markdown
See [Literature Module Documentation](../../infrastructure/literature/AGENTS.md) for implementation details.
```

### Links to Architecture

When explaining design:
```markdown
See [Architecture Overview](../architecture.md) for system architecture.
```

## Reference Documentation Maintenance

### Keeping Documentation Current

1. **Synchronize with Code**: Update when APIs/CLI/data formats change
2. **Test All Examples**: Verify all code examples work
3. **Review Regularly**: Check for accuracy and completeness
4. **Update Cross-References**: Ensure links are valid
5. **Version Changes**: Document format version changes

### Review Checklist

- [ ] All APIs are documented
- [ ] All CLI commands are documented
- [ ] All data formats are documented
- [ ] All examples are tested and working
- [ ] All parameters are documented
- [ ] All return values are documented
- [ ] All exceptions are documented
- [ ] All cross-references are valid

### Adding New Reference Documentation

When adding new reference documentation:

1. **Choose Appropriate File**: Add to existing file or create new one
2. **Follow Structure**: Use existing reference docs as templates
3. **Add to README**: Update `reference/README.md` with link
4. **Add Cross-References**: Link to guides and modules
5. **Test Examples**: Verify all code examples work
6. **Review for Consistency**: Ensure style matches existing docs

## Accuracy and Completeness Requirements

### Accuracy Requirements

- All information must match actual implementation exactly
- All code examples must be tested and working
- All parameter types must be correct
- All return values must be documented accurately
- All exceptions must be documented

### Completeness Requirements

- All public APIs must be documented
- All CLI commands must be documented
- All data formats must be documented
- All parameters must be documented
- All return values must be documented
- All exceptions must be documented

### Consistency Requirements

- Terminology must be consistent
- Formatting must be consistent
- Structure must be consistent
- Cross-reference patterns must be consistent

## See Also

- [Reference README](README.md) - Reference documentation index
- [Documentation AGENTS.md](../AGENTS.md) - Documentation structure
- [Guides](../guides/) - How-to guides
- [Module Documentation](../modules/) - Module documentation


