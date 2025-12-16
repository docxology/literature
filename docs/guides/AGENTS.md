# Guides Directory - Documentation Guide

## Purpose

The `guides/` directory contains task-oriented how-to guides for common operations with the Literature Search and Management System. These guides focus on practical usage, step-by-step instructions, and examples.

## Guide Organization

### Available Guides

1. **search-papers.md** - Guide to searching for academic papers
   - Multi-source search configuration
   - Search options and limits
   - Processing results
   - Deduplication
   - PDF download integration

2. **summarize-papers.md** - Guide to generating LLM-powered summaries
   - Prerequisites and setup
   - Summarization process
   - Quality validation
   - Progress tracking
   - Troubleshooting

3. **llm-operations.md** - Multi-paper LLM operations
   - Available operations (6 types)
   - Paper selection
   - Operation workflow
   - Input quality considerations
   - Output locations

4. **meta-analysis.md** - Running meta-analysis on library
   - Standard vs. embedding analysis
   - Analysis types
   - Configuration
   - Output files
   - Best practices

5. **configuration.md** - System configuration guide
   - Environment variables
   - Configuration methods
   - Source-specific settings
   - LLM configuration
   - File path settings

6. **troubleshooting.md** - Troubleshooting guide
   - Ollama connection issues
   - PDF download failures
   - Library issues
   - Summarization problems
   - Configuration issues
   - Performance issues

## Guide Writing Standards

### Structure

Each guide should follow this structure:

1. **Title and Overview** - Clear purpose and scope
2. **Prerequisites** - Requirements and setup
3. **Quick Start** - Fast path to getting started
4. **Detailed Sections** - Coverage
5. **Examples** - Concrete, working examples
6. **Best Practices** - Recommendations
7. **Troubleshooting** - Common issues (or link to troubleshooting guide)
8. **See Also** - Related documentation

### Writing Principles

**"Show Not Tell" Principle:**
- Include concrete code examples, not just descriptions
- Show actual usage patterns and workflows
- Provide scenarios
- Demonstrate error handling

**Understated and Factual:**
- Avoid marketing language or hyperbole
- Focus on accurate, clear descriptions
- Let functionality speak for itself
- Use precise terminology

**Clear and Interpretable:**
- Use clear, concise language
- Structure information logically
- Make accessible to all skill levels
- Use consistent terminology

### Code Examples

All code examples must:
- Be tested and working
- Use current best practices
- Include error handling where appropriate
- Show both simple and complex usage
- Include necessary imports and setup

Example format:
```python
from infrastructure.literature import LiteratureSearch

searcher = LiteratureSearch()
papers = searcher.search("machine learning", limit=10)
```

## Cross-Referencing Patterns

### Links to Module Documentation

When referencing implementation details:
```markdown
See [Literature Module Documentation](../modules/literature.md) for details.
```

When referencing specific modules:
```markdown
See [Summarization Module Documentation](../../infrastructure/literature/summarization/AGENTS.md) for documentation.
```

### Links to Other Guides

When referencing related tasks:
```markdown
See [Configuration Guide](configuration.md) for configuration options.
```

When referencing troubleshooting:
```markdown
See [Troubleshooting Guide](troubleshooting.md) for common issues.
```

### Links to Reference Documentation

When referencing APIs:
```markdown
See [API Reference](../reference/api-reference.md) for API documentation.
```

When referencing CLI:
```markdown
See [CLI Reference](../reference/cli-reference.md) for command-line options.
```

### Links to Architecture

When explaining system design:
```markdown
See [Architecture Overview](../architecture.md) for system design.
```

## Guide Maintenance

### Keeping Guides Current

1. **Update When Features Change**: Modify guides when functionality changes
2. **Test All Examples**: Verify code examples work with current implementation
3. **Review Regularly**: Check for accuracy and completeness
4. **Update Cross-References**: Ensure links are valid
5. **Remove Obsolete Content**: Delete outdated information

### Review Checklist

- [ ] All code examples are tested and working
- [ ] All prerequisites are documented
- [ ] All steps are clear and accurate
- [ ] All cross-references are valid
- [ ] Troubleshooting covers common issues
- [ ] Best practices are current
- [ ] Configuration options are documented
- [ ] Output locations are accurate

### Adding New Guides

When adding a new guide:

1. **Choose Descriptive Name**: Use kebab-case (e.g., `new-feature.md`)
2. **Follow Structure**: Use existing guides as templates
3. **Add to README**: Update `guides/README.md` with link
4. **Add Cross-References**: Link to related guides and modules
5. **Test Examples**: Verify all code examples work
6. **Review for Consistency**: Ensure style matches existing guides

## Common Issues and Solutions

### Guide-Specific Issues

**Outdated Examples:**
- Regularly test all code examples
- Update when APIs change
- Remove examples that no longer work

**Missing Information:**
- Review guides for completeness
- Add missing prerequisites
- Document all configuration options
- Include troubleshooting for common issues

**Inconsistent Terminology:**
- Use consistent terminology across all guides
- Reference module documentation for technical terms
- Maintain glossary if needed

**Broken Cross-References:**
- Validate all links regularly
- Use relative paths consistently
- Update links when files move

## Integration with Module Documentation

Guides complement module documentation:

- **Guides**: Task-oriented, user-focused, practical examples
- **Module AGENTS.md**: Technical details, implementation, architecture

Guides should:
- Reference module documentation for technical details
- Focus on usage and workflows
- Provide practical examples
- Link to module docs for deep dives

## See Also

- [Guides README](README.md) - Guide index
- [Documentation AGENTS.md](../AGENTS.md) - Documentation structure
- [Module Documentation](../modules/) - Module documentation
- [Reference Documentation](../reference/) - API/CLI reference


