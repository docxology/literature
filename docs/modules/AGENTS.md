# Module Documentation Directory - Guide

## Purpose

The `modules/` directory contains user-facing documentation for all system modules. This documentation provides overviews, key components, usage examples, and integration patterns for each module in the infrastructure layer.

## Module Documentation Organization

### Available Module Documentation

1. **infrastructure.md** - Infrastructure layer overview
   - Module structure and organization
   - Module dependencies
   - Usage patterns
   - Integration guidelines

2. **core.md** - Foundation utilities module
   - Key components (logging, exceptions, configuration, etc.)
   - Usage examples
   - Integration patterns

3. **llm.md** - Local LLM integration module
   - LLMClient interface
   - Configuration management
   - Templates and validation
   - Usage examples

4. **literature.md** - Literature search and management module
   - Key components (core, sources, PDF, library, etc.)
   - Usage examples
   - Workflow operations
   - Meta-analysis tools

5. **validation.md** - PDF validation and text extraction module
   - PDF text extraction
   - Multi-library support
   - Error handling

## Module Documentation Standards

### Structure

Each module documentation file should follow this structure:

1. **Overview** - Module purpose and scope
2. **Key Components** - Main classes, functions, and features
3. **Usage Examples** - Concrete code examples
4. **Integration** - How to use with other modules
5. **See Also** - Links to related documentation

### Writing Principles

**User-Focused:**
- Focus on usage, not implementation details
- Provide practical examples
- Explain integration patterns
- Show common use cases

**Complementary to Module AGENTS.md:**
- Module docs (`docs/modules/`) = user-facing overview
- Module AGENTS.md (`infrastructure/*/AGENTS.md`) = technical details
- Both should be consistent but serve different purposes

**Clear Examples:**
- Include working code examples
- Show both simple and advanced usage
- Demonstrate integration patterns
- Include error handling

## Cross-Referencing Patterns

### Links to Module AGENTS.md Files

When referencing detailed technical information:
```markdown
See [Literature Module Documentation](../../infrastructure/literature/AGENTS.md) for complete documentation.
```

When referencing specific submodules:
```markdown
See [Summarization Module Documentation](../../infrastructure/literature/summarization/AGENTS.md) for details.
```

### Links to Guides

When referencing usage guides:
```markdown
See [Searching Papers Guide](../guides/search-papers.md) for usage examples.
```

When referencing configuration:
```markdown
See [Configuration Guide](../guides/configuration.md) for configuration options.
```

### Links to Reference Documentation

When referencing APIs:
```markdown
See [API Reference](../reference/api-reference.md) for complete API documentation.
```

### Links to Architecture

When explaining module architecture:
```markdown
See [Architecture Overview](../architecture.md) for system architecture.
```

## Integration with Infrastructure Module AGENTS.md Files

### Relationship

**Module Documentation (`docs/modules/`):**
- User-facing overview
- Usage-focused
- Practical examples
- Integration patterns
- Quick reference

**Module AGENTS.md (`infrastructure/*/AGENTS.md`):**
- Technical documentation
- Implementation details
- Internal architecture
- Developer-focused
- Comprehensive reference

### Integration Patterns

1. **Cross-Reference**: Module docs link to AGENTS.md for technical details
2. **Complementary Content**: Module docs focus on usage, AGENTS.md on implementation
3. **Consistent Terminology**: Both use same terminology
4. **Synchronized Updates**: Changes should update both levels

### Example Integration

**In `docs/modules/literature.md`:**
```markdown
## Key Components

### Core (`core/`)
Main interface for literature operations:
- LiteratureSearch class
- Configuration management
- CLI interface

See [Literature Module Documentation](../../infrastructure/literature/AGENTS.md) for complete technical documentation.
```

**In `infrastructure/literature/AGENTS.md`:**
```markdown
## Core Module

### LiteratureSearch Class

Complete implementation details, methods, error handling, etc.
```

## Module Documentation Maintenance

### Keeping Documentation Current

1. **Synchronize with Code**: Update when modules change
2. **Test Examples**: Verify all code examples work
3. **Update Cross-References**: Ensure links are valid
4. **Review Regularly**: Check for accuracy and completeness
5. **Coordinate with AGENTS.md**: Keep both levels synchronized

### Review Checklist

- [ ] All code examples are tested and working
- [ ] All components are documented
- [ ] All cross-references are valid
- [ ] Integration patterns are clear
- [ ] Terminology matches module AGENTS.md
- [ ] Usage examples are practical
- [ ] Links to guides and reference docs are current

### Adding New Module Documentation

When adding documentation for a new module:

1. **Create Module Doc**: Add `new-module.md` in `docs/modules/`
2. **Follow Structure**: Use existing module docs as templates
3. **Add to README**: Update `modules/README.md` with link
4. **Add Cross-References**: Link to module AGENTS.md, guides, reference
5. **Test Examples**: Verify all code examples work
6. **Review for Consistency**: Ensure style matches existing docs

## Module Documentation Standards

### Accuracy Requirements

- All information must match actual module implementation
- All code examples must be tested and working
- All component names must be correct
- All integration patterns must be accurate

### Completeness Requirements

- All modules must have documentation
- All key components must be covered
- All common use cases must have examples
- All integration patterns must be documented

### Consistency Requirements

- Terminology must match module AGENTS.md files
- Code style must be consistent
- Structure must be consistent
- Cross-reference patterns must be consistent

## See Also

- [Modules README](README.md) - Module documentation index
- [Documentation AGENTS.md](../AGENTS.md) - Documentation structure
- [Guides](../guides/) - How-to guides
- [Reference Documentation](../reference/) - API/CLI reference
- [Infrastructure Modules](../../infrastructure/) - Actual module implementations


