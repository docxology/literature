# Architecture and Design Principles

## Modular Architecture

### Modular Design
- Prefer modular, well-documented, clearly reasoned architecture
- Organize code into logical, reusable modules
- Maintain clear separation of concerns
- Design modules to be independent and testable

### Module Organization
- Structure modules to reflect system architecture
- Maintain consistent module organization patterns
- Use clear module boundaries and interfaces
- Ensure modules have well-defined responsibilities

### Module Dependencies
- Minimize dependencies between modules
- Avoid circular dependencies
- Use dependency injection where appropriate
- Design for loose coupling and high cohesion

## Design Patterns

### Appropriate Pattern Selection
- Choose design patterns that fit the problem
- Avoid over-engineering with unnecessary patterns
- Use patterns to improve maintainability and clarity
- Document pattern usage and rationale

### Common Patterns
- Use appropriate patterns for common problems
- Apply patterns consistently across the codebase
- Document pattern implementations
- Ensure patterns enhance rather than complicate code

## System Organization

### Directory Structure
- Organize code to reflect system architecture
- Maintain consistent directory structure
- Use clear naming conventions
- Group related functionality together

### Code Organization
- Organize code logically within modules
- Maintain clear file and class organization
- Use appropriate abstraction levels
- Keep related code together

### Interface Design
- Design clear, well-defined interfaces
- Keep interfaces focused and minimal
- Document interface contracts clearly
- Ensure interfaces are stable and maintainable

## Integration Considerations

### Module Integration
- Design modules for easy integration
- Use clear integration patterns
- Document integration requirements
- Ensure integration points are well-defined

### System Integration
- Design for system-wide integration
- Consider integration with external systems
- Plan for scalability and extensibility
- Document integration patterns and requirements

### Dependency Management
- Manage dependencies carefully
- Minimize external dependencies
- Document dependency requirements
- Ensure dependencies are well-maintained

## Architecture Principles

### Thin Orchestrator Pattern
- Business logic lives in infrastructure modules, not in orchestrator scripts
- Orchestrator scripts (`scripts/`) are thin coordinators that delegate to infrastructure
- Scripts handle user interaction, argument parsing, and workflow coordination
- All business logic must be in `infrastructure/` modules for testability and reusability
- Clear separation: Logic (infrastructure) vs. Orchestration (scripts)

### Separation of Concerns
- Separate concerns clearly and consistently
- Avoid mixing responsibilities
- Design focused, single-purpose components
- Maintain clear boundaries between concerns

### Abstraction and Encapsulation
- Use appropriate levels of abstraction
- Encapsulate implementation details
- Expose only necessary interfaces
- Hide complexity behind clear abstractions

### Scalability and Extensibility
- Design for future growth and changes
- Plan for extensibility
- Consider scalability requirements
- Design flexible, adaptable architectures

## Design Quality

### Clear Reasoning
- Document architectural decisions clearly
- Explain design choices and trade-offs
- Provide rationale for architectural patterns
- Make reasoning accessible to others

### Maintainability
- Design for long-term maintainability
- Keep architecture understandable
- Plan for future modifications
- Ensure architecture supports maintenance

### Testability
- Design architecture to support testing
- Ensure components are testable in isolation
- Plan for integration testing
- Make testing straightforward and reliable

## System Structure

### Module Organization
The system follows a modular architecture with clear module boundaries:

**Infrastructure Core (`infrastructure/core/`):**
- Foundation utilities (logging, exceptions, configuration, progress, checkpoint, retry, performance)
- No dependencies on other infrastructure modules

**LLM Module (`infrastructure/llm/`):**
- Local LLM integration (Ollama client, templates, validation, review system)
- Depends on: `infrastructure/core/`

**Literature Module (`infrastructure/literature/`):**
- Complete literature search and management functionality
- Submodules:
  - `core/` - Main search interface and configuration
  - `sources/` - API adapters (arXiv, Semantic Scholar, PubMed, CrossRef, OpenAlex, DBLP, bioRxiv, Europe PMC, Unpaywall)
  - `pdf/` - PDF downloading, extraction, and failed download tracking
  - `library/` - Library indexing and BibTeX generation
  - `workflow/` - Workflow orchestration with progress tracking
  - `summarization/` - AI-powered paper summarization
  - `meta_analysis/` - Bibliographic and statistical analysis tools
  - `analysis/` - Paper analysis, domain detection, context building
  - `html_parsers/` - Publisher-specific PDF URL extraction (Elsevier, Springer, IEEE, ACM, Wiley, generic)
  - `reporting/` - Multi-format export (JSON, CSV, HTML)
  - `llm/` - Advanced LLM operations (literature review, science communication, comparative analysis, research gaps, citation network analysis)
- Depends on: `infrastructure/core/`, `infrastructure/llm/` (for summarization and LLM operations), `infrastructure/validation/` (for PDF text extraction)

**Validation Module (`infrastructure/validation/`):**
- PDF validation and text extraction with multi-library support
- No dependencies on other infrastructure modules

### Output Structure
System outputs are organized as follows:

**Data Directory (`data/`):**
- `library.json` - Paper metadata index
- `references.bib` - BibTeX bibliography
- `summarization_progress.json` - Summarization progress tracking
- `failed_downloads.json` - Failed download tracking
- `pdfs/` - Downloaded PDFs (named by citation key)
- `summaries/` - AI-generated summaries
- `extracted_text/` - Extracted PDF text
- `embeddings/` - Cached embedding files (JSON) for semantic analysis
- `output/` - Meta-analysis outputs and visualizations

**LLM Outputs (`literature/llm_outputs/` at repo root):**
- Advanced LLM operation results (created at repo root, not in data/)
- `review_outputs/` - Literature review synthesis
- `communication_outputs/` - Science communication narratives
- `compare_outputs/` - Comparative analysis
- `gaps_outputs/` - Research gap identification
- `network_outputs/` - Citation network analysis (text-based)

### Layered Architecture
- Use appropriate layering when beneficial
- Maintain clear layer boundaries
- Define layer responsibilities clearly
- Ensure layers communicate through well-defined interfaces

### Component Design
- Design components with clear responsibilities
- Ensure components are reusable
- Make components independently testable
- Document component interfaces and behavior

### Data Flow
- Design clear data flow patterns
- Document data transformations
- Ensure data flow is traceable
- Plan for error handling in data flow

## Extension Points

The architecture supports extension at multiple levels:

1. **New Sources**: Add source adapters in `infrastructure/literature/sources/`
2. **New Analyzers**: Add analysis tools in `infrastructure/literature/analysis/`
3. **New Templates**: Add LLM templates in `infrastructure/llm/templates/`
4. **New HTML Parsers**: Add publisher-specific parsers in `infrastructure/literature/html_parsers/`
5. **New LLM Operations**: Add advanced LLM operations in `infrastructure/literature/llm/`
6. **New Workflows**: Add workflow orchestrators in `scripts/` (thin coordinators only)
7. **New Report Formats**: Add export formats in `infrastructure/literature/reporting/`

## Architecture Documentation

### Architecture Documentation
- Document system architecture comprehensively
- Explain architectural decisions and rationale
- Provide architecture diagrams when helpful
- Keep architecture documentation current

### Design Documentation
- Document design patterns and their usage
- Explain module organization and structure
- Provide integration guidelines
- Document system-wide conventions

## Evolution and Refactoring

### Architecture Evolution
- Plan for architecture evolution
- Design for incremental improvements
- Maintain backward compatibility when possible
- Document architectural changes

### Refactoring Guidelines
- Refactor to improve architecture
- Maintain architectural consistency
- Document refactoring decisions
- Ensure refactoring improves maintainability





