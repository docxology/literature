# Reporting Module - Documentation

## Purpose

The reporting module provides reporting capabilities with multiple export formats.

## Components

### LiteratureReporter (reporter.py)

Reporter for literature operations.

**Key Methods:**
- `generate_workflow_report(workflow_result, library_entries=None, format="json")` - Generate workflow report
  - Supports formats: "json", "csv", "html", or "all" (generates all formats)
  - Returns Path to generated report file(s)

**Export Formats:**
- JSON - Machine-readable format
- CSV - Spreadsheet-compatible
- HTML - Human-readable with styling

## Usage Examples

### Generate Reports

```python
from infrastructure.literature.reporting import LiteratureReporter
from pathlib import Path

reporter = LiteratureReporter(Path("data/output/reports"))

# Workflow report (all formats)
report_path = reporter.generate_workflow_report(
    workflow_result,
    library_entries=entries,
    format="all"  # Generates JSON, CSV, and HTML
)

# Workflow report (single format)
json_path = reporter.generate_workflow_report(
    workflow_result,
    library_entries=entries,
    format="json"  # or "csv" or "html"
)
```

## See Also

- [`README.md`](README.md) - Quick reference
- [`../AGENTS.md`](../AGENTS.md) - Literature module overview


