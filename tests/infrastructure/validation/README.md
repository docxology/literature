# literature/tests/infrastructure/validation/

Tests for `infrastructure/validation/pdf_validator.py` — PDF validation and text extraction (`extract_text_from_pdf`, `PDFValidationError`). Built with real generated PDFs (reportlab-style fixture PDFs), no mocks, per the suite policy.

## Run

```bash
pytest tests/infrastructure/validation -v
```

Run from the `literature/` repo root (tests patch `sys.path` themselves, but the
repo root must be the working directory for imports to resolve).
See [`../../AGENTS.md`](../../AGENTS.md) for the suite's no-mocks policy and
[`../AGENTS.md`](../AGENTS.md) for the infrastructure-test organization.
