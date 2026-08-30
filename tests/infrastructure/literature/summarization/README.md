# literature/tests/infrastructure/literature/summarization/

Tests for `infrastructure/literature/summarization` output validation: `test_output_validation.py` covers `MethodsToolsValidator` (methods/tools output validation, quote validation, summary-quality checks, output-size anomaly detection); `test_regressions.py` pins known past summarization failure cases so they cannot silently recur.

## Run

```bash
pytest tests/infrastructure/literature/summarization -v
```

Run from the `literature/` repo root (tests patch `sys.path` themselves, but the
repo root must be the working directory for imports to resolve).
See [`../../AGENTS.md`](../../AGENTS.md) for the suite's no-mocks policy and
[`../AGENTS.md`](../AGENTS.md) for the infrastructure-test organization.
