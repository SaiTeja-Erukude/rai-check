# Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) in the repository root for the full contributing guide.

## Quick Reference

- All packages live under `packages/`
- Each package has its own `pyproject.toml`, `src/`, and `tests/`
- Run tests: `pytest packages/rai-check-core/tests`
- Add new checks in the appropriate module under `src/rai_audit/<module>/`
- Every new check must produce an `AuditFinding` with severity, evidence, recommendation, and `standards_refs`
