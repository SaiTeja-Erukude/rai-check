# Configuration Schemas

RAI Check Kit uses versioned JSON Schemas for audit configs, reports, evidence
manifests, agent traces, and LLM suites. The current schema version is `1.0`.

## Versioning

New documents should include:

```yaml
schema_version: "1.0"
```

Existing unversioned documents remain supported. Loaders treat them as legacy
inputs, migrate a copy to the current schema, and validate the migrated document.
Unsupported future versions are rejected explicitly.

LLM suites using the legacy `tests` key are migrated to `cases`.

LLM suite cases accept either `prompt` or an ordered `messages` array. Conversation
messages have a `role` (`system`, `user`, or `assistant`) and `content`. An optional
system message precedes alternating user/assistant pairs. Assistant content may
be omitted or `null` for live Python responder calls. Captured replies use string
content. Case-level checks apply to each assistant reply; findings identify the
one-based message position as `evidence.turn_index`.

## Python API

```python
from rai_audit.core import get_schema, migrate_document, validate_document

schema = get_schema("config")
migrated = migrate_document("config", {"audit": {"type": "classification"}})
validate_document("config", migrated)
```

Supported document types:

- `config`
- `report`
- `trace`
- `suite`
- `evidence_manifest`

Package loaders apply schema validation before their domain-specific validation.
For example, agent trace validation still checks supported operations and unique
event IDs after validating the trace structure.
