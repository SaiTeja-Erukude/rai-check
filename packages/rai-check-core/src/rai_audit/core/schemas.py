from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "1.0"
SUPPORTED_DOCUMENT_TYPES = frozenset(
    {"config", "report", "trace", "suite", "evidence_manifest"}
)

_TEXT = {"type": "string", "minLength": 1}
_STRING_LIST = {"type": "array", "items": _TEXT}

SCHEMAS: dict[str, dict[str, Any]] = {
    "config": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://rai-check.dev/schemas/config-1.0.json",
        "type": "object",
        "required": ["schema_version", "audit"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "project": {"type": "object"},
            "audit": {"type": "object"},
            "checks": {"type": "object"},
            "gate": {"type": "object"},
        },
        "additionalProperties": True,
    },
    "report": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://rai-check.dev/schemas/report-1.0.json",
        "type": "object",
        "required": [
            "schema_version",
            "project_name",
            "audit_type",
            "findings",
            "risk_matrix",
            "metadata",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "project_name": _TEXT,
            "audit_type": _TEXT,
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["check_id", "title", "severity"],
                    "properties": {
                        "check_id": _TEXT,
                        "title": _TEXT,
                        "severity": {
                            "enum": ["critical", "high", "medium", "low", "info", "passed"]
                        },
                        "evidence": {"type": "object"},
                        "standards_refs": _STRING_LIST,
                    },
                    "additionalProperties": True,
                },
            },
            "risk_matrix": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["category", "risk_level", "finding_count", "passed_count"],
                    "properties": {
                        "category": _TEXT,
                        "risk_level": {"enum": ["critical", "high", "medium", "low"]},
                        "finding_count": {"type": "integer", "minimum": 0},
                        "passed_count": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": True,
                },
            },
            "metadata": {"type": "object"},
        },
        "additionalProperties": True,
    },
    "trace": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://rai-check.dev/schemas/trace-1.0.json",
        "type": "object",
        "required": ["schema_version", "trace_id", "workflow_name", "events"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "trace_id": _TEXT,
            "workflow_name": _TEXT,
            "metadata": {"type": "object"},
            "events": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": _TEXT,
                        "operation": _TEXT,
                        "attributes": {"type": "object"},
                    },
                    "anyOf": [
                        {"required": ["operation"]},
                        {
                            "required": ["attributes"],
                            "properties": {
                                "attributes": {
                                    "required": ["gen_ai.operation.name"],
                                }
                            },
                        },
                    ],
                    "additionalProperties": True,
                },
            },
        },
        "additionalProperties": True,
    },
    "suite": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://rai-check.dev/schemas/suite-1.0.json",
        "type": "object",
        "required": ["schema_version", "name", "cases"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "name": _TEXT,
            "defaults": {"type": "object"},
            "metadata": {"type": "object"},
            "cases": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "prompt"],
                    "properties": {
                        "id": _TEXT,
                        "prompt": _TEXT,
                        "type": _TEXT,
                        "checks": _STRING_LIST,
                    },
                    "anyOf": [{"required": ["type"]}, {"required": ["checks"]}],
                    "additionalProperties": True,
                },
            },
        },
        "additionalProperties": True,
    },
    "evidence_manifest": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://rai-check.dev/schemas/evidence-manifest-1.0.json",
        "type": "object",
        "required": [
            "schema_version",
            "audit_id",
            "generated_at",
            "project_name",
            "audit_type",
            "runtime",
            "library_versions",
            "inputs",
            "artifacts",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "audit_id": _TEXT,
            "generated_at": _TEXT,
            "project_name": _TEXT,
            "audit_type": _TEXT,
            "runtime": {"type": "object"},
            "library_versions": {"type": "object"},
            "inputs": {"type": "object"},
            "artifacts": {"type": "object"},
        },
        "additionalProperties": True,
    },
}


class SchemaDocumentError(ValueError):
    """Raised when a versioned document cannot be migrated or validated."""


def prepare_document(document_type: str, value: Any) -> dict[str, Any]:
    """Migrate a supported document to the current version, then validate it."""
    migrated = migrate_document(document_type, value)
    validate_document(document_type, migrated)
    return migrated


def migrate_document(document_type: str, value: Any) -> dict[str, Any]:
    """Return a migrated copy of a supported schema-bearing document."""
    _schema(document_type)
    if not isinstance(value, Mapping):
        raise SchemaDocumentError(f"{document_type} must be a mapping")
    document = deepcopy(dict(value))
    version = str(document.get("schema_version", "legacy"))
    while version != SCHEMA_VERSION:
        migration = _MIGRATIONS.get((document_type, version)) or _MIGRATIONS.get(("*", version))
        if migration is None:
            raise SchemaDocumentError(
                f"{document_type}.schema_version is unsupported: {version!r}"
            )
        document = migration(document)
        version = str(document.get("schema_version", "legacy"))
    return document


def validate_document(document_type: str, value: Any) -> None:
    """Validate a current-version document against its JSON Schema."""
    schema = _schema(document_type)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=_error_path)
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path)
    location = f"{document_type}.{path}" if path else document_type
    raise SchemaDocumentError(f"{location}: {error.message}")


def get_schema(document_type: str) -> dict[str, Any]:
    """Return a copy of a registered JSON Schema."""
    return deepcopy(_schema(document_type))


def _schema(document_type: str) -> dict[str, Any]:
    try:
        return SCHEMAS[document_type]
    except KeyError as exc:
        raise SchemaDocumentError(f"Unsupported document type: {document_type!r}") from exc


def _legacy_to_current(document: dict[str, Any]) -> dict[str, Any]:
    document["schema_version"] = SCHEMA_VERSION
    return document


def _legacy_suite_to_current(document: dict[str, Any]) -> dict[str, Any]:
    if "cases" not in document and "tests" in document:
        document["cases"] = document.pop("tests")
    return _legacy_to_current(document)


def _error_path(error: Any) -> tuple[str, ...]:
    return tuple(str(part) for part in error.absolute_path)


_MIGRATIONS: dict[tuple[str, str], Callable[[dict[str, Any]], dict[str, Any]]] = {
    ("*", "legacy"): _legacy_to_current,
    ("*", "0.1"): _legacy_to_current,
    ("suite", "legacy"): _legacy_suite_to_current,
    ("suite", "0.1"): _legacy_suite_to_current,
}
