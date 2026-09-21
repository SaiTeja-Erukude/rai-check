from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rai_audit.genai.agents.models import CONTENT_SOURCES, SUPPORTED_OPERATIONS, AgentTrace, TraceEvent
from rai_audit.core.schemas import SchemaDocumentError, prepare_document


class TraceValidationError(ValueError):
    """Raised when a captured agent trace does not match the canonical schema."""


def load_trace(path: str | Path) -> AgentTrace:
    """Load a canonical agent trace from JSON."""
    trace_path = Path(path)
    try:
        raw = json.loads(trace_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TraceValidationError(f"Could not read trace {trace_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TraceValidationError(f"Invalid JSON in {trace_path}: {exc}") from exc
    return trace_from_dict(raw)


def trace_from_dict(raw: Any) -> AgentTrace:
    try:
        root = prepare_document("trace", raw)
    except SchemaDocumentError as exc:
        raise TraceValidationError(f"Invalid trace schema: {exc}") from exc
    events_raw = root.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        raise TraceValidationError("trace.events must be a non-empty list")
    events = tuple(_event(item, index) for index, item in enumerate(events_raw))
    ids = [event.id for event in events]
    if len(ids) != len(set(ids)):
        raise TraceValidationError("trace.events must have unique ids")
    return AgentTrace(
        trace_id=_required_text(root, "trace_id", "trace"),
        workflow_name=_required_text(root, "workflow_name", "trace"),
        group_id=_optional_text(root.get("group_id"), "trace.group_id"),
        metadata=_mapping(root.get("metadata", {}), "trace.metadata"),
        events=events,
    )


def _event(raw: Any, index: int) -> TraceEvent:
    label = f"trace.events[{index}]"
    item = _mapping(raw, label)
    attributes = _mapping(item.get("attributes", {}), f"{label}.attributes")
    operation = _optional_text(
        item.get("operation", attributes.get("gen_ai.operation.name")),
        f"{label}.operation",
    )
    if operation is None:
        raise TraceValidationError(f"{label}.operation is required")
    if operation not in SUPPORTED_OPERATIONS:
        raise TraceValidationError(f"{label}.operation is unsupported: {operation}")
    source = _optional_text(item.get("source", "agent"), f"{label}.source") or "agent"
    if source not in CONTENT_SOURCES:
        raise TraceValidationError(f"{label}.source is unsupported: {source}")
    return TraceEvent(
        id=_required_text(item, "id", label),
        operation=operation,
        source=source,
        content=_optional_text(item.get("content"), f"{label}.content"),
        agent_name=_optional_text(
            item.get("agent_name", attributes.get("gen_ai.agent.name")),
            f"{label}.agent_name",
        ),
        tool_name=_optional_text(
            item.get("tool_name", attributes.get("gen_ai.tool.name")),
            f"{label}.tool_name",
        ),
        data_source_id=_optional_text(
            item.get("data_source_id", attributes.get("gen_ai.data_source.id")),
            f"{label}.data_source_id",
        ),
        trusted=_bool(item.get("trusted", True), f"{label}.trusted"),
        status=_optional_text(item.get("status", "ok"), f"{label}.status") or "ok",
        required_permissions=_text_tuple(
            item.get("required_permissions", ()),
            f"{label}.required_permissions",
        ),
        granted_permissions=_text_tuple(
            item.get("granted_permissions", ()),
            f"{label}.granted_permissions",
        ),
        approved=_optional_bool(item.get("approved"), f"{label}.approved"),
        related_event_ids=_text_tuple(
            item.get("related_event_ids", ()),
            f"{label}.related_event_ids",
        ),
        parent_event_id=_optional_text(item.get("parent_event_id"), f"{label}.parent_event_id"),
        attributes=attributes,
        timestamp=_optional_text(item.get("timestamp"), f"{label}.timestamp"),
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceValidationError(f"{label} must be a mapping")
    return value


def _required_text(value: Mapping[str, Any], key: str, label: str) -> str:
    text = _optional_text(value.get(key), f"{label}.{key}")
    if text is None:
        raise TraceValidationError(f"{label}.{key} is required")
    return text


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TraceValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _text_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TraceValidationError(f"{label} must be a list")
    return tuple(_required_list_text(item, label) for item in value)


def _required_list_text(value: Any, label: str) -> str:
    text = _optional_text(value, label)
    if text is None:
        raise TraceValidationError(f"{label} entries must be non-empty strings")
    return text


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TraceValidationError(f"{label} must be a boolean")
    return value


def _optional_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, label)
