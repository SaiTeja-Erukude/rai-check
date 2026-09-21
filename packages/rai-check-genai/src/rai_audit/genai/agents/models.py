from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from rai_audit.core.schemas import SCHEMA_VERSION

OTEL_OPERATIONS = frozenset(
    {
        "chat",
        "create_agent",
        "embeddings",
        "execute_tool",
        "generate_content",
        "invoke_agent",
        "invoke_workflow",
        "retrieval",
        "text_completion",
    }
)

AUDIT_OPERATIONS = frozenset({"memory_read", "memory_write", "message", "handoff"})
SUPPORTED_OPERATIONS = OTEL_OPERATIONS | AUDIT_OPERATIONS
CONTENT_SOURCES = frozenset(
    {"user", "tool", "retrieval", "email", "webpage", "memory", "system", "agent"}
)


@dataclass(frozen=True)
class TraceEvent:
    """Canonical agent event with OpenTelemetry GenAI-aligned attributes."""

    id: str
    operation: str
    source: str = "agent"
    content: str | None = None
    agent_name: str | None = None
    tool_name: str | None = None
    data_source_id: str | None = None
    trusted: bool = True
    status: str = "ok"
    required_permissions: tuple[str, ...] = ()
    granted_permissions: tuple[str, ...] = ()
    approved: bool | None = None
    related_event_ids: tuple[str, ...] = ()
    parent_event_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        attributes = dict(self.attributes)
        attributes.setdefault("gen_ai.operation.name", self.operation)
        if self.agent_name:
            attributes.setdefault("gen_ai.agent.name", self.agent_name)
        if self.tool_name:
            attributes.setdefault("gen_ai.tool.name", self.tool_name)
        if self.data_source_id:
            attributes.setdefault("gen_ai.data_source.id", self.data_source_id)
        return {
            "id": self.id,
            "operation": self.operation,
            "source": self.source,
            "content": self.content,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "data_source_id": self.data_source_id,
            "trusted": self.trusted,
            "status": self.status,
            "required_permissions": list(self.required_permissions),
            "granted_permissions": list(self.granted_permissions),
            "approved": self.approved,
            "related_event_ids": list(self.related_event_ids),
            "parent_event_id": self.parent_event_id,
            "attributes": attributes,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class AgentTrace:
    trace_id: str
    workflow_name: str
    events: tuple[TraceEvent, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    group_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "workflow_name": self.workflow_name,
            "group_id": self.group_id,
            "metadata": dict(self.metadata),
            "events": [event.to_dict() for event in self.events],
        }
