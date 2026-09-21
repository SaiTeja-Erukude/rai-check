from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from rai_audit.genai.agents.models import AgentTrace, TraceEvent


def adapt_langgraph_events(
    events: Iterable[Any],
    *,
    trace_id: str = "langgraph-trace",
    workflow_name: str = "LangGraph workflow",
) -> AgentTrace:
    """Normalize LangGraph stream events or callback events into an agent trace."""
    normalized = []
    for index, raw in enumerate(events):
        event = _mapping(raw)
        event_type = str(event.get("event", event.get("type", "")))
        data = _mapping(event.get("data", {}))
        if event_type in {"on_tool_start", "tool_start"}:
            normalized.append(
                _tool_event(
                    index,
                    tool_name=str(event.get("name", data.get("name", "unknown_tool"))),
                    content=_stringify(data.get("input", event.get("input"))),
                    source="agent",
                    attributes={"framework": "langgraph", "phase": "request"},
                )
            )
        elif event_type in {"on_tool_end", "tool_end"}:
            normalized.append(
                _tool_event(
                    index,
                    tool_name=str(event.get("name", data.get("name", "unknown_tool"))),
                    content=_stringify(data.get("output", event.get("output"))),
                    source="tool",
                    trusted=False,
                    attributes={"framework": "langgraph", "phase": "result"},
                )
            )
        elif event_type in {"on_retriever_end", "retrieval"}:
            normalized.append(
                TraceEvent(
                    id=f"langgraph-{index}",
                    operation="retrieval",
                    source=str(event.get("source", "retrieval")),
                    content=_stringify(data.get("output", event.get("content"))),
                    data_source_id=str(event.get("name", data.get("name", "retriever"))),
                    trusted=bool(event.get("trusted", False)),
                    attributes={"framework": "langgraph"},
                )
            )
        else:
            normalized.extend(_langgraph_messages(event, index))
    return AgentTrace(trace_id=trace_id, workflow_name=workflow_name, events=tuple(normalized))


def adapt_openai_agents_trace(trace: Any) -> AgentTrace:
    """Normalize an exported OpenAI Agents SDK trace and its span data."""
    root = _export(trace)
    trace_id = str(root.get("trace_id", "openai-agents-trace"))
    workflow_name = str(root.get("workflow_name", "OpenAI Agents workflow"))
    spans = root.get("spans", root.get("events", ()))
    normalized = []
    for index, raw in enumerate(spans):
        span = _export(raw)
        data = _export(span.get("span_data", span.get("data", {})))
        span_type = str(data.get("type", span.get("type", "")))
        event_id = str(span.get("span_id", span.get("id", f"openai-{index}")))
        parent_id = span.get("parent_id")
        related = (str(parent_id),) if parent_id else ()
        attributes = {"framework": "openai-agents-sdk", "sdk_span_type": span_type}
        if span_type == "agent":
            normalized.append(
                TraceEvent(
                    id=event_id,
                    operation="invoke_agent",
                    agent_name=_optional_string(data.get("name")),
                    attributes=attributes,
                    related_event_ids=related,
                    parent_event_id=str(parent_id) if parent_id else None,
                )
            )
        elif span_type == "function":
            normalized.append(
                _tool_event(
                    index,
                    event_id=event_id,
                    tool_name=str(data.get("name", "unknown_tool")),
                    content=_stringify(data.get("output", data.get("input"))),
                    source="tool",
                    trusted=False,
                    status="error" if span.get("error") else "ok",
                    attributes=attributes,
                    related_event_ids=related,
                    parent_event_id=str(parent_id) if parent_id else None,
                )
            )
        elif span_type in {"generation", "response"}:
            normalized.append(
                TraceEvent(
                    id=event_id,
                    operation="chat",
                    source="agent",
                    content=_stringify(data.get("output", data.get("response_id"))),
                    attributes=attributes,
                    related_event_ids=related,
                    parent_event_id=str(parent_id) if parent_id else None,
                )
            )
        elif span_type == "handoff":
            normalized.append(
                TraceEvent(
                    id=event_id,
                    operation="handoff",
                    agent_name=_optional_string(data.get("to_agent")),
                    attributes={
                        **attributes,
                        "from_agent": data.get("from_agent"),
                        "to_agent": data.get("to_agent"),
                    },
                    related_event_ids=related,
                    parent_event_id=str(parent_id) if parent_id else None,
                )
            )
    return AgentTrace(
        trace_id=trace_id,
        workflow_name=workflow_name,
        group_id=_optional_string(root.get("group_id")),
        metadata=_mapping(root.get("metadata", {})),
        events=tuple(normalized),
    )


def adapt_autogen_messages(
    messages: Iterable[Any],
    *,
    trace_id: str = "autogen-trace",
    workflow_name: str = "AutoGen workflow",
) -> AgentTrace:
    """Normalize AutoGen AgentChat messages and events into an agent trace."""
    normalized = []
    for index, raw in enumerate(messages):
        message = _export(raw)
        message_type = str(message.get("type", type(raw).__name__))
        content = message.get("content")
        if message_type == "ToolCallRequestEvent":
            for call_index, call in enumerate(_sequence(content)):
                item = _export(call)
                normalized.append(
                    _tool_event(
                        index,
                        event_id=f"autogen-{index}-{call_index}",
                        tool_name=str(item.get("name", "unknown_tool")),
                        content=_stringify(item.get("arguments")),
                        source="agent",
                        attributes={"framework": "autogen", "phase": "request"},
                    )
                )
        elif message_type == "ToolCallExecutionEvent":
            for result_index, result in enumerate(_sequence(content)):
                item = _export(result)
                normalized.append(
                    _tool_event(
                        index,
                        event_id=f"autogen-{index}-{result_index}",
                        tool_name=str(item.get("name", item.get("call_id", "unknown_tool"))),
                        content=_stringify(item.get("content")),
                        source="tool",
                        trusted=False,
                        status="error" if item.get("is_error") else "ok",
                        attributes={"framework": "autogen", "phase": "result"},
                    )
                )
        elif message_type == "MemoryQueryEvent":
            normalized.append(
                TraceEvent(
                    id=f"autogen-{index}",
                    operation="memory_read",
                    source="memory",
                    content=_stringify(content),
                    trusted=False,
                    attributes={"framework": "autogen"},
                )
            )
        else:
            normalized.append(
                TraceEvent(
                    id=f"autogen-{index}",
                    operation="message",
                    source=_message_source(message),
                    content=_stringify(content),
                    trusted=bool(message.get("trusted", True)),
                    agent_name=_optional_string(message.get("source")),
                    attributes={"framework": "autogen", "message_type": message_type},
                )
            )
    return AgentTrace(trace_id=trace_id, workflow_name=workflow_name, events=tuple(normalized))


def adapt_otel_spans(
    spans: Any,
    *,
    trace_id: str | None = None,
    workflow_name: str = "OpenTelemetry GenAI workflow",
) -> AgentTrace:
    """Normalize raw OTLP JSON spans while preserving parent-child relationships."""
    normalized = []
    flattened = list(_otel_spans(spans))
    for index, raw in enumerate(flattened):
        span = _export(raw)
        attributes = _otel_attributes(span.get("attributes", {}))
        event_id = str(span.get("spanId", span.get("span_id", span.get("id", f"otel-{index}"))))
        parent = span.get("parentSpanId", span.get("parent_span_id"))
        operation = str(
            attributes.get(
                "gen_ai.operation.name",
                _operation_from_span_name(str(span.get("name", ""))),
            )
        )
        normalized.append(
            TraceEvent(
                id=event_id,
                operation=operation,
                source=str(attributes.get("rai_audit.source", "agent")),
                content=_optional_string(
                    attributes.get(
                        "gen_ai.input.messages", attributes.get("gen_ai.output.messages")
                    )
                ),
                agent_name=_optional_string(attributes.get("gen_ai.agent.name")),
                tool_name=_optional_string(attributes.get("gen_ai.tool.name")),
                data_source_id=_optional_string(attributes.get("gen_ai.data_source.id")),
                trusted=bool(attributes.get("rai_audit.trusted", True)),
                status=_otel_status(span),
                related_event_ids=(str(parent),) if parent else (),
                parent_event_id=str(parent) if parent else None,
                attributes={
                    **attributes,
                    "telemetry.schema_url": span.get("schemaUrl", span.get("schema_url")),
                },
                timestamp=_optional_string(span.get("startTimeUnixNano", span.get("start_time"))),
            )
        )
    resolved_trace_id = trace_id or _otel_trace_id(flattened) or "otel-trace"
    return AgentTrace(
        trace_id=resolved_trace_id, workflow_name=workflow_name, events=tuple(normalized)
    )


def _langgraph_messages(event: Mapping[str, Any], index: int) -> list[TraceEvent]:
    messages = event.get("messages")
    if messages is None:
        for value in event.values():
            if isinstance(value, Mapping) and "messages" in value:
                messages = value["messages"]
                break
    normalized = []
    for message_index, raw in enumerate(_sequence(messages)):
        message = _export(raw)
        content = message.get("content")
        source = _message_source(message)
        normalized.append(
            TraceEvent(
                id=f"langgraph-{index}-{message_index}",
                operation="message",
                source=source,
                content=_stringify(content),
                trusted=bool(message.get("trusted", source not in {"tool", "retrieval"})),
                agent_name=_optional_string(message.get("name")),
                attributes={"framework": "langgraph"},
            )
        )
    return normalized


def _tool_event(
    index: int,
    *,
    tool_name: str,
    content: str | None,
    source: str,
    event_id: str | None = None,
    trusted: bool = True,
    status: str = "ok",
    attributes: Mapping[str, Any] | None = None,
    related_event_ids: tuple[str, ...] = (),
    parent_event_id: str | None = None,
) -> TraceEvent:
    return TraceEvent(
        id=event_id or f"tool-{index}",
        operation="execute_tool",
        source=source,
        content=content,
        tool_name=tool_name,
        trusted=trusted,
        status=status,
        attributes=attributes or {},
        related_event_ids=related_event_ids,
        parent_event_id=parent_event_id,
    )


def _message_source(message: Mapping[str, Any]) -> str:
    source = str(message.get("source", message.get("role", "agent"))).lower()
    if source in {"user", "tool", "system", "agent", "email", "webpage", "memory"}:
        return source
    return "agent"


def _export(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    export = getattr(value, "export", None)
    if callable(export):
        return _mapping(export())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _mapping(model_dump())
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _otel_spans(value: Any):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _otel_spans(item)
        return
    item = _export(value)
    if "resourceSpans" in item:
        yield from _otel_spans(item["resourceSpans"])
    elif "scopeSpans" in item:
        yield from _otel_spans(item["scopeSpans"])
    elif "instrumentationLibrarySpans" in item:
        yield from _otel_spans(item["instrumentationLibrarySpans"])
    elif "spans" in item:
        yield from _otel_spans(item["spans"])
    elif item:
        yield item


def _otel_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    result = {}
    for item in _sequence(value):
        attribute = _export(item)
        raw = attribute.get("value")
        if isinstance(raw, Mapping) and len(raw) == 1:
            raw = next(iter(raw.values()))
        result[str(attribute.get("key"))] = raw
    return result


def _operation_from_span_name(name: str) -> str:
    lowered = name.lower()
    if "tool" in lowered:
        return "execute_tool"
    if "retriev" in lowered:
        return "retrieval"
    if "workflow" in lowered:
        return "invoke_workflow"
    if "agent" in lowered:
        return "invoke_agent"
    return "chat"


def _otel_status(span: Mapping[str, Any]) -> str:
    status = _export(span.get("status", {}))
    code = str(status.get("code", status.get("status_code", "ok"))).lower()
    return "error" if code in {"error", "status_code_error", "2"} else "ok"


def _otel_trace_id(spans) -> str | None:
    for span in spans:
        value = span.get("traceId", span.get("trace_id"))
        if value:
            return str(value)
    return None
