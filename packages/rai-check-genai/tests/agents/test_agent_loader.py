import json
from pathlib import Path

import pytest
from rai_audit.genai.agents.loader import TraceValidationError, load_trace, trace_from_dict
from rai_audit.genai.agents.models import TraceEvent


def test_load_trace_uses_otel_attributes(tmp_path: Path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "trace_id": "trace-1",
                "workflow_name": "support",
                "events": [
                    {
                        "id": "event-1",
                        "attributes": {
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": "lookup_order",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    trace = load_trace(path)

    assert trace.events[0].operation == "execute_tool"
    assert trace.events[0].tool_name == "lookup_order"


def test_trace_event_ids_must_be_unique():
    with pytest.raises(TraceValidationError, match="unique ids"):
        trace_from_dict(
            {
                "trace_id": "trace-1",
                "workflow_name": "support",
                "events": [
                    {"id": "duplicate", "operation": "message"},
                    {"id": "duplicate", "operation": "message"},
                ],
            }
        )


def test_trace_rejects_unknown_operation():
    with pytest.raises(TraceValidationError, match="unsupported"):
        trace_from_dict(
            {
                "trace_id": "trace-1",
                "workflow_name": "support",
                "events": [{"id": "event-1", "operation": "unknown"}],
            }
        )


def test_trace_event_emits_otel_genai_attributes():
    event = TraceEvent(
        id="tool-1",
        operation="execute_tool",
        tool_name="lookup_order",
        agent_name="Support Agent",
    )

    assert event.to_dict()["attributes"] == {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.agent.name": "Support Agent",
        "gen_ai.tool.name": "lookup_order",
    }


def test_trace_serialization_uses_current_schema_version():
    trace = trace_from_dict(
        {
            "trace_id": "trace-1",
            "workflow_name": "support",
            "events": [{"id": "event-1", "operation": "message"}],
        }
    )

    assert trace.to_dict()["schema_version"] == "1.0"


def test_trace_rejects_unknown_schema_version():
    with pytest.raises(TraceValidationError, match="unsupported"):
        trace_from_dict(
            {
                "schema_version": "99.0",
                "trace_id": "trace-1",
                "workflow_name": "support",
                "events": [{"id": "event-1", "operation": "message"}],
            }
        )
