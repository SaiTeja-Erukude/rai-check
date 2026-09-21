from pathlib import Path

import typer
from rai_audit.genai.agents import AgentAudit, AgentTrace, TraceEvent
from rai_audit.genai.agents.cli import register
from rai_audit.core.findings import RiskLevel
from typer.testing import CliRunner


def test_agent_audit_produces_report():
    trace = AgentTrace(
        trace_id="trace-1",
        workflow_name="support",
        events=(TraceEvent(id="tool-1", operation="execute_tool", tool_name="lookup_order"),),
    )

    report = AgentAudit(trace, allowed_tools=["lookup_order"], persist=False).run()

    assert report.audit_type == "agent_trace"
    assert report.metadata["trace_id"] == "trace-1"
    assert report.overall_risk_level == RiskLevel.LOW


def test_cli_writes_json_report(tmp_path: Path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        """
{
  "trace_id": "trace-1",
  "workflow_name": "support",
  "events": [
    {
      "id": "tool-1",
      "operation": "execute_tool",
      "tool_name": "lookup_order"
    }
  ]
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    app = typer.Typer()
    register(app)

    result = CliRunner().invoke(
        app,
        [
            "agents",
            "run",
            "--trace",
            str(trace),
            "--allowed-tools",
            "lookup_order",
            "--format",
            "json",
            "--out",
            str(output),
            "--no-persist",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.json").exists()
