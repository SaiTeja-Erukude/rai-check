from __future__ import annotations

from pathlib import Path

import typer
from rai_audit.genai.agents.audit import AgentAudit
from rai_audit.genai.agents.loader import TraceValidationError, load_trace
from rich.console import Console

console = Console()
agents_app = typer.Typer(name="agents", help="Audit agentic AI execution traces.")


def register(parent: typer.Typer) -> None:
    parent.add_typer(agents_app)


@agents_app.command("run")
def agents_run(
    trace: Path = typer.Option(..., help="Canonical agent trace JSON file"),
    allowed_tools: str | None = typer.Option(None, help="Comma-separated tool allowlist"),
    project: str | None = typer.Option(None, help="Project name override"),
    out: Path = typer.Option(Path("agent_audit_report.html"), help="Output report path"),
    format: str = typer.Option("html", help="Report format: html | markdown | json"),
    persist: bool = typer.Option(True, help="Save run to .rai-check/history/"),
) -> None:
    """Audit a captured canonical agent execution trace."""
    try:
        captured_trace = load_trace(trace)
        allowlist = (
            [tool.strip() for tool in allowed_tools.split(",") if tool.strip()]
            if allowed_tools
            else None
        )
        report = AgentAudit(
            captured_trace,
            allowed_tools=allowlist,
            project_name=project,
            persist=persist,
        ).run()
    except TraceValidationError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    if format == "html":
        report.to_html(str(out))
    elif format == "markdown":
        out = out.with_suffix(".md")
        report.to_markdown(str(out))
    elif format == "json":
        out = out.with_suffix(".json")
        report.to_json(str(out))
    else:
        console.print(f"[red]Unknown format:[/red] {format}")
        raise typer.Exit(1)

    active = sum(
        1 for finding in report.findings if finding.severity.value not in ("passed", "info")
    )
    console.print(
        f"[green]Audit complete.[/green] "
        f"Overall risk: [bold]{report.overall_risk_level.value.upper()}[/bold] "
        f"| Findings: {active} | Report: {out}"
    )
