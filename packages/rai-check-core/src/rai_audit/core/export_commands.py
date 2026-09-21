from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rai_audit.core.history import load_run, write_history_dashboard
from rai_audit.core.monitoring import write_eu_ai_act_post_market_report


def register_export_commands(export_app: typer.Typer, console: Any) -> None:
    """Register history and standards export commands on the core CLI."""

    @export_app.command("standards-coverage")
    def standards_coverage(
        input: Path = typer.Argument(..., help="Path to saved audit JSON run"),
        output: Path | None = typer.Option(
            None,
            help="Output .json or .md path (default: <input>.standards-coverage.json)",
        ),
        required_ref: list[str] = typer.Option(
            [],
            "--required-ref",
            help="Standards reference to require; repeat to define a custom coverage set",
        ),
    ) -> None:
        """Export mapped and missing standards evidence without a compliance claim."""
        if not input.exists():
            console.print(f"[red]Error:[/red] {input} not found")
            raise typer.Exit(1)
        if output is None:
            output = input.with_suffix(".standards-coverage.json")
        from rai_audit.core.cli import _dict_to_report

        report = _dict_to_report(load_run(input))
        report.to_standards_coverage(str(output), required_refs=required_ref or None)
        console.print(f"[green]Done.[/green] Standards coverage written to {output}")

    @export_app.command("history-dashboard")
    def history_dashboard(
        directory: Path = typer.Option(Path(".rai-check/history"), help="History directory"),
        output: Path = typer.Option(Path("audit-history.html"), help="Output HTML path"),
        project: str | None = typer.Option(None, help="Optional project-name filter"),
    ) -> None:
        """Export an HTML audit-history dashboard with trends and regressions."""
        write_history_dashboard(output, directory, project_name=project)
        console.print(f"[green]Done.[/green] History dashboard written to {output}")

    @export_app.command("eu-post-market")
    def eu_post_market(
        directory: Path = typer.Option(Path(".rai-check/history"), help="History directory"),
        output: Path = typer.Option(
            Path("eu-ai-act-post-market.md"),
            help="Output .md or .json path",
        ),
        project: str | None = typer.Option(None, help="Optional project-name filter"),
    ) -> None:
        """Export an EU AI Act-oriented post-market report from audit history."""
        write_eu_ai_act_post_market_report(output, directory, project_name=project)
        console.print(f"[green]Done.[/green] EU post-market report written to {output}")
