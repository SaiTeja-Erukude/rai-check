from __future__ import annotations

from pathlib import Path

import typer
from rai_audit.genai.llm.audit import LLMAudit, RAGAudit, RAGSecurityAudit
from rai_audit.genai.llm.loader import SuiteValidationError, load_test_suite
from rich.console import Console

console = Console()
llm_app = typer.Typer(name="llm", help="Audit LLM applications and RAG systems.")


def register(parent: typer.Typer) -> None:
    parent.add_typer(llm_app)


@llm_app.command("run")
def llm_run(
    suite: Path = typer.Option(..., help="YAML suite with prompts and captured responses"),
    audit_type: str = typer.Option("llm", help="Audit type: llm | rag | rag-security"),
    project: str | None = typer.Option(None, help="Project name override"),
    out: Path = typer.Option(Path("llm_audit_report.html"), help="Output report path"),
    format: str = typer.Option("html", help="Report format: html | markdown | json"),
    persist: bool = typer.Option(True, help="Save run to .rai-check/history/"),
) -> None:
    """Audit captured LLM responses from a YAML test suite."""
    try:
        test_suite = load_test_suite(suite)
        audit_class = {
            "llm": LLMAudit,
            "rag": RAGAudit,
            "rag-security": RAGSecurityAudit,
        }[audit_type]
        report = audit_class(test_suite, project_name=project, persist=persist).run()
    except KeyError:
        console.print(f"[red]Unknown audit type:[/red] {audit_type}")
        raise typer.Exit(1) from None
    except (SuiteValidationError, ValueError) as exc:
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
        f"[green]Audit complete.[/green] Overall risk: "
        f"[bold]{report.overall_risk_level.value.upper()}[/bold] "
        f"| Findings: {active} | Report: {out}"
    )
