from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()
ml_app = typer.Typer(name="ml", help="Audit traditional ML models (classification, regression).")


def register(parent: typer.Typer) -> None:
    """Register the ml subcommand on the parent CLI app."""
    parent.add_typer(ml_app)


@ml_app.command("run")
def ml_run(
    data: Path = typer.Option(..., help="CSV file with y_true, y_pred, and optional columns"),
    target: str = typer.Option("y_true", help="Column name for true labels"),
    prediction: str = typer.Option("y_pred", help="Column name for predictions"),
    probability: Optional[str] = typer.Option(None, help="Column name for predicted probabilities"),
    sensitive: Optional[str] = typer.Option(None, help="Comma-separated sensitive feature columns"),
    task: str = typer.Option("classification", help="Task type: classification | regression"),
    project: str = typer.Option("ML Audit", help="Project name"),
    out: Path = typer.Option(Path("audit_report.html"), help="Output report path"),
    format: str = typer.Option("html", help="Report format: html | markdown | json"),
    persist: bool = typer.Option(True, help="Save run to .rai-check/history/"),
) -> None:
    """Run an ML audit from a predictions CSV file."""
    import pandas as pd
    from rai_audit.core.report import render_html, render_markdown
    from rai_audit.ml.classification import ClassificationAudit
    from rai_audit.ml.regression import RegressionAudit

    if not data.exists():
        console.print(f"[red]Error:[/red] {data} not found")
        raise typer.Exit(1)

    df = pd.read_csv(data)
    for col in [target, prediction]:
        if col not in df.columns:
            console.print(f"[red]Error:[/red] Column '{col}' not found in {data}")
            raise typer.Exit(1)

    y_true = df[target].values
    y_pred = df[prediction].values
    y_prob = df[probability].values if probability and probability in df.columns else None

    sensitive_df = None
    if sensitive:
        sens_cols = [c.strip() for c in sensitive.split(",")]
        missing = [c for c in sens_cols if c not in df.columns]
        if missing:
            console.print(f"[red]Error:[/red] Sensitive columns not found: {missing}")
            raise typer.Exit(1)
        sensitive_df = df[sens_cols]

    if task == "classification":
        audit = ClassificationAudit(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            sensitive_features=sensitive_df,
            project_name=project,
            persist=persist,
        )
    elif task == "regression":
        audit = RegressionAudit(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_df,
            project_name=project,
            persist=persist,
        )
    else:
        console.print(f"[red]Unknown task:[/red] {task}. Use 'classification' or 'regression'.")
        raise typer.Exit(1)

    console.print(f"Running [bold]{task}[/bold] audit for [cyan]{project}[/cyan]...")
    report = audit.run()

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

    risk = report.overall_risk_level.value.upper()
    active = sum(1 for f in report.findings if f.severity.value not in ("passed", "info"))
    passed = sum(1 for f in report.findings if f.severity.value == "passed")
    console.print(
        f"[green]✓ Audit complete.[/green] "
        f"Overall risk: [bold]{risk}[/bold] | "
        f"Findings: {active} | Passed: {passed}"
    )
    console.print(f"Report: {out}")
