from __future__ import annotations

import json
from pathlib import Path

import typer
from rai_audit.core.export_commands import register_export_commands
from rai_audit.core.history import (
    diff_runs,
    list_runs,
    load_run,
    render_diff_text,
)
from rai_audit.core.scoring import gate_check
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="rai-audit",
    help="Responsible AI (RAI) Audit Kit — evidence-grade audits for responsible AI systems.",
    no_args_is_help=True,
)
export_app = typer.Typer(no_args_is_help=True, help="Export audit results to different formats.")
app.add_typer(export_app, name="export")
console = Console()
register_export_commands(export_app, console)


@app.command()
def init(
    project: str = typer.Option("my-project", help="Project name"),
    output: Path = typer.Option(Path("audit.yaml"), help="Output config path"),
    sample_data: bool = typer.Option(
        True,
        "--sample-data/--no-sample-data",
        help="Write a sample predictions.csv beside the config",
    ),
) -> None:
    """Scaffold a starter audit.yaml config file."""
    config = f"""schema_version: "1.0"

project:
  name: {project}
  owner: ""
  version: 0.1.0

audit:
  type: classification
  data: predictions.csv
  target: y_true
  prediction: y_pred
  sensitive_features: []
  output_dir: ./audit-report
  report_formats:
    - html
    - markdown
    - json

checks:
  fairness:
    enabled: true
    max_demographic_parity_diff: 0.10
    max_equal_opportunity_diff: 0.10
  robustness:
    enabled: true
  data_quality:
    enabled: true
  privacy:
    enabled: true
  reproducibility:
    enabled: true
  drift:
    enabled: false

gate:
  min_score: null
  fail_on_critical: true
"""
    output.write_text(config, encoding="utf-8")
    console.print(f"[green]✓[/green] Created {output}")
    if sample_data:
        sample_path = output.parent / "predictions.csv"
        if sample_path.exists():
            console.print(f"[yellow]Skipped sample data; {sample_path} already exists[/yellow]")
        else:
            sample_path.write_text(_starter_predictions_csv(), encoding="utf-8")
            console.print(f"[green]✓[/green] Created {sample_path}")


def _starter_predictions_csv() -> str:
    return """y_true,y_pred,group,feature_income,feature_tenure_months,feature_score
0,0,A,52000,12,0.32
1,1,A,61000,28,0.58
0,0,B,84000,43,0.81
1,1,B,47000,9,0.27
0,1,C,76000,36,0.74
1,1,C,68000,18,0.35
0,0,A,91000,51,0.49
1,0,A,56000,24,0.87
0,0,B,73000,15,0.22
1,1,B,95000,47,0.41
0,0,C,50000,55,0.91
1,1,C,81000,21,0.38
0,0,A,88000,39,0.79
1,1,A,59000,11,0.29
0,0,B,79000,34,0.46
1,0,B,64000,26,0.84
0,0,C,99000,45,0.33
1,1,C,71000,17,0.62
0,1,A,90000,60,0.78
1,1,A,54000,19,0.31
0,0,B,83000,49,0.76
1,1,B,66000,14,0.36
0,0,C,87000,41,0.82
1,1,C,62000,23,0.52
"""


@app.command("run")
def run_from_config(
    config: Path = typer.Option(Path("audit.yaml"), help="Audit YAML config path"),
    enforce_gate: bool = typer.Option(True, help="Exit with code 1 when configured gate fails"),
) -> None:
    """Run a configured audit and write reports plus an evidence manifest."""
    from rai_audit.core.config import ConfigValidationError, run_config

    try:
        result = run_config(config)
    except (ConfigValidationError, ImportError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    console.print(
        f"[green]Audit complete.[/green] "
        f"Overall risk: [bold]{result.report.overall_risk_level.value.upper()}[/bold]"
    )
    for format_name, path in result.artifacts.items():
        console.print(f"{format_name}: {path}", soft_wrap=True)
    console.print(f"evidence: {result.manifest_path}", soft_wrap=True)
    if result.gate_passed:
        console.print(f"[green]GATE PASSED:[/green] {result.gate_reason}")
    else:
        console.print(f"[red]GATE FAILED:[/red] {result.gate_reason}")
        if enforce_gate:
            raise typer.Exit(1)


@app.command()
def report(
    input: Path = typer.Argument(..., help="Path to a saved audit JSON run"),
    format: str = typer.Option(
        "html",
        help="Output format: html | markdown | json | sarif | junit",
    ),
    output: Path | None = typer.Option(None, help="Output file path"),
) -> None:
    """Render a report from a saved audit run JSON file."""
    if not input.exists():
        console.print(f"[red]Error:[/red] {input} not found")
        raise typer.Exit(1)

    run = load_run(input)

    if output is None:
        output = input.with_suffix(f".{format}" if format != "json" else ".out.json")

    if format == "json":
        output.write_text(json.dumps(run, indent=2), encoding="utf-8")
    elif format == "markdown":
        from rai_audit.core.report import render_markdown

        report_obj = _dict_to_report(run)
        output.write_text(render_markdown(report_obj), encoding="utf-8")
    elif format == "html":
        from rai_audit.core.report import render_html
        report_obj = _dict_to_report(run)
        output.write_text(render_html(report_obj), encoding="utf-8")
    elif format == "sarif":
        from rai_audit.core.ci_formats import render_sarif

        report_obj = _dict_to_report(run)
        output.write_text(render_sarif(report_obj), encoding="utf-8")
    elif format == "junit":
        from rai_audit.core.ci_formats import render_junit

        report_obj = _dict_to_report(run)
        output.write_text(render_junit(report_obj), encoding="utf-8")
    else:
        console.print(f"[red]Unknown format:[/red] {format}")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Report written to {output}")


@app.command()
def gate(
    input: Path = typer.Argument(..., help="Path to saved audit JSON run"),
    min_score: float | None = typer.Option(None, help="Minimum required score"),
    fail_on_critical: bool = typer.Option(True, help="Fail if any critical findings exist"),
    output_json: Path | None = typer.Option(None, help="Write gate result to JSON file"),
) -> None:
    """
    CI/CD deployment gate. Exits with code 1 on failure, 0 on pass.
    """
    if not input.exists():
        console.print(f"[red]Error:[/red] {input} not found")
        raise typer.Exit(1)

    run = load_run(input)
    passed, reason = gate_check(run, min_score=min_score, fail_on_critical=fail_on_critical)

    risk_matrix = {r["category"]: r["risk_level"] for r in run.get("risk_matrix", [])}
    critical_count = sum(
        1 for f in run.get("findings", []) if f.get("severity") == "critical"
    )

    result = {
        "passed": passed,
        "reason": reason,
        "critical_count": critical_count,
        "risk_matrix": risk_matrix,
    }

    if output_json:
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if passed:
        console.print(f"[green]✓ GATE PASSED:[/green] {reason}")
    else:
        console.print(f"[red]✗ GATE FAILED:[/red] {reason}")
        raise typer.Exit(1)


@app.command()
def diff(
    run_a: Path = typer.Argument(..., help="Older audit run JSON"),
    run_b: Path = typer.Argument(..., help="Newer audit run JSON"),
    output_json: Path | None = typer.Option(None, help="Write diff to JSON file"),
) -> None:
    """Compare two audit runs and show what changed."""
    for p in [run_a, run_b]:
        if not p.exists():
            console.print(f"[red]Error:[/red] {p} not found")
            raise typer.Exit(1)

    result = diff_runs(run_a, run_b)

    if output_json:
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    console.print(render_diff_text(result))


@app.command()
def history(
    directory: Path = typer.Option(Path(".rai-audit/history"), help="History directory"),
) -> None:
    """List past audit runs."""
    runs = list_runs(directory)
    if not runs:
        console.print("No audit runs found.")
        return

    table = Table(title="Audit History")
    table.add_column("File", style="cyan")
    table.add_column("Project")
    table.add_column("Risk")
    table.add_column("Findings", justify="right")

    for run_path in runs:
        try:
            run = load_run(run_path)
            risk_levels = [r["risk_level"] for r in run.get("risk_matrix", [])]
            worst = (
                max(risk_levels, key=lambda r: ["low", "medium", "high", "critical"].index(r))
                if risk_levels
                else "n/a"
            )
            count = sum(
                1
                for f in run.get("findings", [])
                if f.get("severity") not in ("passed", "info")
            )
            table.add_row(run_path.name, run.get("project_name", "?"), worst.upper(), str(count))
        except Exception:
            table.add_row(run_path.name, "?", "?", "?")

    console.print(table)


def _dict_to_report(d: dict):
    """Reconstruct an AuditReport from a saved dict (for re-rendering)."""
    from rai_audit.core.findings import (
        AuditFinding,
        AuditReport,
        CategoryRisk,
        RemediationEffort,
        RiskLevel,
        Severity,
    )

    findings = [
        AuditFinding(
            check_id=f["check_id"],
            title=f["title"],
            severity=Severity(f["severity"]),
            description=f["description"],
            evidence=f.get("evidence", {}),
            recommendation=f.get("recommendation", ""),
            category=f.get("category", ""),
            affected_group=f.get("affected_group"),
            remediation_effort=RemediationEffort(f.get("remediation_effort", "medium")),
            standards_refs=f.get("standards_refs", []),
            timestamp=f.get("timestamp"),
        )
        for f in d.get("findings", [])
    ]

    risk_matrix = [
        CategoryRisk(
            category=r["category"],
            risk_level=RiskLevel(r["risk_level"]),
            finding_count=r["finding_count"],
            passed_count=r["passed_count"],
        )
        for r in d.get("risk_matrix", [])
    ]

    return AuditReport(
        project_name=d.get("project_name", "Audit"),
        audit_type=d.get("audit_type", ""),
        risk_matrix=risk_matrix,
        findings=findings,
        metadata=d.get("metadata", {}),
        overall_score=d.get("overall_score"),
    )


@export_app.command("model-card")
def model_card(
    input: Path = typer.Argument(..., help="Path to saved audit JSON run"),
    output: Path | None = typer.Option(
        None,
        help="Output .md file path (default: <input>.model-card.md)",
    ),
    model_name: str = typer.Option("", help="Model display name"),
    model_version: str = typer.Option("", help="Model version string"),
    author: str = typer.Option("", help="Author / team name"),
    license_id: str = typer.Option("MIT", help="SPDX license identifier"),
    language: str = typer.Option("en", help="ISO 639-1 language code"),
) -> None:
    """Export an audit run as a Markdown model card (HuggingFace-compatible)."""
    if not input.exists():
        console.print(f"[red]Error:[/red] {input} not found")
        raise typer.Exit(1)

    run = load_run(input)
    report_obj = _dict_to_report(run)

    if output is None:
        output = input.with_suffix(".model-card.md")

    from rai_audit.core.model_card import render_model_card

    card_text = render_model_card(
        report_obj,
        model_name=model_name,
        model_version=model_version,
        author=author,
        license_id=license_id,
        language=language,
    )
    output.write_text(card_text, encoding="utf-8")
    console.print(f"[green]✓[/green] Model card written to {output}")
