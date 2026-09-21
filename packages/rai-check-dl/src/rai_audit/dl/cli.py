from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rai_audit.dl.image import ImageClassificationAudit
from rai_audit.dl.medical import MedicalImagingAudit
from rai_audit.dl.scientific import ScientificAIAudit
from rich.console import Console

console = Console()
dl_app = typer.Typer(name="dl", help="Audit image, medical imaging, and scientific AI models.")


def register(parent: typer.Typer) -> None:
    parent.add_typer(dl_app)


@dl_app.command("run")
def dl_run(
    data: Path = typer.Option(..., help="CSV with labels, predictions, and optional metadata"),
    target: str = typer.Option("y_true", help="Column name for true labels"),
    prediction: str = typer.Option("y_pred", help="Column name for predictions"),
    task: str = typer.Option("image", help="Audit type: image | medical | scientific"),
    transformed_prefix: str = typer.Option(
        "transform_",
        help="Prefix for columns containing predictions under transformations",
    ),
    patient_id: str | None = typer.Option(
        None,
        help="Patient identifier column for medical audits",
    ),
    split: str | None = typer.Option(None, help="Dataset split column for medical audits"),
    site: str | None = typer.Option(None, help="Collection-site column for medical audits"),
    domain: str = typer.Option("scientific imaging", help="Domain label for scientific audits"),
    project: str = typer.Option("DL Audit", help="Project name"),
    out: Path = typer.Option(Path("dl_audit_report.html"), help="Output report path"),
    format: str = typer.Option("html", help="Report format: html | markdown | json"),
    persist: bool = typer.Option(True, help="Save run to .rai-check/history/"),
) -> None:
    """Audit recorded image classification predictions from a CSV file."""
    if not data.exists():
        console.print(f"[red]Error:[/red] {data} not found")
        raise typer.Exit(1)
    frame = pd.read_csv(data)
    _require_columns(frame, [target, prediction])
    transformed_predictions = {
        column.removeprefix(transformed_prefix): frame[column].values
        for column in frame.columns
        if column.startswith(transformed_prefix)
    }
    common = {
        "y_true": frame[target].values,
        "y_pred": frame[prediction].values,
        "transformed_predictions": transformed_predictions,
        "project_name": project,
        "persist": persist,
    }
    if task == "image":
        audit = ImageClassificationAudit(**common)
    elif task == "medical":
        _require_columns(frame, [value for value in (patient_id, split, site) if value])
        audit = MedicalImagingAudit(
            **common,
            patient_ids=frame[patient_id].values if patient_id else None,
            splits=frame[split].values if split else None,
            sites=frame[site].values if site else None,
        )
    elif task == "scientific":
        audit = ScientificAIAudit(**common, scientific_domain=domain)
    else:
        console.print(f"[red]Unknown task:[/red] {task}")
        raise typer.Exit(1)

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

    active = sum(
        1 for finding in report.findings if finding.severity.value not in ("passed", "info")
    )
    console.print(
        f"[green]Audit complete.[/green] "
        f"Overall risk: [bold]{report.overall_risk_level.value.upper()}[/bold] "
        f"| Findings: {active} | Report: {out}"
    )


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        console.print(f"[red]Error:[/red] Columns not found: {missing}")
        raise typer.Exit(1)
