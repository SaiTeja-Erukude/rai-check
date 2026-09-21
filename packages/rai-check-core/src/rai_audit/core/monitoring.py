from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rai_audit.core.history import build_history_summary, load_run
from rai_audit.core.standards import NON_COMPLIANCE_CLAIM, describe_ref

_EU_AI_ACT_REFS = {
    "EU-AI-ACT-ART-9",
    "EU-AI-ACT-ART-10",
    "EU-AI-ACT-ART-13",
    "EU-AI-ACT-ART-14",
    "EU-AI-ACT-ART-15",
}


def build_eu_ai_act_post_market_report(
    directory: Path | None = None,
    *,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Build an EU AI Act-oriented post-market monitoring report from audit history."""
    history = build_history_summary(directory, project_name=project_name)
    run_details = [(run, load_run(Path(run["path"]))) for run in history["runs"]]
    incidents = []
    for summary, run in run_details:
        metadata = run.get("metadata", {})
        annotations = metadata.get("incident_annotations", metadata.get("incidents", []))
        for annotation in annotations if isinstance(annotations, list) else []:
            incidents.append(
                {
                    "run_timestamp": summary["timestamp"],
                    "project_name": summary["project_name"],
                    "annotation": annotation,
                }
            )

    latest_runs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for summary, run in run_details:
        latest_runs[summary["project_name"]] = (summary, run)
    active_findings = []
    for summary, run in latest_runs.values():
        for finding in run.get("findings", []):
            if finding.get("severity") in {"passed", "info"}:
                continue
            active_findings.append(
                {
                    "project_name": summary["project_name"],
                    "run_timestamp": summary["timestamp"],
                    "check_id": finding.get("check_id", "unknown"),
                    "title": finding.get("title", ""),
                    "severity": finding.get("severity", "unknown"),
                    "category": finding.get("category", ""),
                }
            )

    eu_evidence = _eu_ai_act_evidence(run for _, run in run_details)
    timestamps = [run["timestamp"] for run in history["runs"]]
    return {
        "report_type": "eu_ai_act_post_market_monitoring",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name,
        "disclaimer": NON_COMPLIANCE_CLAIM,
        "reporting_period": {
            "start": min(timestamps) if timestamps else None,
            "end": max(timestamps) if timestamps else None,
        },
        "summary": {
            "run_count": history["summary"]["run_count"],
            "project_count": history["summary"]["project_count"],
            "regression_count": history["summary"]["regression_count"],
            "incident_count": len(incidents),
            "active_finding_count": len(active_findings),
        },
        "incidents": incidents,
        "regressions": history["regressions"],
        "active_findings": active_findings,
        "eu_ai_act_evidence": eu_evidence,
        "audit_runs": history["runs"],
    }


def render_eu_ai_act_post_market_markdown(report: dict[str, Any]) -> str:
    """Render an EU AI Act-oriented post-market monitoring report as Markdown."""
    summary = report["summary"]
    period = report["reporting_period"]
    lines = [
        "# EU AI Act Post-Market Monitoring Report",
        "",
        f"> {report['disclaimer']}",
        "",
        "## Reporting Period",
        "",
        f"- Start: {period['start'] or 'No persisted runs'}",
        f"- End: {period['end'] or 'No persisted runs'}",
        "",
        "## Monitoring Summary",
        "",
        f"- Audit runs reviewed: {summary['run_count']}",
        f"- Projects reviewed: {summary['project_count']}",
        f"- Risk regressions: {summary['regression_count']}",
        f"- Incident annotations: {summary['incident_count']}",
        f"- Active findings in latest runs: {summary['active_finding_count']}",
        "",
        "## EU AI Act Evidence Coverage",
        "",
        "| Reference | Status | Mapped Checks |",
        "|-----------|--------|---------------|",
    ]
    for item in report["eu_ai_act_evidence"]:
        lines.append(
            f"| `{item['reference']}` | {item['status']} "
            f"| {', '.join(item['mapped_checks']) or '-'} |"
        )

    lines.extend(["", "## Risk Regressions", ""])
    if report["regressions"]:
        lines.extend(
            f"- {item['timestamp']} `{item['project_name']}`: {item['category']} "
            f"{item['from']} -> {item['to']}"
            for item in report["regressions"]
        )
    else:
        lines.append("- No category risk regressions recorded.")

    lines.extend(["", "## Incident Annotations", ""])
    if report["incidents"]:
        lines.extend(
            f"- {item['run_timestamp']} `{item['project_name']}`: {item['annotation']}"
            for item in report["incidents"]
        )
    else:
        lines.append("- No incident annotations recorded.")
    return "\n".join(lines).rstrip() + "\n"


def write_eu_ai_act_post_market_report(
    path: str | Path,
    directory: Path | None = None,
    *,
    project_name: str | None = None,
) -> None:
    """Write a JSON or Markdown EU AI Act-oriented post-market monitoring report."""
    output = Path(path)
    report = build_eu_ai_act_post_market_report(directory, project_name=project_name)
    if output.suffix.lower() in {".md", ".markdown"}:
        output.write_text(render_eu_ai_act_post_market_markdown(report), encoding="utf-8")
    else:
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _eu_ai_act_evidence(runs: Any) -> list[dict[str, Any]]:
    evidence_by_ref = {ref: set() for ref in _EU_AI_ACT_REFS}
    for run in runs:
        for finding in run.get("findings", []):
            for ref in finding.get("standards_refs", []):
                if ref in evidence_by_ref:
                    evidence_by_ref[ref].add(str(finding.get("check_id", "unknown")))
    return [
        {
            "reference": ref,
            "description": describe_ref(ref),
            "status": "evidence_recorded" if checks else "missing_evidence",
            "mapped_checks": sorted(checks),
        }
        for ref, checks in sorted(evidence_by_ref.items())
    ]
