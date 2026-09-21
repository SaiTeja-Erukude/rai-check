from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from rai_audit.core.schemas import prepare_document

DEFAULT_HISTORY_DIR = Path(".rai-check") / "history"
_RISK_ORDER = {"n/a": -1, "low": 0, "medium": 1, "high": 2, "critical": 3}


def save_run(report_dict: dict, directory: Path | None = None) -> Path:
    """Persist an audit run as a timestamped JSON file."""
    report_dict = prepare_document("report", report_dict)
    directory = directory or DEFAULT_HISTORY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    slug = report_dict.get("project_name", "audit").replace(" ", "_").lower()
    path = directory / f"{slug}_{ts}.json"
    path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    return path


def load_run(path: Path) -> dict:
    """Load a persisted audit run."""
    return prepare_document("report", json.loads(Path(path).read_text(encoding="utf-8")))


def list_runs(directory: Path | None = None) -> list[Path]:
    """List all persisted audit run files, newest first."""
    directory = directory or DEFAULT_HISTORY_DIR
    if not directory.exists():
        return []
    return sorted(
        directory.glob("*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )


def diff_runs(path_a: Path, path_b: Path) -> dict:
    """
    Compare two audit runs and return a structured diff.
    path_a is the older run, path_b is the newer run.
    """
    a = load_run(Path(path_a))
    b = load_run(Path(path_b))

    risk_a = {r["category"]: r["risk_level"] for r in a.get("risk_matrix", [])}
    risk_b = {r["category"]: r["risk_level"] for r in b.get("risk_matrix", [])}

    risk_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
    risk_changes: list[dict] = []
    all_cats = sorted(set(risk_a) | set(risk_b))
    for cat in all_cats:
        old = risk_a.get(cat, "n/a")
        new = risk_b.get(cat, "n/a")
        if old != new:
            old_rank = risk_order.get(old, -1)
            new_rank = risk_order.get(new, -1)
            direction = "REGRESSION" if new_rank > old_rank else "IMPROVED"
            risk_changes.append(
                {"category": cat, "from": old, "to": new, "direction": direction}
            )

    ids_a = {f["check_id"] for f in a.get("findings", []) if f.get("severity") != "passed"}
    ids_b = {f["check_id"] for f in b.get("findings", []) if f.get("severity") != "passed"}

    new_findings = [
        f for f in b.get("findings", [])
        if f["check_id"] not in ids_a and f.get("severity") != "passed"
    ]
    resolved_findings = [
        f for f in a.get("findings", [])
        if f["check_id"] not in ids_b and f.get("severity") != "passed"
    ]

    return {
        "project_name": b.get("project_name", "unknown"),
        "run_a": str(path_a),
        "run_b": str(path_b),
        "risk_changes": risk_changes,
        "new_findings": new_findings,
        "resolved_findings": resolved_findings,
        "summary": {
            "regressions": sum(1 for c in risk_changes if c["direction"] == "REGRESSION"),
            "improvements": sum(1 for c in risk_changes if c["direction"] == "IMPROVED"),
            "new_finding_count": len(new_findings),
            "resolved_finding_count": len(resolved_findings),
        },
    }


def render_diff_text(diff: dict) -> str:
    """Render a diff dict as a human-readable text report."""
    lines: list[str] = []
    lines.append(f"Audit Diff: {diff['project_name']}")
    lines.append("─" * 60)

    if diff["risk_changes"]:
        for c in diff["risk_changes"]:
            arrow = "⚠ REGRESSION" if c["direction"] == "REGRESSION" else "✓ IMPROVED"
            lines.append(f"{c['category']:30s}  {c['from'].upper()} → {c['to'].upper()}    {arrow}")
    else:
        lines.append("No risk level changes.")

    if diff["new_findings"]:
        lines.append("\nNew findings:")
        for f in diff["new_findings"]:
            lines.append(f"  [{f['severity'].upper()}] {f['check_id']}: {f['title']}")

    if diff["resolved_findings"]:
        lines.append("\nResolved findings:")
        for f in diff["resolved_findings"]:
            lines.append(f"  ✓ {f['check_id']}: {f['title']}")

    s = diff["summary"]
    lines.append(
        f"\nSummary: {s['regressions']} regression(s), {s['improvements']} improvement(s), "
        f"{s['new_finding_count']} new, {s['resolved_finding_count']} resolved."
    )
    return "\n".join(lines)


def build_history_summary(
    directory: Path | None = None,
    *,
    project_name: str | None = None,
) -> dict[str, Any]:
    """Summarize persisted audit history for dashboards and monitoring reports."""
    runs = []
    for path in reversed(list_runs(directory)):
        run = load_run(path)
        if project_name is not None and run.get("project_name") != project_name:
            continue
        runs.append(_summarize_run(path, run))

    regressions = []
    runs_by_project: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        runs_by_project.setdefault(run["project_name"], []).append(run)
    for project_runs in runs_by_project.values():
        for older, newer in zip(project_runs, project_runs[1:]):
            diff = diff_runs(Path(older["path"]), Path(newer["path"]))
            for change in diff["risk_changes"]:
                if change["direction"] == "REGRESSION":
                    regressions.append(
                        {
                            "project_name": newer["project_name"],
                            "timestamp": newer["timestamp"],
                            "category": change["category"],
                            "from": change["from"],
                            "to": change["to"],
                            "run_path": newer["path"],
                        }
                    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": project_name,
        "summary": {
            "run_count": len(runs),
            "project_count": len(runs_by_project),
            "regression_count": len(regressions),
        },
        "runs": runs,
        "regressions": regressions,
    }


def render_history_dashboard(summary: dict[str, Any]) -> str:
    """Render an HTML dashboard with audit trends, regressions, and artifact links."""
    run_rows = []
    for run in reversed(summary["runs"]):
        artifacts = " ".join(
            f'<a href="{escape(link["href"], quote=True)}">{escape(link["label"])}</a>'
            for link in run["artifacts"]
        )
        run_rows.append(
            "<tr>"
            f"<td>{escape(run['timestamp'])}</td>"
            f"<td>{escape(run['project_name'])}</td>"
            f"<td>{escape(run['audit_type'])}</td>"
            f"<td class='risk risk-{escape(run['overall_risk'])}'>"
            f"{escape(run['overall_risk'].upper())}</td>"
            f"<td>{run['active_finding_count']}</td>"
            f"<td>{artifacts}</td>"
            "</tr>"
        )
    regression_rows = "".join(
        "<tr>"
        f"<td>{escape(item['timestamp'])}</td>"
        f"<td>{escape(item['project_name'])}</td>"
        f"<td>{escape(item['category'])}</td>"
        f"<td>{escape(item['from'].upper())} &rarr; {escape(item['to'].upper())}</td>"
        "</tr>"
        for item in reversed(summary["regressions"])
    )
    regression_section = (
        "<table><thead><tr><th>Run</th><th>Project</th><th>Category</th><th>Change</th>"
        f"</tr></thead><tbody>{regression_rows}</tbody></table>"
        if regression_rows
        else "<p>No category risk regressions recorded.</p>"
    )
    run_table_rows = "".join(run_rows) or "<tr><td colspan='6'>No audit runs found.</td></tr>"
    headline = summary["summary"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAI Check History Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2937; }}
h1, h2 {{ color: #111827; }}
.summary {{ display: flex; gap: 1rem; margin: 1rem 0 2rem; }}
.metric {{ padding: 0.8rem 1rem; background: #f3f4f6; border-radius: 0.4rem; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 0.6rem; text-align: left; }}
th {{ background: #f9fafb; }}
a {{ margin-right: 0.6rem; }}
.risk {{ font-weight: 700; }}
.risk-critical {{ color: #b91c1c; }} .risk-high {{ color: #c2410c; }}
.risk-medium {{ color: #a16207; }} .risk-low {{ color: #15803d; }}
</style>
</head>
<body>
<h1>RAI Check History Dashboard</h1>
<div class="summary">
  <div class="metric"><strong>{headline['run_count']}</strong><br>Runs</div>
  <div class="metric"><strong>{headline['project_count']}</strong><br>Projects</div>
  <div class="metric"><strong>{headline['regression_count']}</strong><br>Regressions</div>
</div>
<h2>Risk Trends</h2>
<table>
<thead><tr><th>Run</th><th>Project</th><th>Audit Type</th><th>Risk</th>
<th>Active Findings</th><th>Artifacts</th></tr></thead>
<tbody>{run_table_rows}</tbody>
</table>
<h2>Regressions</h2>
{regression_section}
</body>
</html>
"""


def write_history_dashboard(
    path: str | Path,
    directory: Path | None = None,
    *,
    project_name: str | None = None,
) -> None:
    """Write an HTML history dashboard."""
    Path(path).write_text(
        render_history_dashboard(build_history_summary(directory, project_name=project_name)),
        encoding="utf-8",
    )


def _summarize_run(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    risk_levels = [item["risk_level"] for item in run.get("risk_matrix", [])]
    overall_risk = max(risk_levels, key=lambda value: _RISK_ORDER.get(value, -1), default="n/a")
    findings = run.get("findings", [])
    return {
        "path": str(path),
        "timestamp": _run_timestamp(path, run),
        "project_name": str(run.get("project_name", "unknown")),
        "audit_type": str(run.get("audit_type", "unknown")),
        "overall_risk": overall_risk,
        "active_finding_count": sum(
            finding.get("severity") not in {"passed", "info"} for finding in findings
        ),
        "finding_counts": _finding_counts(findings),
        "artifacts": _artifact_links(path, run),
    }


def _run_timestamp(path: Path, run: dict[str, Any]) -> str:
    metadata = run.get("metadata", {})
    return str(
        metadata.get("monitored_at")
        or metadata.get("generated_at")
        or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    )


def _finding_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        severity = str(finding.get("severity", "unknown"))
        counts[severity] = counts.get(severity, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_links(path: Path, run: dict[str, Any]) -> list[dict[str, str]]:
    links = [{"label": "audit-json", "path": str(path), "href": _path_href(path)}]
    artifacts = run.get("metadata", {}).get("evidence_manifest", {}).get("artifacts", {})
    for label, value in sorted(artifacts.items()):
        artifact_path = Path(value["path"] if isinstance(value, dict) else value)
        links.append(
            {
                "label": str(label),
                "path": str(artifact_path),
                "href": _path_href(artifact_path),
            }
        )
    return links


def _path_href(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except ValueError:
        return str(path).replace("\\", "/")
