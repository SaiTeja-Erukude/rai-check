"""Log a rai-check drift report to an active MLflow run."""

from pathlib import Path

import mlflow
from rai_audit.core.findings import AuditReport, Severity


def log_drift_report(report: AuditReport, artifact_dir: str = "rai-check") -> None:
    """Record monitoring metrics and the full JSON report in MLflow."""
    active_findings = [
        finding
        for finding in report.findings
        if finding.severity not in (Severity.PASSED, Severity.INFO)
    ]
    report_path = Path("drift-report.json")
    report.to_json(str(report_path))

    mlflow.log_metrics(
        {
            "rai_audit.active_findings": len(active_findings),
            "rai_audit.high_findings": len(report.high_findings),
            "rai_audit.critical_findings": len(report.critical_findings),
        }
    )
    mlflow.log_artifact(str(report_path), artifact_path=artifact_dir)
