from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rai_audit.core.engine import BaseAudit
from rai_audit.core.findings import AuditFinding, AuditReport, RemediationEffort, Severity
from rai_audit.core.history import save_run
from rai_audit.core.scoring import compute_risk_matrix
from rai_audit.ml.data_quality import data_quality_findings

_REG_STANDARDS = ["EU-AI-ACT-ART-10", "NIST-AI-RMF-MEASURE-2.5"]


def _group_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    group_values: np.ndarray,
) -> list[dict]:
    results = []
    for val in np.unique(group_values):
        mask = group_values == val
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) < 5:
            continue
        residuals = yt - yp
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        results.append(
            {
                "group_col": group_col,
                "group_val": str(val),
                "n": int(mask.sum()),
                "mae": round(float(mean_absolute_error(yt, yp)), 4),
                "rmse": round(rmse, 4),
                "r2": round(float(r2_score(yt, yp)), 4),
                "residual_mean": round(float(residuals.mean()), 4),
                "residual_std": round(float(residuals.std()), 4),
            }
        )
    return results


def regression_fairness_findings(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
    max_mae_ratio: float = 1.5,
    max_residual_mean_diff: float = 0.1,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    for col in sensitive_features.columns:
        group_vals = sensitive_features[col].astype(str).values
        group_metrics = _group_regression_metrics(y_true, y_pred, col, group_vals)

        if len(group_metrics) < 2:
            continue

        # MAE ratio (worst / best)
        maes = {m["group_val"]: m["mae"] for m in group_metrics}
        max_mae = max(maes.values())
        min_mae = min(maes.values())
        mae_ratio = max_mae / min_mae if min_mae > 0 else float("inf")

        severity = Severity.HIGH if mae_ratio > max_mae_ratio * 1.5 else (
            Severity.MEDIUM if mae_ratio > max_mae_ratio else Severity.PASSED
        )
        worst_group = max(maes, key=maes.get)
        best_group = min(maes, key=maes.get)

        findings.append(
            AuditFinding(
                check_id="FAIR-REG-001",
                title=f"MAE disparity across groups in '{col}'",
                severity=severity,
                description=(
                    f"MAE ratio (worst/best group) in '{col}' is {mae_ratio:.2f} "
                    f"(threshold: {max_mae_ratio}). "
                    f"Worst: {worst_group} (MAE={max_mae:.4f}), "
                    f"best: {best_group} (MAE={min_mae:.4f})."
                ),
                evidence={"mae_by_group": maes, "mae_ratio": round(mae_ratio, 4)},
                recommendation=(
                    "The model has higher prediction error for some groups. "
                    "Review training data representation and consider group-aware calibration."
                ),
                category="Fairness",
                affected_group=col,
                remediation_effort=RemediationEffort.HIGH,
                standards_refs=_REG_STANDARDS,
            )
        )

        # Residual mean (systematic bias per group)
        residual_means = {m["group_val"]: m["residual_mean"] for m in group_metrics}
        res_diff = max(residual_means.values()) - min(residual_means.values())
        severity = Severity.HIGH if res_diff > max_residual_mean_diff * 2 else (
            Severity.MEDIUM if res_diff > max_residual_mean_diff else Severity.PASSED
        )

        findings.append(
            AuditFinding(
                check_id="FAIR-REG-002",
                title=f"Systematic prediction bias by group in '{col}'",
                severity=severity,
                description=(
                    f"Residual mean difference across groups in '{col}' is {res_diff:.4f}. "
                    "Positive residual mean = model underpredicts; "
                    "negative = model overpredicts."
                ),
                evidence={"residual_mean_by_group": residual_means, "residual_diff": round(res_diff, 4)},
                recommendation=(
                    "Systematic underprediction or overprediction for a group indicates bias. "
                    "Consider bias correction or group-stratified model fitting."
                ),
                category="Fairness",
                affected_group=col,
                remediation_effort=RemediationEffort.HIGH,
                standards_refs=_REG_STANDARDS,
            )
        )

    return findings


class RegressionAudit(BaseAudit):
    """Full audit for tabular regression models."""

    def __init__(
        self,
        y_true,
        y_pred,
        sensitive_features: pd.DataFrame | None = None,
        data: pd.DataFrame | None = None,
        project_name: str = "Regression Audit",
        metadata: dict | None = None,
        thresholds: dict | None = None,
        persist: bool = True,
    ):
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_pred = np.asarray(y_pred, dtype=float)
        self.sensitive_features = sensitive_features
        self.data = data
        self.project_name = project_name
        self.metadata = metadata or {}
        self.thresholds = thresholds or {}
        self.persist = persist

    def run(self) -> AuditReport:
        findings: list[AuditFinding] = []
        ts = datetime.now(timezone.utc).isoformat()

        residuals = self.y_true - self.y_pred
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = float(np.sqrt(mean_squared_error(self.y_true, self.y_pred)))
        r2 = r2_score(self.y_true, self.y_pred)

        findings.append(
            AuditFinding(
                check_id="REG-PERF-001",
                title="Overall regression performance",
                severity=Severity.INFO,
                description="Overall regression metrics across all samples.",
                evidence={
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                    "residual_mean": round(float(residuals.mean()), 4),
                    "residual_std": round(float(residuals.std()), 4),
                    "n_samples": len(self.y_true),
                },
                recommendation="",
                category="Performance",
                timestamp=ts,
            )
        )

        if self.sensitive_features is not None:
            max_mae_ratio = self.thresholds.get("max_mae_ratio", 1.5)
            max_res_diff = self.thresholds.get("max_residual_mean_diff", 0.1)
            findings.extend(
                regression_fairness_findings(
                    self.y_true,
                    self.y_pred,
                    self.sensitive_features,
                    max_mae_ratio=max_mae_ratio,
                    max_residual_mean_diff=max_res_diff,
                )
            )

        if self.data is not None:
            findings.extend(data_quality_findings(self.data))

        risk_matrix = compute_risk_matrix(findings)
        meta = {
            **self.metadata,
            "n_samples": len(self.y_true),
            "sensitive_features": list(self.sensitive_features.columns) if self.sensitive_features is not None else [],
        }

        report = AuditReport(
            project_name=self.project_name,
            audit_type="tabular_regression",
            risk_matrix=risk_matrix,
            findings=findings,
            metadata=meta,
        )

        if self.persist:
            save_run(report.to_dict())

        return report
