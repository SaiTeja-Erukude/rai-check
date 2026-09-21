from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rai_audit.core.engine import BaseAudit
from rai_audit.core.findings import AuditFinding, AuditReport, Severity
from rai_audit.core.history import save_run
from rai_audit.core.scoring import compute_risk_matrix
from rai_audit.ml.data_quality import data_quality_findings
from rai_audit.ml.explainability import explainability_findings, shap_explainability_findings
from rai_audit.ml.fairness import fairness_findings_classification
from rai_audit.ml.robustness import robustness_findings_classification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ClassificationAudit(BaseAudit):
    """
    Full audit for binary/multiclass tabular classification models.

    Runs fairness, data quality, and robustness checks and produces
    an AuditReport with a per-category risk matrix.
    """

    def __init__(
        self,
        y_true,
        y_pred,
        sensitive_features: pd.DataFrame | None = None,
        y_prob=None,
        data: pd.DataFrame | None = None,
        project_name: str = "Classification Audit",
        metadata: dict | None = None,
        thresholds: dict | None = None,
        positive_label=None,
        include_intersections: bool = False,
        feature_importances: dict[str, float] | None = None,
        explainability_model=None,
        persist: bool = True,
    ):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_prob = np.asarray(y_prob) if y_prob is not None else None
        self.sensitive_features = sensitive_features
        self.data = data
        self.project_name = project_name
        self.metadata = metadata or {}
        self.thresholds = thresholds or {}
        self.positive_label = positive_label
        self.include_intersections = include_intersections
        self.feature_importances = feature_importances
        self.explainability_model = explainability_model
        self.persist = persist

    def run(self) -> AuditReport:
        findings: list[AuditFinding] = []
        ts = datetime.now(timezone.utc).isoformat()

        # Overall performance finding (info)
        acc = accuracy_score(self.y_true, self.y_pred)
        prec = precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        rec = recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)

        findings.append(
            AuditFinding(
                check_id="CLS-PERF-001",
                title="Overall classification performance",
                severity=Severity.INFO,
                description="Overall weighted metrics across all classes.",
                evidence={
                    "accuracy": round(acc, 4),
                    "precision_weighted": round(prec, 4),
                    "recall_weighted": round(rec, 4),
                    "f1_weighted": round(f1, 4),
                    "n_samples": len(self.y_true),
                    "n_classes": int(len(np.unique(self.y_true))),
                },
                recommendation="",
                category="Performance",
                timestamp=ts,
            )
        )

        # Fairness checks
        if self.sensitive_features is not None:
            max_dp = self.thresholds.get("max_demographic_parity_diff", 0.10)
            max_eo = self.thresholds.get("max_equal_opportunity_diff", 0.10)
            max_fnr = self.thresholds.get("max_fnr_diff", 0.15)
            findings.extend(
                fairness_findings_classification(
                    self.y_true,
                    self.y_pred,
                    self.sensitive_features,
                    max_demographic_parity_diff=max_dp,
                    max_equal_opportunity_diff=max_eo,
                    max_fnr_diff=max_fnr,
                    max_equalized_odds_diff=self.thresholds.get(
                        "max_equalized_odds_diff",
                        0.10,
                    ),
                    max_group_calibration_diff=self.thresholds.get(
                        "max_group_calibration_diff",
                        0.10,
                    ),
                    max_group_ci_width=self.thresholds.get("max_group_ci_width", 0.25),
                    positive_label=self.positive_label,
                    include_intersections=self.include_intersections,
                    y_prob=self.y_prob,
                    confidence_level=self.thresholds.get("fairness_confidence_level", 0.95),
                    min_group_size=self.thresholds.get("min_group_size", 5),
                )
            )

        # Data quality checks
        if self.data is not None:
            findings.extend(data_quality_findings(self.data, y_true=self.y_true))

        # Robustness checks
        if self.y_prob is not None:
            findings.extend(
                robustness_findings_classification(
                    self.y_true,
                    self.y_pred,
                    self.y_prob,
                    max_calibration_error=self.thresholds.get("max_calibration_error", 0.10),
                )
            )

        # Explainability checks
        if self.feature_importances is not None:
            findings.extend(
                explainability_findings(
                    self.feature_importances,
                    top_n=self.thresholds.get("explainability_top_n", 10),
                    max_concentration=self.thresholds.get("max_importance_concentration", 0.8),
                )
            )
        elif self.explainability_model is not None and self.data is not None:
            findings.extend(
                shap_explainability_findings(
                    self.explainability_model,
                    self.data,
                    top_n=self.thresholds.get("explainability_top_n", 10),
                    max_concentration=self.thresholds.get("max_importance_concentration", 0.8),
                )
            )

        risk_matrix = compute_risk_matrix(findings)
        meta = {
            **self.metadata,
            "n_samples": len(self.y_true),
            "n_classes": int(len(np.unique(self.y_true))),
            "sensitive_features": (
                list(self.sensitive_features.columns)
                if self.sensitive_features is not None
                else []
            ),
            "intersectional_fairness": self.include_intersections,
        }

        report = AuditReport(
            project_name=self.project_name,
            audit_type="tabular_classification",
            risk_matrix=risk_matrix,
            findings=findings,
            metadata=meta,
        )

        if self.persist:
            save_run(report.to_dict())

        return report
