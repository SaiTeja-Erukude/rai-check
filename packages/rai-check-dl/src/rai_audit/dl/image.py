from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

import numpy as np
from rai_audit.core.engine import BaseAudit
from rai_audit.core.findings import AuditFinding, AuditReport, RemediationEffort, Severity
from rai_audit.core.history import save_run
from rai_audit.core.scoring import compute_risk_matrix
from rai_audit.dl.robustness import transformation_robustness_findings
from rai_audit.dl.transformations import (
    ImageTransform,
    Predictor,
    evaluate_transformations,
    prediction_labels,
)

_IMAGE_STANDARDS = ["EU-AI-ACT-ART-15", "NIST-AI-RMF-MEASURE-2.6"]


def image_classification_findings(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_class_recall: float = 0.60,
) -> list[AuditFinding]:
    """Evaluate overall accuracy and per-class recall for image predictions."""
    truth, predictions = _validated_labels(y_true, y_pred)
    accuracy = float(np.mean(truth == predictions))
    recalls = {}
    support = {}
    for label in np.unique(truth):
        mask = truth == label
        support[str(label)] = int(mask.sum())
        recalls[str(label)] = round(float(np.mean(predictions[mask] == label)), 4)

    low_recall = {
        label: recall for label, recall in recalls.items() if recall < min_class_recall
    }
    findings = [
        AuditFinding(
            check_id="IMG-PERF-001",
            title="Overall image classification performance",
            severity=Severity.INFO,
            description="Overall accuracy across the evaluated image dataset.",
            evidence={
                "accuracy": round(accuracy, 4),
                "n_samples": len(truth),
                "n_classes": len(recalls),
            },
            recommendation="",
            category="Performance",
        )
    ]
    findings.append(
        AuditFinding(
            check_id="IMG-PERF-002",
            title=(
                "Low recall for one or more image classes"
                if low_recall
                else "Per-class image recall is within threshold"
            ),
            severity=Severity.MEDIUM if low_recall else Severity.PASSED,
            description=(
                f"{len(low_recall)} class(es) have recall below {min_class_recall}."
                if low_recall
                else f"All image classes have recall of at least {min_class_recall}."
            ),
            evidence={
                "recall_by_class": recalls,
                "support_by_class": support,
                "low_recall_classes": low_recall,
                "threshold": min_class_recall,
            },
            recommendation=(
                "Review mislabeled samples and add representative training images for weak classes."
                if low_recall
                else ""
            ),
            category="Performance",
            remediation_effort=RemediationEffort.MEDIUM,
            standards_refs=_IMAGE_STANDARDS,
        )
    )
    return findings


class ImageClassificationAudit(BaseAudit):
    """Audit image classification predictions and transformation robustness."""

    audit_type = "image_classification"

    def __init__(
        self,
        y_true,
        y_pred=None,
        *,
        images=None,
        predictor: Predictor | None = None,
        transformations: Mapping[str, ImageTransform] | None = None,
        transformed_predictions: Mapping[str, np.ndarray] | None = None,
        project_name: str = "Image Classification Audit",
        metadata: dict | None = None,
        thresholds: dict | None = None,
        persist: bool = True,
    ):
        self.y_true = np.asarray(y_true)
        self.images = np.asarray(images) if images is not None else None
        self.predictor = predictor
        self.transformations = transformations
        self.project_name = project_name
        self.metadata = metadata or {}
        self.thresholds = thresholds or {}
        self.persist = persist
        self.y_pred = self._resolve_predictions(y_pred)
        self.transformed_predictions = self._resolve_transformed_predictions(
            transformed_predictions
        )

    def run(self) -> AuditReport:
        findings = image_classification_findings(
            self.y_true,
            self.y_pred,
            min_class_recall=self.thresholds.get("min_class_recall", 0.60),
        )
        if self.transformed_predictions:
            findings.extend(
                transformation_robustness_findings(
                    self.y_true,
                    self.y_pred,
                    self.transformed_predictions,
                    max_accuracy_drop=self.thresholds.get("max_accuracy_drop", 0.10),
                )
            )
        findings.extend(self._additional_findings())
        timestamp = datetime.now(timezone.utc).isoformat()
        for finding in findings:
            finding.timestamp = finding.timestamp or timestamp

        report = AuditReport(
            project_name=self.project_name,
            audit_type=self.audit_type,
            risk_matrix=compute_risk_matrix(findings),
            findings=findings,
            metadata={
                **self.metadata,
                **self._additional_metadata(),
                "n_samples": len(self.y_true),
                "n_classes": int(len(np.unique(self.y_true))),
                "transformations": sorted(self.transformed_predictions),
            },
        )
        if self.persist:
            save_run(report.to_dict())
        return report

    def _resolve_predictions(self, y_pred) -> np.ndarray:
        if y_pred is None:
            if self.images is None or self.predictor is None:
                raise ValueError("Provide y_pred or both images and predictor")
            y_pred = self.predictor(self.images)
        predictions = prediction_labels(np.asarray(y_pred))
        _validated_labels(self.y_true, predictions)
        return predictions

    def _resolve_transformed_predictions(
        self,
        transformed_predictions: Mapping[str, np.ndarray] | None,
    ) -> dict[str, np.ndarray]:
        if transformed_predictions is not None:
            return {
                name: prediction_labels(values) for name, values in transformed_predictions.items()
            }
        if self.images is not None and self.predictor is not None:
            return evaluate_transformations(self.predictor, self.images, self.transformations)
        return {}

    def _additional_findings(self) -> list[AuditFinding]:
        return []

    def _additional_metadata(self) -> dict:
        return {}


def _validated_labels(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if truth.ndim != 1 or predictions.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional label arrays")
    if len(truth) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must contain the same number of labels")
    return truth, predictions
