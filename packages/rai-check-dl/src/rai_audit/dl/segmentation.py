from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from rai_audit.core.engine import BaseAudit
from rai_audit.core.findings import AuditFinding, AuditReport, RemediationEffort, Severity
from rai_audit.core.history import save_run
from rai_audit.core.scoring import compute_risk_matrix

_SEGMENTATION_STANDARDS = ["EU-AI-ACT-ART-15", "NIST-AI-RMF-MEASURE-2.6"]


def segmentation_findings(
    y_true_masks,
    y_pred_masks,
    *,
    class_labels=None,
    subgroups=None,
    min_dice: float = 0.70,
    min_iou: float = 0.55,
    max_subgroup_dice_diff: float = 0.10,
) -> list[AuditFinding]:
    """Measure segmentation overlap, class coverage, subgroup parity, and mask quality."""
    truth, predictions = _validated_masks(y_true_masks, y_pred_masks)
    labels = np.asarray(
        class_labels if class_labels is not None else np.union1d(truth, predictions)
    )
    labels = labels[labels != 0]
    metrics = {}
    missing_predictions = []
    for label in labels:
        truth_mask = truth == label
        prediction_mask = predictions == label
        intersection = int(np.logical_and(truth_mask, prediction_mask).sum())
        truth_pixels = int(truth_mask.sum())
        predicted_pixels = int(prediction_mask.sum())
        union = int(np.logical_or(truth_mask, prediction_mask).sum())
        metrics[str(label)] = {
            "dice": round(_ratio(2 * intersection, truth_pixels + predicted_pixels), 4),
            "iou": round(_ratio(intersection, union), 4),
            "truth_pixels": truth_pixels,
            "predicted_pixels": predicted_pixels,
        }
        if truth_pixels and not predicted_pixels:
            missing_predictions.append(str(label))

    weak_classes = {
        label: values
        for label, values in metrics.items()
        if values["dice"] < min_dice or values["iou"] < min_iou
    }
    invalid_truth = int((~np.isfinite(truth)).sum())
    invalid_predictions = int((~np.isfinite(predictions)).sum())
    empty_truth_masks = int(np.all(truth == 0, axis=tuple(range(1, truth.ndim))).sum())
    empty_prediction_masks = int(
        np.all(predictions == 0, axis=tuple(range(1, predictions.ndim))).sum()
    )
    findings = [
        AuditFinding(
            check_id="SEG-PERF-001",
            title="Segmentation overlap metrics",
            severity=Severity.INFO,
            description="Dice and IoU were measured for each non-background class.",
            evidence={"metrics_by_class": metrics, "n_masks": len(truth)},
            recommendation="",
            category="Performance",
            standards_refs=_SEGMENTATION_STANDARDS,
        ),
        AuditFinding(
            check_id="SEG-PERF-002",
            title="Weak segmentation class performance"
            if weak_classes
            else "Segmentation thresholds",
            severity=Severity.HIGH if weak_classes else Severity.PASSED,
            description=f"{len(weak_classes)} class(es) fall below the Dice or IoU threshold.",
            evidence={
                "weak_classes": weak_classes,
                "min_dice": min_dice,
                "min_iou": min_iou,
            },
            recommendation=(
                "Review mask labels and add representative training samples for weak classes."
            ),
            category="Performance",
            remediation_effort=RemediationEffort.HIGH,
            standards_refs=_SEGMENTATION_STANDARDS,
        ),
        AuditFinding(
            check_id="SEG-MASK-001",
            title="Segmentation mask-quality issues"
            if invalid_truth or invalid_predictions
            else "Mask quality",
            severity=Severity.HIGH if invalid_truth or invalid_predictions else Severity.PASSED,
            description="Segmentation masks were screened for invalid values and empty masks.",
            evidence={
                "invalid_truth_pixels": invalid_truth,
                "invalid_prediction_pixels": invalid_predictions,
                "empty_truth_masks": empty_truth_masks,
                "empty_prediction_masks": empty_prediction_masks,
                "classes_missing_predictions": missing_predictions,
            },
            recommendation=(
                "Validate mask encoding, label coverage, and empty-mask handling before training."
            ),
            category="Data Quality",
            remediation_effort=RemediationEffort.MEDIUM,
            standards_refs=_SEGMENTATION_STANDARDS,
        ),
    ]
    if subgroups is not None:
        findings.append(
            _subgroup_finding(
                truth,
                predictions,
                subgroups,
                max_subgroup_dice_diff=max_subgroup_dice_diff,
            )
        )
    return findings


class SegmentationAudit(BaseAudit):
    """Audit semantic segmentation masks."""

    def __init__(
        self,
        y_true_masks,
        y_pred_masks,
        *,
        class_labels=None,
        subgroups=None,
        project_name: str = "Segmentation Audit",
        metadata: dict | None = None,
        thresholds: dict | None = None,
        persist: bool = True,
    ):
        self.y_true_masks = y_true_masks
        self.y_pred_masks = y_pred_masks
        self.class_labels = class_labels
        self.subgroups = subgroups
        self.project_name = project_name
        self.metadata = metadata or {}
        self.thresholds = thresholds or {}
        self.persist = persist

    def run(self) -> AuditReport:
        findings = segmentation_findings(
            self.y_true_masks,
            self.y_pred_masks,
            class_labels=self.class_labels,
            subgroups=self.subgroups,
            min_dice=self.thresholds.get("min_dice", 0.70),
            min_iou=self.thresholds.get("min_iou", 0.55),
            max_subgroup_dice_diff=self.thresholds.get("max_subgroup_dice_diff", 0.10),
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        for finding in findings:
            finding.timestamp = timestamp
        truth = np.asarray(self.y_true_masks)
        report = AuditReport(
            project_name=self.project_name,
            audit_type="image_segmentation",
            risk_matrix=compute_risk_matrix(findings),
            findings=findings,
            metadata={**self.metadata, "n_masks": len(truth), "mask_shape": list(truth.shape[1:])},
        )
        if self.persist:
            save_run(report.to_dict())
        return report


def _subgroup_finding(truth, predictions, subgroups, *, max_subgroup_dice_diff):
    values = np.asarray(subgroups).astype(str)
    if values.ndim != 1 or len(values) != len(truth):
        raise ValueError("subgroups must contain one value per mask")
    dice_by_group = {}
    for group in np.unique(values):
        selected = values == group
        intersection = int(np.logical_and(truth[selected] > 0, predictions[selected] > 0).sum())
        total = int((truth[selected] > 0).sum() + (predictions[selected] > 0).sum())
        dice_by_group[group] = round(_ratio(2 * intersection, total), 4)
    difference = max(dice_by_group.values()) - min(dice_by_group.values()) if dice_by_group else 0.0
    return AuditFinding(
        check_id="SEG-SUBGROUP-001",
        title="Segmentation subgroup disparity"
        if difference > max_subgroup_dice_diff
        else "Segmentation subgroup parity",
        severity=Severity.MEDIUM if difference > max_subgroup_dice_diff else Severity.PASSED,
        description=f"Maximum subgroup Dice difference is {difference:.3f}.",
        evidence={"dice_by_group": dice_by_group, "max_dice_difference": round(difference, 4)},
        recommendation=(
            "Review subgroup representation and evaluate class-specific subgroup performance."
        ),
        category="Fairness",
        standards_refs=_SEGMENTATION_STANDARDS,
    )


def _validated_masks(y_true_masks, y_pred_masks):
    truth = np.asarray(y_true_masks)
    predictions = np.asarray(y_pred_masks)
    if truth.shape != predictions.shape or truth.ndim < 3 or not len(truth):
        raise ValueError("segmentation masks must be non-empty arrays with matching batch shapes")
    return truth, predictions


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 1.0
