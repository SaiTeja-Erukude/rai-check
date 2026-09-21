from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np
from rai_audit.core.engine import BaseAudit
from rai_audit.core.findings import AuditFinding, AuditReport, RemediationEffort, Severity
from rai_audit.core.history import save_run
from rai_audit.core.scoring import compute_risk_matrix

_DETECTION_STANDARDS = ["EU-AI-ACT-ART-15", "NIST-AI-RMF-MEASURE-2.6"]


def object_detection_findings(
    ground_truth: Sequence[Sequence[Mapping[str, Any]]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    iou_thresholds=(0.5, 0.75),
    subgroups=None,
    min_map: float = 0.50,
    max_subgroup_map_diff: float = 0.10,
) -> list[AuditFinding]:
    """Measure detection average precision, class coverage, and subgroup parity."""
    if not ground_truth or len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must contain the same non-zero image count")
    thresholds = tuple(float(value) for value in iou_thresholds)
    classes = sorted({_label(box) for image in ground_truth for box in image})
    metrics = {
        str(threshold): _metrics_at_threshold(ground_truth, predictions, threshold, classes)
        for threshold in thresholds
    }
    map_score = float(np.mean([item["map"] for item in metrics.values()])) if metrics else 0.0
    predicted_classes = {_label(box) for image in predictions for box in image}
    missing_classes = [label for label in classes if label not in predicted_classes]
    findings = [
        AuditFinding(
            check_id="DET-PERF-001",
            title="Object-detection performance"
            if map_score >= min_map
            else "Object-detection mAP below threshold",
            severity=Severity.PASSED if map_score >= min_map else Severity.HIGH,
            description=f"Mean AP across configured IoU thresholds is {map_score:.3f}.",
            evidence={
                "map": round(map_score, 4),
                "metrics_by_iou_threshold": metrics,
                "min_map": min_map,
            },
            recommendation="Review localization errors and improve representative class coverage.",
            category="Performance",
            remediation_effort=RemediationEffort.HIGH,
            standards_refs=_DETECTION_STANDARDS,
        ),
        AuditFinding(
            check_id="DET-COVERAGE-001",
            title="Missing object-detection classes"
            if missing_classes
            else "Detection class coverage",
            severity=Severity.HIGH if missing_classes else Severity.PASSED,
            description=f"{len(missing_classes)} ground-truth class(es) have no predicted objects.",
            evidence={
                "ground_truth_classes": classes,
                "classes_missing_predictions": missing_classes,
            },
            recommendation="Add representative training samples and verify label mappings.",
            category="Data Quality",
            standards_refs=_DETECTION_STANDARDS,
        ),
    ]
    if subgroups is not None:
        values = np.asarray(subgroups).astype(str)
        if values.ndim != 1 or len(values) != len(ground_truth):
            raise ValueError("subgroups must contain one value per image")
        maps = {}
        for group in np.unique(values):
            indexes = np.flatnonzero(values == group)
            gt = [ground_truth[index] for index in indexes]
            pred = [predictions[index] for index in indexes]
            maps[group] = round(_metrics_at_threshold(gt, pred, thresholds[0], classes)["map"], 4)
        difference = max(maps.values()) - min(maps.values()) if maps else 0.0
        findings.append(
            AuditFinding(
                check_id="DET-SUBGROUP-001",
                title="Detection subgroup disparity"
                if difference > max_subgroup_map_diff
                else "Detection subgroup parity",
                severity=Severity.MEDIUM if difference > max_subgroup_map_diff else Severity.PASSED,
                description=f"Maximum subgroup mAP difference is {difference:.3f}.",
                evidence={"map_by_group": maps, "max_map_difference": round(difference, 4)},
                recommendation="Review subgroup representation and localization performance.",
                category="Fairness",
                standards_refs=_DETECTION_STANDARDS,
            )
        )
    return findings


class ObjectDetectionAudit(BaseAudit):
    """Audit recorded object-detection annotations."""

    def __init__(
        self,
        ground_truth,
        predictions,
        *,
        subgroups=None,
        project_name="Object Detection Audit",
        metadata=None,
        thresholds=None,
        persist=True,
    ):
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.subgroups = subgroups
        self.project_name = project_name
        self.metadata = metadata or {}
        self.thresholds = thresholds or {}
        self.persist = persist

    def run(self):
        findings = object_detection_findings(
            self.ground_truth,
            self.predictions,
            subgroups=self.subgroups,
            iou_thresholds=self.thresholds.get("iou_thresholds", (0.5, 0.75)),
            min_map=self.thresholds.get("min_map", 0.50),
            max_subgroup_map_diff=self.thresholds.get("max_subgroup_map_diff", 0.10),
        )
        timestamp = datetime.now(timezone.utc).isoformat()
        for finding in findings:
            finding.timestamp = timestamp
        report = AuditReport(
            self.project_name,
            "object_detection",
            compute_risk_matrix(findings),
            findings,
            {**self.metadata, "n_images": len(self.ground_truth)},
        )
        if self.persist:
            save_run(report.to_dict())
        return report


def intersection_over_union(first, second) -> float:
    """Return IoU for [x1, y1, x2, y2] boxes."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("boxes must contain [x1, y1, x2, y2]")
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (
        max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
        + max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        - intersection
    )
    return intersection / union if union else 0.0


def _metrics_at_threshold(ground_truth, predictions, threshold, classes):
    ap_by_class = {}
    for label in classes:
        tp = fp = fn = 0
        for expected, actual in zip(ground_truth, predictions):
            truth_boxes = [_box(item) for item in expected if _label(item) == label]
            pred_boxes = sorted(
                (item for item in actual if _label(item) == label),
                key=lambda item: float(item.get("score", 1.0)),
                reverse=True,
            )
            matched = set()
            for prediction in pred_boxes:
                overlaps = [
                    intersection_over_union(_box(prediction), truth) if index not in matched else -1
                    for index, truth in enumerate(truth_boxes)
                ]
                best = int(np.argmax(overlaps)) if overlaps else -1
                if best >= 0 and overlaps[best] >= threshold:
                    matched.add(best)
                    tp += 1
                else:
                    fp += 1
            fn += len(truth_boxes) - len(matched)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        ap_by_class[str(label)] = round(precision * recall, 4)
    return {
        "map": round(float(np.mean(list(ap_by_class.values()))) if ap_by_class else 0.0, 4),
        "ap_by_class": ap_by_class,
    }


def _label(item):
    if "label" not in item:
        raise ValueError("detection annotations require a label")
    return str(item["label"])


def _box(item):
    if "box" not in item:
        raise ValueError("detection annotations require a box")
    return item["box"]
