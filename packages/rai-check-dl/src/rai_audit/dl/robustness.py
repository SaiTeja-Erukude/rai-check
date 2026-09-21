from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from rai_audit.core.findings import AuditFinding, RemediationEffort, Severity

_ROBUSTNESS_STANDARDS = ["EU-AI-ACT-ART-15", "NIST-AI-RMF-MEASURE-2.6", "OWASP-ML-01"]


def transformation_robustness_findings(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transformed_predictions: Mapping[str, np.ndarray],
    max_accuracy_drop: float = 0.10,
) -> list[AuditFinding]:
    """Compare baseline accuracy with predictions under image transformations."""
    truth = np.asarray(y_true)
    baseline = np.asarray(y_pred)
    _validate_labels("y_pred", baseline, len(truth))
    baseline_accuracy = float(np.mean(truth == baseline))
    findings = []

    for name, values in transformed_predictions.items():
        predictions = np.asarray(values)
        _validate_labels(f"transformed_predictions[{name!r}]", predictions, len(truth))
        transformed_accuracy = float(np.mean(truth == predictions))
        accuracy_drop = baseline_accuracy - transformed_accuracy
        if accuracy_drop > max_accuracy_drop * 2:
            severity = Severity.HIGH
        elif accuracy_drop > max_accuracy_drop:
            severity = Severity.MEDIUM
        else:
            severity = Severity.PASSED
        findings.append(
            AuditFinding(
                check_id=f"IMG-ROB-{_slug(name)}",
                title=f"Robustness under {name}",
                severity=severity,
                description=(
                    f"Accuracy changed from {baseline_accuracy:.3f} to "
                    f"{transformed_accuracy:.3f} under '{name}' "
                    f"(drop: {accuracy_drop:.3f}, threshold: {max_accuracy_drop})."
                ),
                evidence={
                    "transformation": name,
                    "baseline_accuracy": round(baseline_accuracy, 4),
                    "transformed_accuracy": round(transformed_accuracy, 4),
                    "accuracy_drop": round(accuracy_drop, 4),
                    "threshold": max_accuracy_drop,
                },
                recommendation=(
                    "Augment training data and evaluate preprocessing consistency for this "
                    "transformation."
                    if severity != Severity.PASSED
                    else ""
                ),
                category="Robustness",
                remediation_effort=RemediationEffort.HIGH,
                standards_refs=_ROBUSTNESS_STANDARDS,
            )
        )
    return findings


def _validate_labels(name: str, values: np.ndarray, expected_length: int) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional label array")
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} labels")


def _slug(value: str) -> str:
    normalized = "".join(char if char.isalnum() else "-" for char in value.upper())
    return "-".join(part for part in normalized.split("-") if part)
