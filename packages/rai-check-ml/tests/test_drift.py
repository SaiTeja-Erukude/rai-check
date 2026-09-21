import numpy as np
import pandas as pd
import pytest
from rai_audit.core.findings import Severity
from rai_audit.ml import DriftAudit
from rai_audit.ml.drift import drift_findings


def _feature_batches(n: int = 200):
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({"score": rng.normal(0, 1, n)})
    current = pd.DataFrame({"score": rng.normal(3, 1, n)})
    return reference, current


def test_numeric_feature_drift_is_detected():
    reference, current = _feature_batches()
    findings = drift_findings(reference, current)
    finding = next(f for f in findings if f.check_id == "DRIFT-001")
    assert finding.severity == Severity.MEDIUM
    assert finding.evidence["drifted_columns"] == ["score"]


def test_prediction_drift_does_not_require_sensitive_features():
    reference, current = _feature_batches()
    findings = drift_findings(
        reference,
        current,
        y_pred_ref=np.zeros(len(reference)),
        y_pred_cur=np.ones(len(current)),
    )
    finding = next(f for f in findings if f.check_id == "DRIFT-002")
    assert finding.severity == Severity.MEDIUM


def test_sensitive_feature_distribution_drift_is_detected():
    reference, current = _feature_batches()
    ref_sensitive = pd.DataFrame({"region": ["A"] * 100 + ["B"] * 100})
    cur_sensitive = pd.DataFrame({"region": ["A"] * 20 + ["B"] * 180})
    findings = drift_findings(
        reference,
        current,
        reference_sensitive_features=ref_sensitive,
        current_sensitive_features=cur_sensitive,
    )
    finding = next(f for f in findings if f.check_id == "DRIFT-003")
    assert finding.severity == Severity.HIGH
    assert finding.evidence["total_variation_distance"] == pytest.approx(0.4)


def test_error_rate_drift_by_group_is_detected():
    n = 200
    reference = pd.DataFrame({"score": np.arange(n)})
    current = reference.copy()
    sensitive = pd.DataFrame({"region": ["A"] * 100 + ["B"] * 100})
    y_true = np.zeros(n, dtype=int)
    y_pred_ref = y_true.copy()
    y_pred_cur = y_true.copy()
    y_pred_cur[100:150] = 1

    findings = drift_findings(
        reference,
        current,
        reference_sensitive_features=sensitive,
        current_sensitive_features=sensitive.copy(),
        y_true_ref=y_true,
        y_pred_ref=y_pred_ref,
        y_true_cur=y_true,
        y_pred_cur=y_pred_cur,
    )
    finding = next(f for f in findings if f.check_id == "DRIFT-004")
    assert finding.severity == Severity.HIGH
    assert finding.evidence["error_rate_differences"]["B"] == pytest.approx(0.5)
    assert finding.evidence["error_rate_differences"]["A"] == pytest.approx(0.0)


def test_drift_audit_produces_monitoring_report():
    reference, current = _feature_batches()
    report = DriftAudit(reference, current, persist=False).run()
    assert report.audit_type == "tabular_drift_monitoring"
    assert report.metadata["reference_samples"] == len(reference)
    assert report.metadata["current_samples"] == len(current)


def test_sensitive_feature_batches_must_be_supplied_together():
    reference, current = _feature_batches()
    sensitive = pd.DataFrame({"region": ["A"] * len(reference)})
    with pytest.raises(ValueError, match="provided together"):
        drift_findings(reference, current, sensitive_features=sensitive)


def test_categorical_feature_drift_is_detected():
    reference = pd.DataFrame({"region": ["A"] * 90 + ["B"] * 10})
    current = pd.DataFrame({"region": ["A"] * 10 + ["B"] * 90})

    findings = drift_findings(reference, current)

    finding = next(f for f in findings if f.check_id == "DRIFT-005")
    assert finding.severity == Severity.MEDIUM
    assert finding.evidence["drifted_columns"]["region"]["total_variation_distance"] == 0.8


def test_feature_schema_drift_is_detected():
    reference = pd.DataFrame({"score": [1, 2, 3], "required": [1, 2, 3]})
    current = pd.DataFrame({"score": [1, 2, 3], "added": [1, 2, 3]})

    findings = drift_findings(reference, current)

    finding = next(f for f in findings if f.check_id == "DRIFT-006")
    assert finding.severity == Severity.HIGH
    assert finding.evidence["missing_columns"] == ["required"]
    assert finding.evidence["added_columns"] == ["added"]


def test_numeric_drift_includes_psi_js_and_multiple_testing_correction():
    reference, current = _feature_batches()

    findings = drift_findings(reference, current)

    finding = next(f for f in findings if f.check_id == "DRIFT-001")
    metrics = finding.evidence["distribution_metrics"]["score"]
    assert metrics["population_stability_index"] > 0
    assert metrics["jensen_shannon_divergence"] > 0
    assert finding.evidence["multiple_testing_correction"] == "benjamini-hochberg"
    assert "score" in finding.evidence["adjusted_ks_pvalues"]


def test_psi_threshold_can_detect_drift_when_ks_threshold_is_disabled():
    reference, current = _feature_batches()

    findings = drift_findings(
        reference,
        current,
        ks_pvalue_threshold=0.0,
        max_population_stability_index=0.01,
        max_jensen_shannon_divergence=1.0,
    )

    finding = next(f for f in findings if f.check_id == "DRIFT-001")
    assert finding.severity == Severity.MEDIUM
    assert finding.evidence["drifted_columns"] == ["score"]
