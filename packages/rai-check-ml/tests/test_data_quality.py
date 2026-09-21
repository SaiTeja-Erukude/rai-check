import numpy as np
import pandas as pd
from rai_audit.core.findings import Severity
from rai_audit.ml.data_quality import (
    data_quality_findings,
    split_data_quality_findings,
)


def test_missing_values_flagged():
    df = pd.DataFrame({"a": [1, None, None, None, None, None, 1, 1, 1, 1]})
    findings = data_quality_findings(df)
    f = next(f for f in findings if f.check_id == "DQ-001")
    assert f.severity == Severity.MEDIUM


def test_no_missing_passes():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    findings = data_quality_findings(df)
    f = next(f for f in findings if f.check_id == "DQ-001")
    assert f.severity == Severity.PASSED


def test_duplicate_rows_flagged():
    df = pd.DataFrame({"a": [1] * 10, "b": [2] * 10})
    dup = pd.concat([df, df], ignore_index=True)
    findings = data_quality_findings(dup)
    f = next(f for f in findings if f.check_id == "DQ-002")
    assert f.severity in (Severity.LOW, Severity.MEDIUM)


def test_class_imbalance_flagged():
    y = np.array([0] * 95 + [1] * 5)
    df = pd.DataFrame({"feat": np.random.randn(100)})
    findings = data_quality_findings(df, y_true=y)
    f = next(f for f in findings if f.check_id == "DQ-004")
    assert f.severity in (Severity.LOW, Severity.MEDIUM)


def test_leakage_warning():
    n = 100
    y = np.array([0] * 50 + [1] * 50)
    df = pd.DataFrame({
        "leaked": y.astype(float) + np.random.normal(0, 0.001, n),
        "normal": np.random.randn(n),
    })
    findings = data_quality_findings(df, y_true=y)
    leakage = next((f for f in findings if f.check_id == "DQ-005"), None)
    assert leakage is not None
    assert leakage.severity == Severity.HIGH


def test_categorical_target_leakage_is_detected():
    y = np.array([0] * 50 + [1] * 50)
    df = pd.DataFrame({"review_status": np.where(y == 1, "approved", "denied")})

    findings = data_quality_findings(df, y_true=y)

    leakage = next(f for f in findings if f.check_id == "DQ-005")
    assert leakage.evidence["target_deterministic_features"] == {"review_status": 1.0}


def test_outliers_are_detected():
    df = pd.DataFrame({"amount": [10.0] * 95 + [1000.0] * 5})

    findings = data_quality_findings(df, max_outlier_pct=0.04)

    outliers = next(f for f in findings if f.check_id == "DQ-006")
    assert outliers.severity == Severity.MEDIUM
    assert outliers.evidence["columns"]["amount"] > 0.04


def test_pii_patterns_are_detected_without_echoing_values():
    df = pd.DataFrame({"contact": ["person@example.com", "safe"]})

    findings = data_quality_findings(df)

    pii = next(f for f in findings if f.check_id == "DQ-007")
    assert pii.severity == Severity.HIGH
    assert pii.evidence["columns"] == {"contact": {"email": 1}}
    assert "person@example.com" not in str(pii.evidence)


def test_split_quality_detects_entity_overlap_and_cross_split_duplicates():
    train = pd.DataFrame({"patient_id": [1, 2], "score": [0.1, 0.2]})
    test = pd.DataFrame({"patient_id": [2, 3], "score": [0.2, 0.3]})

    findings = split_data_quality_findings(train, test, id_columns=["patient_id"])

    overlap = next(f for f in findings if f.check_id == "DQ-008")
    duplicates = next(f for f in findings if f.check_id == "DQ-009")
    assert overlap.severity == Severity.HIGH
    assert overlap.evidence["overlapping_entity_count"] == 1
    assert duplicates.severity == Severity.HIGH
    assert duplicates.evidence["cross_split_duplicate_count"] == 1
