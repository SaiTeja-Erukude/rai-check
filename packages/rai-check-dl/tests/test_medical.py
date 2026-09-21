import numpy as np
import pytest
from rai_audit.core.findings import Severity
from rai_audit.dl import MedicalImagingAudit
from rai_audit.dl.medical import (
    dicom_metadata_finding,
    near_duplicate_leakage_finding,
    patient_leakage_finding,
    patient_level_performance_finding,
    site_bias_finding,
)


def test_patient_leakage_is_critical():
    finding = patient_leakage_finding(
        patient_ids=["p1", "p1", "p2"],
        splits=["train", "test", "test"],
    )

    assert finding.severity == Severity.CRITICAL
    assert finding.evidence["leaked_patients"] == {"p1": ["test", "train"]}


def test_site_bias_detects_accuracy_gap():
    finding = site_bias_finding(
        y_true=np.array([0] * 10 + [1] * 10),
        y_pred=np.array([0] * 10 + [0] * 10),
        sites=np.array(["A"] * 10 + ["B"] * 10),
    )

    assert finding.severity == Severity.HIGH
    assert finding.evidence["max_accuracy_difference"] == 1.0


def test_medical_audit_runs_specialized_checks():
    report = MedicalImagingAudit(
        y_true=[0, 0, 1, 1, 1, 1],
        y_pred=[0, 0, 1, 1, 1, 0],
        patient_ids=["p1", "p2", "p3", "p4", "p5", "p6"],
        splits=["train", "train", "test", "test", "test", "test"],
        sites=["A", "A", "A", "B", "B", "B"],
        persist=False,
    ).run()

    assert report.audit_type == "medical_image_classification"
    assert report.metadata["patient_leakage_checked"] is True
    assert {finding.check_id for finding in report.findings} >= {
        "MED-LEAK-001",
        "MED-SITE-001",
    }


def test_medical_audit_rejects_misaligned_patient_metadata():
    audit = MedicalImagingAudit(
        y_true=[0, 1],
        y_pred=[0, 1],
        patient_ids=["p1"],
        splits=["train"],
        persist=False,
    )

    with pytest.raises(ValueError, match="one value per prediction"):
        audit.run()


def test_medical_metadata_duplicate_and_patient_aggregation_checks():
    assert dicom_metadata_finding([{"PatientID": "p1"}]).severity == Severity.HIGH
    assert (
        near_duplicate_leakage_finding(["same", "same"], ["train", "test"]).severity
        == Severity.CRITICAL
    )
    finding = patient_level_performance_finding([1, 1], [1, 0], ["p1", "p1"])
    assert finding.evidence["accuracy_by_patient"] == {"p1": 0.5}
