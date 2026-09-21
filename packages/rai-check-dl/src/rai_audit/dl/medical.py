from __future__ import annotations

import numpy as np
from rai_audit.core.findings import AuditFinding, RemediationEffort, Severity
from rai_audit.dl.image import ImageClassificationAudit, _validated_labels

_MEDICAL_STANDARDS = ["EU-AI-ACT-ART-10", "NIST-AI-RMF-MEASURE-2.5"]
_REQUIRED_DICOM_FIELDS = ("PatientID", "StudyInstanceUID", "Modality")


def patient_leakage_finding(patient_ids, splits) -> AuditFinding:
    """Detect patients represented in more than one dataset split."""
    patients = np.asarray(patient_ids).astype(str)
    split_names = np.asarray(splits).astype(str)
    if patients.ndim != 1 or split_names.ndim != 1 or len(patients) != len(split_names):
        raise ValueError("patient_ids and splits must be one-dimensional arrays of equal length")
    patient_splits: dict[str, set[str]] = {}
    for patient, split in zip(patients, split_names):
        patient_splits.setdefault(patient, set()).add(split)
    leaked = {
        patient: sorted(values) for patient, values in patient_splits.items() if len(values) > 1
    }
    return AuditFinding(
        check_id="MED-LEAK-001",
        title="Patient leakage across dataset splits" if leaked else "No patient split leakage",
        severity=Severity.CRITICAL if leaked else Severity.PASSED,
        description=(
            f"{len(leaked)} patient(s) appear in multiple dataset splits."
            if leaked
            else "Each patient appears in only one dataset split."
        ),
        evidence={"leaked_patients": leaked, "leaked_patient_count": len(leaked)},
        recommendation=(
            "Split medical imaging datasets at the patient level before training or evaluation."
            if leaked
            else ""
        ),
        category="Data Leakage",
        remediation_effort=RemediationEffort.HIGH,
        standards_refs=_MEDICAL_STANDARDS,
    )


def site_bias_finding(
    y_true,
    y_pred,
    sites,
    max_site_accuracy_diff: float = 0.10,
    min_site_samples: int = 5,
) -> AuditFinding:
    """Compare classification accuracy across medical imaging collection sites."""
    truth, predictions = _validated_labels(y_true, y_pred)
    site_values = np.asarray(sites).astype(str)
    if site_values.ndim != 1 or len(site_values) != len(truth):
        raise ValueError("sites must contain one value per prediction")
    accuracy_by_site = {}
    samples_by_site = {}
    for site in np.unique(site_values):
        mask = site_values == site
        samples_by_site[site] = int(mask.sum())
        if mask.sum() >= min_site_samples:
            accuracy_by_site[site] = round(float(np.mean(truth[mask] == predictions[mask])), 4)
    accuracy_diff = (
        max(accuracy_by_site.values()) - min(accuracy_by_site.values())
        if len(accuracy_by_site) >= 2
        else 0.0
    )
    if accuracy_diff > max_site_accuracy_diff * 2:
        severity = Severity.HIGH
    elif accuracy_diff > max_site_accuracy_diff:
        severity = Severity.MEDIUM
    else:
        severity = Severity.PASSED
    return AuditFinding(
        check_id="MED-SITE-001",
        title=(
            "Medical imaging site bias" if severity != Severity.PASSED else "Site accuracy parity"
        ),
        severity=severity,
        description=(
            f"Maximum accuracy difference across eligible sites is {accuracy_diff:.3f} "
            f"(threshold: {max_site_accuracy_diff})."
        ),
        evidence={
            "accuracy_by_site": accuracy_by_site,
            "samples_by_site": samples_by_site,
            "max_accuracy_difference": round(accuracy_diff, 4),
            "threshold": max_site_accuracy_diff,
            "min_site_samples": min_site_samples,
        },
        recommendation=(
            "Review acquisition protocols and evaluate domain adaptation for underperforming sites."
            if severity != Severity.PASSED
            else ""
        ),
        category="Fairness",
        remediation_effort=RemediationEffort.HIGH,
        standards_refs=_MEDICAL_STANDARDS,
    )


def dicom_metadata_finding(records, required_fields=_REQUIRED_DICOM_FIELDS) -> AuditFinding:
    """Screen extracted DICOM metadata without requiring pydicom."""
    missing = {}
    for index, record in enumerate(records):
        absent = [field for field in required_fields if not record.get(field)]
        if absent:
            missing[str(index)] = absent
    return AuditFinding(
        check_id="MED-DICOM-001",
        title="Incomplete DICOM metadata" if missing else "DICOM metadata completeness",
        severity=Severity.HIGH if missing else Severity.PASSED,
        description=f"{len(missing)} DICOM record(s) lack required metadata.",
        evidence={"required_fields": list(required_fields), "missing_fields_by_record": missing},
        recommendation=(
            "Validate required DICOM fields during ingestion and quarantine incomplete studies."
        ),
        category="Data Quality",
        standards_refs=_MEDICAL_STANDARDS,
    )


def near_duplicate_leakage_finding(image_hashes, splits) -> AuditFinding:
    """Detect identical or precomputed near-duplicate image hashes across splits."""
    hashes = np.asarray(image_hashes).astype(str)
    split_names = np.asarray(splits).astype(str)
    if hashes.ndim != 1 or split_names.ndim != 1 or len(hashes) != len(split_names):
        raise ValueError("image_hashes and splits must be one-dimensional arrays of equal length")
    hash_splits = {}
    for value, split in zip(hashes, split_names):
        hash_splits.setdefault(value, set()).add(split)
    leaked = {value: sorted(names) for value, names in hash_splits.items() if len(names) > 1}
    return AuditFinding(
        check_id="MED-LEAK-002",
        title="Near-duplicate imaging leakage" if leaked else "Near-duplicate leakage",
        severity=Severity.CRITICAL if leaked else Severity.PASSED,
        description=f"{len(leaked)} image hash(es) occur across dataset splits.",
        evidence={"cross_split_hashes": leaked},
        recommendation="Deduplicate studies before patient-level dataset splitting.",
        category="Data Leakage",
        remediation_effort=RemediationEffort.HIGH,
        standards_refs=_MEDICAL_STANDARDS,
    )


def calibration_finding(y_true, probabilities, max_ece: float = 0.10, n_bins: int = 10):
    """Calculate expected calibration error for binary or multiclass probabilities."""
    truth = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim == 1:
        predictions = (scores >= 0.5).astype(int)
        confidence = np.maximum(scores, 1 - scores)
    elif scores.ndim == 2 and len(scores) == len(truth):
        predictions = scores.argmax(axis=1)
        confidence = scores.max(axis=1)
    else:
        raise ValueError("probabilities must be a one- or two-dimensional array")
    if len(scores) != len(truth):
        raise ValueError("probabilities must contain one value per prediction")
    ece = 0.0
    for lower in np.linspace(0, 1, n_bins, endpoint=False):
        selected = (confidence >= lower) & (confidence < lower + 1 / n_bins)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float((predictions[selected] == truth[selected]).mean())
                - float(confidence[selected].mean())
            )
    return AuditFinding(
        check_id="MED-CAL-001",
        title="Medical prediction calibration error"
        if ece > max_ece
        else "Medical prediction calibration",
        severity=Severity.MEDIUM if ece > max_ece else Severity.PASSED,
        description=f"Expected calibration error is {ece:.3f}.",
        evidence={
            "expected_calibration_error": round(ece, 4),
            "max_ece": max_ece,
            "n_bins": n_bins,
        },
        recommendation="Calibrate predicted probabilities on a representative validation cohort.",
        category="Performance",
        standards_refs=_MEDICAL_STANDARDS,
    )


def patient_level_performance_finding(y_true, y_pred, patient_ids) -> AuditFinding:
    """Aggregate slice-level classification correctness at the patient level."""
    truth, predictions = _validated_labels(y_true, y_pred)
    patients = np.asarray(patient_ids).astype(str)
    if patients.ndim != 1 or len(patients) != len(truth):
        raise ValueError("patient_ids must contain one value per prediction")
    accuracy_by_patient = {}
    for patient in np.unique(patients):
        selected = patients == patient
        accuracy_by_patient[patient] = round(
            float(np.mean(truth[selected] == predictions[selected])),
            4,
        )
    return AuditFinding(
        check_id="MED-PATIENT-001",
        title="Patient-level performance aggregation",
        severity=Severity.INFO,
        description="Slice-level correctness was aggregated for each patient.",
        evidence={
            "accuracy_by_patient": accuracy_by_patient,
            "mean_patient_accuracy": round(float(np.mean(list(accuracy_by_patient.values()))), 4),
        },
        recommendation="Review patient-level metrics alongside slice-level model performance.",
        category="Performance",
        standards_refs=_MEDICAL_STANDARDS,
    )


class MedicalImagingAudit(ImageClassificationAudit):
    """Image classification audit with patient leakage and site-bias checks."""

    audit_type = "medical_image_classification"

    def __init__(
        self,
        *args,
        patient_ids=None,
        splits=None,
        sites=None,
        dicom_metadata=None,
        scanner_ids=None,
        protocols=None,
        image_hashes=None,
        probabilities=None,
        **kwargs,
    ):
        self.patient_ids = patient_ids
        self.splits = splits
        self.sites = sites
        self.dicom_metadata = dicom_metadata
        self.scanner_ids = scanner_ids
        self.protocols = protocols
        self.image_hashes = image_hashes
        self.probabilities = probabilities
        super().__init__(*args, **kwargs)

    def _additional_findings(self) -> list[AuditFinding]:
        findings = []
        if (self.patient_ids is None) != (self.splits is None):
            raise ValueError("patient_ids and splits must be provided together")
        if self.patient_ids is not None:
            if len(self.patient_ids) != len(self.y_true):
                raise ValueError("patient_ids and splits must contain one value per prediction")
            findings.append(patient_leakage_finding(self.patient_ids, self.splits))
            findings.append(
                patient_level_performance_finding(self.y_true, self.y_pred, self.patient_ids)
            )
        if self.sites is not None:
            findings.append(
                site_bias_finding(
                    self.y_true,
                    self.y_pred,
                    self.sites,
                    max_site_accuracy_diff=self.thresholds.get(
                        "max_site_accuracy_diff",
                        0.10,
                    ),
                    min_site_samples=self.thresholds.get("min_site_samples", 5),
                )
            )
        for values, label in ((self.scanner_ids, "scanner"), (self.protocols, "protocol")):
            if values is not None:
                finding = site_bias_finding(
                    self.y_true,
                    self.y_pred,
                    values,
                    max_site_accuracy_diff=self.thresholds.get("max_site_accuracy_diff", 0.10),
                    min_site_samples=self.thresholds.get("min_site_samples", 5),
                )
                finding.check_id = f"MED-{label.upper()}-001"
                finding.title = finding.title.replace("site", label).replace("Site", label.title())
                findings.append(finding)
        if self.dicom_metadata is not None:
            findings.append(dicom_metadata_finding(self.dicom_metadata))
        if self.image_hashes is not None:
            if self.splits is None:
                raise ValueError("splits must be provided with image_hashes")
            findings.append(near_duplicate_leakage_finding(self.image_hashes, self.splits))
        if self.probabilities is not None:
            findings.append(
                calibration_finding(
                    self.y_true, self.probabilities, self.thresholds.get("max_ece", 0.10)
                )
            )
        return findings

    def _additional_metadata(self) -> dict:
        return {
            "patient_leakage_checked": self.patient_ids is not None,
            "site_bias_checked": self.sites is not None,
            "dicom_metadata_checked": self.dicom_metadata is not None,
            "scanner_bias_checked": self.scanner_ids is not None,
            "protocol_bias_checked": self.protocols is not None,
            "near_duplicate_leakage_checked": self.image_hashes is not None,
            "calibration_checked": self.probabilities is not None,
        }
