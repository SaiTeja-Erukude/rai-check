import numpy as np
from rai_audit.core.findings import Severity
from rai_audit.dl import ImageClassificationAudit
from rai_audit.dl.robustness import transformation_robustness_findings


def test_image_audit_reports_performance():
    report = ImageClassificationAudit(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 0],
        persist=False,
    ).run()

    assert report.audit_type == "image_classification"
    assert report.metadata["n_samples"] == 4
    assert next(f for f in report.findings if f.check_id == "IMG-PERF-001").evidence[
        "accuracy"
    ] == 0.75


def test_transformation_robustness_detects_accuracy_drop():
    findings = transformation_robustness_findings(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 0, 1, 1]),
        transformed_predictions={"blur": np.array([1, 1, 0, 0])},
    )

    assert findings[0].severity == Severity.HIGH
    assert findings[0].evidence["accuracy_drop"] == 1.0


def test_image_audit_evaluates_predictor_over_transformations():
    images = np.ones((3, 4, 4, 1), dtype=float)
    report = ImageClassificationAudit(
        y_true=np.ones(3, dtype=int),
        images=images,
        predictor=lambda batch: (batch.mean(axis=(1, 2, 3)) > 0.5).astype(int),
        persist=False,
    ).run()

    assert sorted(report.metadata["transformations"]) == [
        "brightness",
        "contrast",
        "gaussian_noise",
        "horizontal_flip",
    ]
