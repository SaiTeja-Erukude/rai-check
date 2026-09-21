import numpy as np
from rai_audit.core.findings import Severity
from rai_audit.dl import ObjectDetectionAudit, SegmentationAudit, intersection_over_union


def test_segmentation_audit_reports_overlap_and_subgroups():
    truth = np.array([[[0, 1], [0, 1]], [[0, 1], [0, 1]]])
    report = SegmentationAudit(
        truth,
        truth.copy(),
        subgroups=["site-a", "site-b"],
        persist=False,
    ).run()

    assert report.audit_type == "image_segmentation"
    performance = next(f for f in report.findings if f.check_id == "SEG-PERF-001")
    assert performance.evidence["metrics_by_class"]["1"]["dice"] == 1.0


def test_object_detection_audit_reports_map_and_class_coverage():
    ground_truth = [[{"label": "lesion", "box": [0, 0, 10, 10]}]]
    predictions = [[{"label": "lesion", "box": [0, 0, 10, 10], "score": 0.9}]]

    report = ObjectDetectionAudit(ground_truth, predictions, persist=False).run()

    assert intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert report.findings[0].severity == Severity.PASSED
    assert report.findings[0].evidence["map"] == 1.0
