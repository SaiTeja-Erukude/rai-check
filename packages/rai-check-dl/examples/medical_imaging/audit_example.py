"""Example: patient leakage and site-bias checks for a medical imaging classifier."""

import numpy as np
from rai_audit.dl import MedicalImagingAudit

y_true = np.array([0] * 20 + [1] * 20)
y_pred = np.array([0] * 20 + [1] * 10 + [0] * 10)
patient_ids = np.array([f"patient-{index}" for index in range(40)])
splits = np.array(["train"] * 20 + ["test"] * 20)
sites = np.array(["hospital-a"] * 20 + ["hospital-b"] * 20)

report = MedicalImagingAudit(
    y_true=y_true,
    y_pred=y_pred,
    patient_ids=patient_ids,
    splits=splits,
    sites=sites,
    project_name="Chest Imaging Classifier",
    persist=False,
).run()

report.to_html("medical_imaging_audit.html")
print(report.overall_risk_level.value.upper())
