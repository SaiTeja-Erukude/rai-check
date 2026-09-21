"""Example: transformation robustness audit for a scientific microscopy classifier."""

import numpy as np
from rai_audit.dl import ScientificAIAudit

rng = np.random.default_rng(42)
n = 120
y_true = rng.integers(0, 3, size=n)
y_pred = y_true.copy()

# Simulate performance under image acquisition changes captured during evaluation.
brightness_predictions = y_true.copy()
brightness_predictions[:8] = (brightness_predictions[:8] + 1) % 3
noise_predictions = y_true.copy()
noise_predictions[:30] = (noise_predictions[:30] + 1) % 3

report = ScientificAIAudit(
    y_true=y_true,
    y_pred=y_pred,
    transformed_predictions={
        "brightness": brightness_predictions,
        "sensor_noise": noise_predictions,
    },
    scientific_domain="cell microscopy",
    project_name="Cell Morphology Classifier",
    persist=False,
).run()

report.to_html("microscopy_audit.html")
print(report.overall_risk_level.value.upper())
