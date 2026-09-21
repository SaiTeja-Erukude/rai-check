# rai-check-dl

Deep learning audits for image classification, medical imaging, and scientific AI.

## Features

- Image classification accuracy and per-class recall checks
- Robustness checks for recorded or callback-generated transformation predictions
- Built-in brightness, contrast, horizontal-flip, and Gaussian-noise transformations
- Grad-CAM heatmaps with PyTorch hooks or TensorFlow `GradientTape`
- Medical imaging checks for patient leakage across splits and site-level accuracy bias
- Segmentation Dice, IoU, mask-quality, and subgroup checks
- Object-detection mAP, IoU-threshold, class-coverage, and subgroup checks
- DICOM metadata, scanner/protocol bias, near-duplicate leakage, calibration, and patient aggregation
- Grad-CAM transformation stability and optional localization-overlap checks

## Python API

```python
from rai_audit.dl import ImageClassificationAudit

report = ImageClassificationAudit(
    y_true=y_true,
    y_pred=y_pred,
    transformed_predictions={"sensor_noise": noisy_predictions},
    persist=False,
).run()
```

To evaluate transformations directly, pass `images` and a `predictor` callback.

## CLI

Audit recorded predictions from CSV:

```bash
rai-check dl run --data predictions.csv --task image --format html
```

Medical imaging CSV audits can add `--patient-id`, `--split`, and `--site`.

## Examples

- [`examples/scientific_ai/microscopy_audit.py`](examples/scientific_ai/microscopy_audit.py)
- [`examples/medical_imaging/audit_example.py`](examples/medical_imaging/audit_example.py)
