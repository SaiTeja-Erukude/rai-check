# Getting Started

## Install

```bash
pip install rai-audit-ml
```

## Run a Classification Audit

```python
import numpy as np
import pandas as pd
from rai_audit.ml import ClassificationAudit

report = ClassificationAudit(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,                                    # optional, for calibration checks
    sensitive_features=pd.DataFrame({"gender": g}),  # optional, for fairness checks
    data=X,                                           # optional, for data quality checks
    project_name="Loan Approval Model",
).run()

# Outputs
report.to_html("audit.html")          # rich HTML report with charts
report.to_json("audit-run.json")      # save for gate / diff / history
report.to_markdown("audit.md")
report.to_model_card("model-card.md") # HuggingFace-compatible
```

## Run a Configured Audit

Generate a starter configuration, point it at captured predictions, and run the
audit workflow:

```bash
rai-audit init --project loan-model
rai-audit run --config audit.yaml
```

`rai-audit init` writes a small starter `predictions.csv` beside `audit.yaml`, so
the generated config can run immediately. Replace that CSV with captured model
predictions, or update `audit.data` to point at your own file. For a classification
audit the CSV needs `y_true` and `y_pred` columns by default.

The configured runner writes the selected report formats and an
`evidence-manifest.json` file. The manifest records input hashes, runtime details,
installed RAI Audit package versions, the Git revision when available, and artifact
hashes. Configured audits also run privacy column-name screening and reproducibility
checks by default. Add `sarif` or `junit` to `audit.report_formats` for CI-native
outputs.

## CI/CD Gate

```bash
rai-audit gate audit-run.json --fail-on-critical
```

Exits `1` on any critical finding, `0` on pass. Use in GitHub Actions:

```yaml
- name: RAI Audit gate
  run: rai-audit gate audit-run.json --fail-on-critical
```

See `packages/rai-audit-core/examples/ci-gate.yml` for a full example workflow.

## Monitor Batch Drift

Use `DriftAudit` to compare a production batch against a reference window. Supplying
labels enables error-rate drift checks for each sensitive group. Numeric,
categorical, and feature-schema changes are checked.

```python
from rai_audit.ml import DriftAudit

report = DriftAudit(
    reference=reference_features,
    current=current_features,
    reference_sensitive_features=reference_sensitive,
    current_sensitive_features=current_sensitive,
    y_true_ref=reference_labels,
    y_pred_ref=reference_predictions,
    y_true_cur=current_labels,
    y_pred_cur=current_predictions,
    project_name="Loan Model - Weekly Drift",
).run()

report.to_json("drift-run.json")
report.to_html("drift-report.html")
```

See `packages/rai-audit-ml/examples/ml_drift_monitoring/batch_monitor.py` for
sequential batch monitoring and `packages/rai-audit-ml/examples/mlops_integrations/`
for MLflow and Airflow templates.

## Audit Image Models

Use `ImageClassificationAudit` with recorded predictions, or pass an image batch and
predictor callback to evaluate built-in transformations.

```python
from rai_audit.dl import ImageClassificationAudit

report = ImageClassificationAudit(
    y_true=y_true,
    y_pred=y_pred,
    transformed_predictions={"sensor_noise": noisy_predictions},
).run()
```

See `packages/rai-audit-dl/examples/scientific_ai/microscopy_audit.py` and
`packages/rai-audit-dl/examples/medical_imaging/audit_example.py`.

## Audit Agent Traces

Capture tool and retrieval operations with the canonical agent trace schema, then run:

```bash
rai-audit agents run \
  --trace packages/rai-audit-agents/examples/customer_support_trace.json \
  --allowed-tools lookup_order
```

The audit checks tool allowlists, failures, memory writes, permissions, approvals, and
prompt injection delivered through tools, retrieval, email, or webpages.

## Export Model Card

```bash
rai-audit export model-card audit-run.json \
  --model-name "Loan Model v2" \
  --model-version "2.0.0" \
  --author "ML Team"
```
