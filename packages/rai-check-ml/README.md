# rai-check-ml

Fairness, robustness, data quality, and production drift audits for tabular ML
models.

```python
from rai_audit.ml import ClassificationAudit, DriftAudit, RegressionAudit
```

`DriftAudit` compares a reference window with a current batch. It checks numeric
feature distributions, prediction distributions, sensitive-feature subgroup
composition, and classification error-rate changes per sensitive group. Numeric
drift evidence includes corrected KS p-values, population stability index, and
Jensen-Shannon divergence.

Classification fairness checks include equalized odds, calibration by group when
probabilities are available, Wilson confidence intervals, and explicit warnings
for undersized groups. Data-quality checks include common PII patterns, numeric
outliers, and target-deterministic features.

Use `split_data_quality_findings` to catch entity overlap and exact duplicate rows
across train and test datasets:

```python
from rai_audit.ml import split_data_quality_findings

findings = split_data_quality_findings(train, test, id_columns=["patient_id"])
```

See [`examples/ml_drift_monitoring/batch_monitor.py`](examples/ml_drift_monitoring/batch_monitor.py)
and [`examples/mlops_integrations/`](examples/mlops_integrations/) for monitoring examples.
