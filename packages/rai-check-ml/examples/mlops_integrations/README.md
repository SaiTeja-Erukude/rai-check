# MLOps Integration Examples

`DriftAudit` returns the standard `AuditReport`, so monitoring jobs can publish JSON
and HTML artifacts, use `.rai-check/history`, or forward summary metrics to an
observability platform.

## MLflow

Call `log_drift_report(report)` from `mlflow_tracking.py` inside an active MLflow run:

```python
import mlflow
from mlflow_tracking import log_drift_report

with mlflow.start_run(run_name="weekly-drift-monitor"):
    report = audit.run()
    log_drift_report(report)
```

## Airflow

`airflow_dag.py` shows a weekly DAG that runs the batch monitoring script. In a
production DAG, replace the example command with the job that loads your reference
window and current batch from your feature store.
