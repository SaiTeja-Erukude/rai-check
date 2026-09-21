"""Airflow DAG template for scheduled drift monitoring."""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="rai_audit_weekly_drift",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
) as dag:
    monitor = BashOperator(
        task_id="monitor_model_drift",
        bash_command="python packages/rai-check-ml/examples/ml_drift_monitoring/batch_monitor.py",
    )
