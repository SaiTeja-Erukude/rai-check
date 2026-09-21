"""
Example: monitor sequential classification batches for responsible AI drift.

Run:
    pip install rai-check-core rai-check-ml
    python packages/rai-check-ml/examples/ml_drift_monitoring/batch_monitor.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rai_audit.ml import DriftAudit


def make_batch(seed: int, group_b_share: float, group_b_error_rate: float, n: int = 500):
    rng = np.random.default_rng(seed)
    group = np.where(rng.random(n) < group_b_share, "B", "A")
    y_true = rng.integers(0, 2, n)
    y_pred = y_true.copy()
    errors = rng.random(n) < np.where(group == "B", group_b_error_rate, 0.05)
    y_pred[errors] = 1 - y_pred[errors]
    features = pd.DataFrame(
        {
            "income": rng.normal(55_000 + 2_000 * group_b_share, 12_000, n),
            "credit_score": rng.normal(680, 45, n),
        }
    )
    return features, pd.DataFrame({"segment": group}), y_true, y_pred


output_dir = Path("drift-reports")
output_dir.mkdir(exist_ok=True)

reference, reference_sensitive, y_true_ref, y_pred_ref = make_batch(
    seed=1,
    group_b_share=0.30,
    group_b_error_rate=0.08,
)

for batch_name, group_b_share, group_b_error_rate in [
    ("2026-05-01", 0.32, 0.09),
    ("2026-05-08", 0.35, 0.12),
    ("2026-05-15", 0.62, 0.35),
]:
    current, current_sensitive, y_true_cur, y_pred_cur = make_batch(
        seed=int(batch_name[-2:]),
        group_b_share=group_b_share,
        group_b_error_rate=group_b_error_rate,
    )
    report = DriftAudit(
        reference=reference,
        current=current,
        reference_sensitive_features=reference_sensitive,
        current_sensitive_features=current_sensitive,
        y_true_ref=y_true_ref,
        y_pred_ref=y_pred_ref,
        y_true_cur=y_true_cur,
        y_pred_cur=y_pred_cur,
        project_name="Loan Approval Model - Weekly Drift",
        metadata={"batch": batch_name},
        persist=False,
    ).run()
    report.to_json(str(output_dir / f"{batch_name}.json"))
    report.to_html(str(output_dir / f"{batch_name}.html"))
    print(batch_name, report.overall_risk_level.value.upper())
