"""
Example: Loan approval classification fairness audit.

Run:
    pip install rai-check-core rai-check-ml
    python packages/rai-check-ml/examples/ml_fairness_audit/example.py
"""

import numpy as np
import pandas as pd
from rai_audit.ml import ClassificationAudit

rng = np.random.default_rng(42)
n = 600

# Simulate a biased loan approval model
# Group B applicants experience ~30% higher false negative rate
group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
y_true = rng.integers(0, 2, size=n)
y_pred = y_true.copy()
b_pos = np.where((group == "B") & (y_true == 1))[0]
flip = rng.choice(b_pos, size=int(len(b_pos) * 0.35), replace=False)
y_pred[flip] = 0  # introduce false negatives for group B
y_prob = np.clip(y_pred.astype(float) + rng.normal(0, 0.15, n), 0.05, 0.95)

features_df = pd.DataFrame(
    {
        "income": rng.normal(50000, 15000, n),
        "credit_score": rng.normal(680, 50, n),
        "loan_amount": rng.normal(20000, 8000, n),
    }
)

sensitive = pd.DataFrame({"gender": group})

audit = ClassificationAudit(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    sensitive_features=sensitive,
    data=features_df,
    project_name="Loan Approval Model — Bias Audit",
    thresholds={"max_demographic_parity_diff": 0.10, "max_equal_opportunity_diff": 0.10},
)

report = audit.run()

print(f"\nProject: {report.project_name}")
print(f"Overall risk: {report.overall_risk_level.value.upper()}")
print("\nRisk Matrix:")
for cat in report.risk_matrix:
    print(f"  {cat.category:20s}  {cat.risk_level.value.upper():8s}  "
          f"{cat.finding_count} finding(s), {cat.passed_count} passed")

active = [f for f in report.findings if f.severity.value not in ("passed", "info")]
print(f"\nActive findings ({len(active)}):")
for f in active:
    print(f"  [{f.severity.value.upper():8s}] {f.check_id}: {f.title}")

report.to_html("loan_audit_report.html")
report.to_markdown("loan_audit_report.md")
report.to_json("loan_audit_report.json")
print("\nReports written: loan_audit_report.html, .md, .json")
