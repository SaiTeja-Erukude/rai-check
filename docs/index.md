# RAI Check Kit

**Evidence-grade audits for responsible, secure, and trustworthy AI systems.**

RAI Check Kit is a Python toolkit that helps developers, researchers, and AI teams audit AI systems for fairness, robustness, security, transparency, and deployment readiness.

## Install

```bash
pip install rai-check-kit          # core + tabular ML
pip install "rai-check-kit[all]"   # all modules
pip install rai-check-ml           # ML audits only
pip install rai-check-genai        # LLM, RAG, and agent audits only
pip install rai-check-core         # engine and reports only
```

## Quick Example

```python
from rai_audit.ml import ClassificationAudit
import pandas as pd

report = ClassificationAudit(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=pd.DataFrame({"gender": gender_col}),
    project_name="My Classifier",
).run()

report.to_html("audit-report.html")
report.to_model_card("model-card.md")
```

## Package Suite

| Package | Description | Status |
|---------|-------------|--------|
| `rai-check-core` | Shared engine, findings, reports, CLI | Available |
| `rai-check-ml` | Tabular ML audits (classification, regression) | Available |
| `rai-check-kit` | Meta-package: installs core + ml | Available |
| `rai-check-dl` | Image, medical imaging, and scientific AI audits | Available |
| `rai-check-genai` | LLM, RAG, and agentic AI audits | Available |