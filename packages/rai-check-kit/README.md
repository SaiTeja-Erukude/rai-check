# RAI Check Kit

**RAI** = **Responsible AI**. A Python package suite for evidence-grade audits of
responsible, secure, and trustworthy AI systems.

Run fairness, data quality, robustness, compliance, image, medical imaging, LLM
safety, RAG security, and agent trace checks. Export HTML, Markdown, or JSON
reports and gate CI pipelines on risk thresholds.

**Author:** Sai Teja Erukude | **License:** MIT

## Why this exists

AI teams often run fairness, robustness, RAG, and agent security checks separately.
RAI Check Kit brings them into one evidence and reporting workflow, so teams can
review findings consistently, preserve audit artifacts, and apply the same CI gates
across model types.

## What it looks like

<table>
  <tr>
    <td><strong>HTML audit report</strong><br>
      <a href="https://raw.githubusercontent.com/SaiTeja-Erukude/rai-check/main/docs/images/html-report.png">
        <img src="https://raw.githubusercontent.com/SaiTeja-Erukude/rai-check/main/docs/images/html-report.png" alt="HTML fairness audit report" width="560">
      </a>
    </td>
    <td><strong>Model card export</strong><br>
      <a href="https://raw.githubusercontent.com/SaiTeja-Erukude/rai-check/main/docs/images/model-card.png">
        <img src="https://raw.githubusercontent.com/SaiTeja-Erukude/rai-check/main/docs/images/model-card.png" alt="Markdown model card preview" width="560">
      </a>
    </td>
  </tr>
  <tr>
    <td><strong>LLM and RAG audit output</strong><br>
      <a href="https://raw.githubusercontent.com/SaiTeja-Erukude/rai-check/main/docs/images/rag-check.png">
        <img src="https://raw.githubusercontent.com/SaiTeja-Erukude/rai-check/main/docs/images/rag-check.png" alt="RAG security audit output" width="560">
      </a>
    </td>
    <td><strong>Agent trace finding</strong><br>
      <a href="https://raw.githubusercontent.com/SaiTeja-Erukude/rai-check/main/docs/images/agent-trace-finding.png">
        <img src="https://raw.githubusercontent.com/SaiTeja-Erukude/rai-check/main/docs/images/agent-trace-finding.png" alt="Agent trace prompt injection finding" width="560">
      </a>
    </td>
  </tr>
</table>

## Packages

| Package | Purpose |
|---------|---------|
| `rai-check-core` | Audit engine, findings, reports, history, CI gates |
| `rai-check-ml` | Tabular ML - fairness, drift, data quality, robustness |
| `rai-check-dl` | Image, medical imaging, and scientific AI audits |
| `rai-check-genai` | LLM, RAG, and agentic AI safety, security, and trace audits |
| `rai-check-kit` | Meta-package - installs core + ml, unified CLI |

## Install

```bash
pip install rai-check-kit          # core + tabular ML
pip install "rai-check-kit[all]"   # all modules (dl + genai)
```

## Quick start

```bash
rai-check ml run --help
```

For repeatable audit workflows, generate and run a YAML configuration:

```bash
rai-check init --project loan-model
rai-check run --config audit.yaml
```

`rai-check init` writes a starter `predictions.csv` beside `audit.yaml`; replace
it with captured model predictions, or update `audit.data` to point at your CSV.
Classification configs expect `y_true` and `y_pred` columns by default.

Configured runs write report artifacts and an evidence manifest with input,
environment, source-revision, and artifact hashes.

```python
from rai_audit.ml import ClassificationAudit

report = ClassificationAudit(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_df,
).run()

report.to_html("audit_report.html")
```

## Examples

- [Fairness audit walkthrough](https://github.com/SaiTeja-Erukude/rai-check/blob/main/packages/rai-check-ml/examples/ml_fairness_audit/example.py)
- [Batch drift monitoring](https://github.com/SaiTeja-Erukude/rai-check/blob/main/packages/rai-check-ml/examples/ml_drift_monitoring/batch_monitor.py)
- [MLflow and Airflow templates](https://github.com/SaiTeja-Erukude/rai-check/tree/main/packages/rai-check-ml/examples/mlops_integrations/)
- [Captured-response LLM and RAG audit suite](https://github.com/SaiTeja-Erukude/rai-check/blob/main/packages/rai-check-genai/examples/llm_audit_suite.yml)
- [Scientific image robustness audit](https://github.com/SaiTeja-Erukude/rai-check/blob/main/packages/rai-check-dl/examples/scientific_ai/microscopy_audit.py)
- [Agent trace with a webpage prompt-injection attempt](https://github.com/SaiTeja-Erukude/rai-check/blob/main/packages/rai-check-genai/examples/customer_support_trace.json)

## Development

```bash
pip install uv
uv sync
uv run pytest
```

See [CONTRIBUTING.md](https://github.com/SaiTeja-Erukude/rai-check/blob/main/CONTRIBUTING.md)
for monorepo layout and release workflow.
