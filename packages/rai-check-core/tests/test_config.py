import json

import pandas as pd
import pytest
from rai_audit.core.cli import app
from rai_audit.core.config import ConfigValidationError, load_audit_config, run_config
from typer.testing import CliRunner


def _write_config(tmp_path):
    data_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 0, 1],
            "y_pred": [0, 1, 0, 1, 0, 1],
            "group": ["A", "A", "A", "B", "B", "B"],
            "email": [f"user-{index}@example.com" for index in range(6)],
            "score": [0.1, 0.8, 0.2, 0.9, 0.3, 0.7],
        }
    ).to_csv(data_path, index=False)
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        """
project:
  name: Configured Audit
audit:
  type: classification
  data: predictions.csv
  target: y_true
  prediction: y_pred
  sensitive_features: [group]
  output_dir: output
  report_formats: [json, html, sarif, junit, standards-coverage]
  persist: false
  metadata:
    random_seed: 42
    model_hash: model-sha256
checks:
  fairness:
    enabled: true
  data_quality:
    enabled: true
  privacy:
    enabled: true
  reproducibility:
    enabled: true
gate:
  fail_on_critical: true
""".strip(),
        encoding="utf-8",
    )
    return config_path


def test_run_config_writes_reports_and_evidence_manifest(tmp_path):
    result = run_config(_write_config(tmp_path))

    assert result.gate_passed
    assert result.artifacts["json"].exists()
    assert result.artifacts["html"].exists()
    assert result.artifacts["sarif"].exists()
    assert result.artifacts["junit"].exists()
    assert result.artifacts["standards-coverage"].exists()
    assert result.manifest_path.exists()
    assert any(finding.check_id == "PRIV-001" for finding in result.report.findings)
    assert any(finding.check_id == "REPRO-004" for finding in result.report.findings)
    assert "standards_crosswalk" in result.report.metadata

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert manifest["inputs"]["data"]["sha256"]
    assert manifest["artifacts"]["json"]["sha256"]
    assert manifest["artifacts"]["sarif"]["sha256"]
    assert manifest["artifacts"]["standards-coverage"]["sha256"]


def test_run_config_cli_command(tmp_path):
    result = CliRunner().invoke(app, ["run", "--config", str(_write_config(tmp_path))])

    assert result.exit_code == 0
    assert "Audit complete" in result.output
    assert "evidence-manifest.json" in result.output


def test_init_creates_runnable_starter_files(tmp_path):
    config_path = tmp_path / "audit.yaml"
    runner = CliRunner()

    init_result = runner.invoke(
        app,
        ["init", "--project", "loan-model", "--output", str(config_path)],
    )

    assert init_result.exit_code == 0
    assert config_path.exists()
    assert (tmp_path / "predictions.csv").exists()

    run_result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert run_result.exit_code == 0
    assert "Audit complete" in run_result.output


def test_load_config_migrates_legacy_document(tmp_path):
    config_path = _write_config(tmp_path)

    config = load_audit_config(config_path)

    assert config["schema_version"] == "1.0"


def test_load_config_rejects_unknown_schema_version(tmp_path):
    config_path = tmp_path / "audit.yaml"
    config_path.write_text('schema_version: "99.0"\naudit: {}\n', encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="unsupported"):
        load_audit_config(config_path)


def test_configured_drift_reports_schema_changes(tmp_path):
    pd.DataFrame({"score": [1, 2, 3], "required": [1, 2, 3]}).to_csv(
        tmp_path / "reference.csv",
        index=False,
    )
    pd.DataFrame({"score": [1, 2, 3], "added": [1, 2, 3]}).to_csv(
        tmp_path / "current.csv",
        index=False,
    )
    config_path = tmp_path / "drift.yaml"
    config_path.write_text(
        """
project:
  name: Drift Audit
audit:
  type: drift
  reference_data: reference.csv
  current_data: current.csv
  output_dir: drift-output
  report_formats: [json]
  persist: false
checks:
  reproducibility:
    enabled: false
gate:
  fail_on_critical: true
""".strip(),
        encoding="utf-8",
    )

    result = run_config(config_path)

    finding = next(f for f in result.report.findings if f.check_id == "DRIFT-006")
    assert finding.evidence["missing_columns"] == ["required"]
    assert finding.evidence["added_columns"] == ["added"]
