import json

from rai_audit.core.history import save_run
from rai_audit.core.monitoring import (
    build_eu_ai_act_post_market_report,
    write_eu_ai_act_post_market_report,
)
from typer.testing import CliRunner


def _run(risk: str, *, incident: str | None = None) -> dict:
    metadata = {}
    if incident:
        metadata["incident_annotations"] = [incident]
    return {
        "project_name": "medical-model",
        "audit_type": "classification",
        "findings": [
            {
                "check_id": "DRIFT-001",
                "title": "Feature drift",
                "severity": risk,
                "category": "Data Quality",
                "standards_refs": ["EU-AI-ACT-ART-9"],
            }
        ],
        "risk_matrix": [
            {
                "category": "Data Quality",
                "risk_level": risk,
                "finding_count": int(risk != "low"),
                "passed_count": int(risk == "low"),
            }
        ],
        "metadata": metadata,
    }


def test_eu_post_market_report_aggregates_history_regressions_and_incidents(tmp_path):
    save_run(_run("low"), tmp_path)
    save_run(_run("high", incident="Drift alert reviewed by model owner."), tmp_path)

    report = build_eu_ai_act_post_market_report(tmp_path, project_name="medical-model")

    assert report["summary"]["run_count"] == 2
    assert report["summary"]["regression_count"] == 1
    assert report["summary"]["incident_count"] == 1
    art_9 = next(
        item for item in report["eu_ai_act_evidence"] if item["reference"] == "EU-AI-ACT-ART-9"
    )
    assert art_9["status"] == "evidence_recorded"
    assert art_9["mapped_checks"] == ["DRIFT-001"]


def test_eu_post_market_report_writes_markdown_and_json(tmp_path):
    save_run(_run("low"), tmp_path)
    markdown_path = tmp_path / "post-market.md"
    json_path = tmp_path / "post-market.json"

    write_eu_ai_act_post_market_report(markdown_path, tmp_path)
    write_eu_ai_act_post_market_report(json_path, tmp_path)

    assert "EU AI Act Post-Market Monitoring Report" in markdown_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["run_count"] == 1


def test_history_dashboard_and_eu_post_market_cli_exports(tmp_path):
    from rai_audit.core.cli import app

    save_run(_run("low"), tmp_path)
    dashboard_path = tmp_path / "history.html"
    monitoring_path = tmp_path / "post-market.md"

    dashboard = CliRunner().invoke(
        app,
        [
            "export",
            "history-dashboard",
            "--directory",
            str(tmp_path),
            "--output",
            str(dashboard_path),
        ],
    )
    monitoring = CliRunner().invoke(
        app,
        [
            "export",
            "eu-post-market",
            "--directory",
            str(tmp_path),
            "--output",
            str(monitoring_path),
        ],
    )

    assert dashboard.exit_code == 0
    assert monitoring.exit_code == 0
    assert dashboard_path.exists()
    assert monitoring_path.exists()
