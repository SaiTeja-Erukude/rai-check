from rai_audit.core.history import (
    build_history_summary,
    diff_runs,
    load_run,
    render_diff_text,
    save_run,
    write_history_dashboard,
)


def _run(project: str, findings: list[dict], risk_matrix: list[dict]) -> dict:
    return {
        "project_name": project,
        "audit_type": "test",
        "findings": findings,
        "risk_matrix": risk_matrix,
        "metadata": {},
    }


def test_save_and_load(tmp_path):
    run = _run("proj", [], [])
    path = save_run(run, directory=tmp_path)
    assert path.exists()
    loaded = load_run(path)
    assert loaded["project_name"] == "proj"
    assert loaded["schema_version"] == "1.0"


def test_diff_regression(tmp_path):
    a = _run(
        "proj",
        [],
        [{"category": "Fairness", "risk_level": "low", "finding_count": 0, "passed_count": 1}],
    )
    b = _run(
        "proj",
        [{"check_id": "F-001", "title": "bias", "severity": "high"}],
        [{"category": "Fairness", "risk_level": "high", "finding_count": 1, "passed_count": 0}],
    )
    pa = save_run(a, tmp_path)
    pb = save_run(b, tmp_path)
    result = diff_runs(pa, pb)
    changes = {c["category"]: c for c in result["risk_changes"]}
    assert changes["Fairness"]["direction"] == "REGRESSION"


def test_diff_improvement(tmp_path):
    a = _run(
        "proj",
        [{"check_id": "F-001", "title": "bias", "severity": "critical"}],
        [
            {
                "category": "Fairness",
                "risk_level": "critical",
                "finding_count": 1,
                "passed_count": 0,
            }
        ],
    )
    b = _run(
        "proj",
        [],
        [{"category": "Fairness", "risk_level": "low", "finding_count": 0, "passed_count": 1}],
    )
    pa = save_run(a, tmp_path)
    pb = save_run(b, tmp_path)
    result = diff_runs(pa, pb)
    changes = {c["category"]: c for c in result["risk_changes"]}
    assert changes["Fairness"]["direction"] == "IMPROVED"


def test_render_diff_text(tmp_path):
    a = _run(
        "proj",
        [],
        [{"category": "Fairness", "risk_level": "low", "finding_count": 0, "passed_count": 1}],
    )
    b = _run(
        "proj",
        [],
        [{"category": "Fairness", "risk_level": "high", "finding_count": 1, "passed_count": 0}],
    )
    pa = save_run(a, tmp_path)
    pb = save_run(b, tmp_path)
    result = diff_runs(pa, pb)
    text = render_diff_text(result)
    assert "REGRESSION" in text
    assert "Fairness" in text


def test_history_dashboard_contains_trends_regressions_and_artifact_links(tmp_path):
    a = _run(
        "proj",
        [],
        [{"category": "Fairness", "risk_level": "low", "finding_count": 0, "passed_count": 1}],
    )
    b = _run(
        "proj",
        [{"check_id": "F-001", "title": "bias", "severity": "high"}],
        [{"category": "Fairness", "risk_level": "high", "finding_count": 1, "passed_count": 0}],
    )
    save_run(a, tmp_path)
    save_run(b, tmp_path)

    summary = build_history_summary(tmp_path)
    output = tmp_path / "history.html"
    write_history_dashboard(output, tmp_path)
    html = output.read_text(encoding="utf-8")

    assert summary["summary"]["run_count"] == 2
    assert summary["summary"]["regression_count"] == 1
    assert "Risk Trends" in html
    assert "Fairness" in html
    assert "audit-json" in html
