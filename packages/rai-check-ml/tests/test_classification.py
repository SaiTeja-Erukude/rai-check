import numpy as np
import pandas as pd
from rai_audit.core.findings import Severity
from rai_audit.ml.classification import ClassificationAudit


def _make_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    noise = rng.integers(0, 2, size=n)
    y_pred = np.where(rng.random(n) > 0.1, y_true, noise)
    y_prob = np.clip(rng.random(n), 0.05, 0.95)
    group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    sens = pd.DataFrame({"group": group})
    df = pd.DataFrame({"feat1": rng.normal(size=n), "feat2": rng.normal(size=n)})
    return y_true, y_pred, y_prob, sens, df


def test_classification_audit_basic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    y_true, y_pred, y_prob, sens, df = _make_data()
    audit = ClassificationAudit(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features=sens,
        data=df,
    )
    report = audit.run()
    assert report.audit_type == "tabular_classification"
    assert len(report.findings) > 0
    assert len(report.risk_matrix) > 0


def test_classification_audit_to_html(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    y_true, y_pred, y_prob, sens, df = _make_data()
    audit = ClassificationAudit(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features=sens,
    )
    report = audit.run()
    out = tmp_path / "report.html"
    report.to_html(str(out))
    content = out.read_text()
    assert "<!DOCTYPE html>" in content


def test_classification_audit_to_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import json
    y_true, y_pred, _, _, _ = _make_data()
    audit = ClassificationAudit(y_true=y_true, y_pred=y_pred)
    report = audit.run()
    out = tmp_path / "report.json"
    report.to_json(str(out))
    data = json.loads(out.read_text())
    assert "findings" in data
    assert "risk_matrix" in data


def test_classification_audit_persist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    y_true, y_pred, _, _, _ = _make_data()
    audit = ClassificationAudit(y_true=y_true, y_pred=y_pred, persist=True)
    audit.run()
    history_dir = tmp_path / ".rai-check" / "history"
    assert history_dir.exists()
    assert len(list(history_dir.glob("*.json"))) == 1


def test_classification_audit_no_sensitive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    y_true, y_pred, _, _, _ = _make_data()
    audit = ClassificationAudit(y_true=y_true, y_pred=y_pred)
    report = audit.run()
    assert report is not None
    perf = next(f for f in report.findings if f.check_id == "CLS-PERF-001")
    assert perf.severity == Severity.INFO


def test_multiclass_audit_uses_multiclass_calibration_and_fairness():
    y_true = np.tile([0, 1, 2], 20)
    y_pred = y_true.copy()
    y_prob = np.eye(3)[y_pred]
    sensitive = pd.DataFrame({"group": ["A"] * 30 + ["B"] * 30})

    report = ClassificationAudit(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        sensitive_features=sensitive,
        persist=False,
    ).run()

    calibration = next(f for f in report.findings if f.check_id == "ROB-CLS-002")
    assert calibration.evidence["calibration_mode"] == "multiclass"
    assert any(f.check_id == "FAIR-CLS-001-0" for f in report.findings)
    assert any(f.check_id == "FAIR-CLS-001-2" for f in report.findings)


def test_classification_audit_includes_feature_importance_findings():
    report = ClassificationAudit(
        y_true=[0, 1, 0, 1],
        y_pred=[0, 1, 0, 1],
        feature_importances={"income": 0.95, "age": 0.05},
        persist=False,
    ).run()

    finding = next(f for f in report.findings if f.check_id == "EXPL-001")
    assert finding.severity == Severity.MEDIUM
