"""Tests for model card export."""
from rai_audit.core.findings import (
    AuditFinding,
    AuditReport,
    CategoryRisk,
    RemediationEffort,
    RiskLevel,
    Severity,
)
from rai_audit.core.model_card import render_model_card


def _sample_report() -> AuditReport:
    findings = [
        AuditFinding(
            check_id="CLS-PERF-001",
            title="Overall classification performance",
            severity=Severity.INFO,
            description="Weighted metrics.",
            evidence={"accuracy": 0.92, "f1_weighted": 0.91, "n_samples": 1000},
            recommendation="",
            category="Performance",
        ),
        AuditFinding(
            check_id="FAIR-CLS-001",
            title="Demographic parity difference for 'gender'",
            severity=Severity.HIGH,
            description="Selection rate gap is 0.22.",
            evidence={
                "demographic_parity_difference": 0.22,
                "threshold": 0.10,
                "group_metrics": {"M": 0.65, "F": 0.43},
            },
            recommendation="Investigate training data.",
            category="Fairness",
            affected_group="gender",
            standards_refs=["EU-AI-ACT-ART-10", "NIST-AI-RMF-MEASURE-2.5"],
        ),
        AuditFinding(
            check_id="DQ-001",
            title="No missing values",
            severity=Severity.PASSED,
            description="All columns complete.",
            evidence={"missing_pct": 0.0},
            recommendation="",
            category="Data Quality",
        ),
    ]
    risk_matrix = [
        CategoryRisk("Fairness", RiskLevel.HIGH, 1, 0),
        CategoryRisk("Performance", RiskLevel.LOW, 0, 1),
        CategoryRisk("Data Quality", RiskLevel.LOW, 0, 1),
    ]
    return AuditReport(
        project_name="Test Classifier",
        audit_type="tabular_classification",
        risk_matrix=risk_matrix,
        findings=findings,
        metadata={"n_samples": 1000, "author": "Test Team"},
    )


def test_render_model_card_returns_string():
    report = _sample_report()
    card = render_model_card(report)
    assert isinstance(card, str)
    assert len(card) > 100


def test_model_card_contains_frontmatter():
    report = _sample_report()
    card = render_model_card(report)
    assert card.startswith("---")
    assert "license:" in card
    assert "tags:" in card


def test_model_card_contains_title():
    report = _sample_report()
    card = render_model_card(report, model_name="My Classifier")
    assert "# Model Card: My Classifier" in card


def test_model_card_risk_summary_present():
    report = _sample_report()
    card = render_model_card(report)
    assert "## Risk Summary" in card
    assert "HIGH" in card
    assert "Fairness" in card


def test_model_card_fairness_section():
    report = _sample_report()
    card = render_model_card(report)
    assert "## Fairness Assessment" in card
    assert "FAIR-CLS-001" in card


def test_model_card_standards_section():
    report = _sample_report()
    card = render_model_card(report)
    assert "## Standards Compliance" in card
    assert "EU-AI-ACT-ART-10" in card


def test_model_card_custom_license():
    report = _sample_report()
    card = render_model_card(report, license_id="Apache-2.0")
    assert "Apache-2.0" in card


def test_to_model_card_method(tmp_path):
    report = _sample_report()
    out = tmp_path / "card.md"
    report.to_model_card(str(out), model_name="Classifier v1")
    assert out.exists()
    content = out.read_text()
    assert "Classifier v1" in content
