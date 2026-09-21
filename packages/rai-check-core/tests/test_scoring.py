import pytest
from rai_audit.core.findings import AuditFinding, RiskLevel, Severity
from rai_audit.core.scoring import compute_risk_matrix, gate_check


def _f(severity: Severity, category: str = "Fairness") -> AuditFinding:
    return AuditFinding(
        check_id="X",
        title="t",
        severity=severity,
        description="d",
        evidence={},
        recommendation="r",
        category=category,
    )


def test_critical_dominates():
    findings = [_f(Severity.CRITICAL), _f(Severity.PASSED), _f(Severity.HIGH)]
    matrix = compute_risk_matrix(findings)
    assert matrix[0].risk_level == RiskLevel.CRITICAL


def test_all_passed_is_low():
    findings = [_f(Severity.PASSED), _f(Severity.PASSED)]
    matrix = compute_risk_matrix(findings)
    assert matrix[0].risk_level == RiskLevel.LOW


def test_multiple_categories():
    findings = [
        _f(Severity.HIGH, "Fairness"),
        _f(Severity.PASSED, "Fairness"),
        _f(Severity.MEDIUM, "Robustness"),
    ]
    matrix = compute_risk_matrix(findings)
    cats = {m.category: m for m in matrix}
    assert cats["Fairness"].risk_level == RiskLevel.HIGH
    assert cats["Robustness"].risk_level == RiskLevel.MEDIUM


def test_gate_fails_on_critical():
    run = {"findings": [{"severity": "critical"}], "risk_matrix": []}
    passed, reason = gate_check(run, fail_on_critical=True)
    assert not passed
    assert "critical" in reason


def test_gate_passes_no_critical():
    run = {"findings": [{"severity": "high"}], "overall_score": 85, "risk_matrix": []}
    passed, _ = gate_check(run, min_score=80, fail_on_critical=True)
    assert passed


def test_gate_fails_low_score():
    run = {"findings": [], "overall_score": 60, "risk_matrix": []}
    passed, reason = gate_check(run, min_score=80, fail_on_critical=True)
    assert not passed
    assert "60" in reason
