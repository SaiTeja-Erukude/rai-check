import json

from rai_audit.core.findings import (
    AuditFinding,
    AuditReport,
    RiskLevel,
    Severity,
)


def make_finding(severity: Severity, category: str = "Fairness") -> AuditFinding:
    return AuditFinding(
        check_id="TEST-001",
        title="Test finding",
        severity=severity,
        description="A test finding.",
        evidence={"value": 0.25},
        recommendation="Fix it.",
        category=category,
    )


def make_report(findings: list[AuditFinding]) -> AuditReport:
    from rai_audit.core.scoring import compute_risk_matrix

    return AuditReport(
        project_name="test-project",
        audit_type="test",
        risk_matrix=compute_risk_matrix(findings),
        findings=findings,
        metadata={},
    )


def test_finding_to_dict():
    f = make_finding(Severity.HIGH)
    d = f.to_dict()
    assert d["severity"] == "high"
    assert d["check_id"] == "TEST-001"
    assert d["evidence"] == {"value": 0.25}


def test_finding_standards_refs():
    f = AuditFinding(
        check_id="TEST-002",
        title="Standards test",
        severity=Severity.MEDIUM,
        description="desc",
        evidence={},
        recommendation="rec",
        standards_refs=["EU-AI-ACT-ART-10", "NIST-AI-RMF-MEASURE-2.5"],
    )
    d = f.to_dict()
    assert "EU-AI-ACT-ART-10" in d["standards_refs"]


def test_report_overall_risk_critical():
    findings = [make_finding(Severity.CRITICAL)]
    report = make_report(findings)
    assert report.overall_risk_level == RiskLevel.CRITICAL


def test_report_overall_risk_passed():
    findings = [make_finding(Severity.PASSED)]
    report = make_report(findings)
    assert report.overall_risk_level == RiskLevel.LOW


def test_report_to_dict_roundtrip():
    findings = [make_finding(Severity.HIGH), make_finding(Severity.PASSED)]
    report = make_report(findings)
    d = report.to_dict()
    assert d["schema_version"] == "1.0"
    assert d["project_name"] == "test-project"
    assert len(d["findings"]) == 2


def test_report_to_json(tmp_path):
    findings = [make_finding(Severity.MEDIUM)]
    report = make_report(findings)
    out = tmp_path / "report.json"
    report.to_json(str(out))
    data = json.loads(out.read_text())
    assert data["project_name"] == "test-project"


def test_report_to_markdown(tmp_path):
    findings = [make_finding(Severity.HIGH), make_finding(Severity.PASSED)]
    report = make_report(findings)
    out = tmp_path / "report.md"
    report.to_markdown(str(out))
    content = out.read_text()
    assert "Audit Report" in content
    assert "Risk Summary" in content


def test_report_to_html(tmp_path):
    findings = [make_finding(Severity.CRITICAL)]
    report = make_report(findings)
    out = tmp_path / "report.html"
    report.to_html(str(out))
    content = out.read_text()
    assert "<!DOCTYPE html>" in content
    assert "CRITICAL" in content


def test_report_to_sarif(tmp_path):
    report = make_report([make_finding(Severity.HIGH), make_finding(Severity.PASSED)])
    out = tmp_path / "report.sarif"
    report.to_sarif(str(out))
    data = json.loads(out.read_text())
    assert data["version"] == "2.1.0"
    assert data["runs"][0]["results"][0]["ruleId"] == "TEST-001"


def test_report_to_junit(tmp_path):
    report = make_report([make_finding(Severity.HIGH), make_finding(Severity.PASSED)])
    out = tmp_path / "report.junit.xml"
    report.to_junit(str(out))
    content = out.read_text()
    assert '<testsuite name="test-project"' in content
    assert "<failure" in content
