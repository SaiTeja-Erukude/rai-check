import json

from rai_audit.core.findings import AuditFinding, AuditReport, Severity
from rai_audit.core.scoring import compute_risk_matrix
from rai_audit.core.standards import (
    NON_COMPLIANCE_CLAIM,
    build_standards_coverage_report,
    render_standards_coverage_markdown,
)
from typer.testing import CliRunner


def _report() -> AuditReport:
    finding = AuditFinding(
        check_id="DQ-001",
        title="Data issue",
        severity=Severity.HIGH,
        description="A data issue was detected.",
        evidence={"column": "age"},
        recommendation="Review the data.",
        standards_refs=["EU-AI-ACT-ART-10"],
    )
    return AuditReport(
        project_name="standards-demo",
        audit_type="classification",
        risk_matrix=compute_risk_matrix([finding]),
        findings=[finding],
        metadata={},
    )


def test_standards_coverage_includes_mapped_and_missing_evidence():
    coverage = build_standards_coverage_report(
        _report(),
        required_refs=["EU-AI-ACT-ART-10", "EU-AI-ACT-ART-15"],
    )

    by_ref = {item["reference"]: item for item in coverage["coverage"]}
    assert coverage["disclaimer"] == NON_COMPLIANCE_CLAIM
    assert by_ref["EU-AI-ACT-ART-10"]["mapped_evidence"][0]["check_id"] == "DQ-001"
    assert by_ref["EU-AI-ACT-ART-15"]["status"] == "missing_evidence"


def test_standards_coverage_markdown_has_non_compliance_language():
    coverage = build_standards_coverage_report(
        _report(),
        required_refs=["EU-AI-ACT-ART-10", "EU-AI-ACT-ART-15"],
    )

    markdown = render_standards_coverage_markdown(coverage)

    assert NON_COMPLIANCE_CLAIM in markdown
    assert "## Missing Evidence" in markdown
    assert "`DQ-001` [HIGH]" in markdown


def test_report_writes_standards_coverage_json(tmp_path):
    output = tmp_path / "standards.json"

    _report().to_standards_coverage(
        str(output),
        required_refs=["EU-AI-ACT-ART-10", "EU-AI-ACT-ART-15"],
    )

    coverage = json.loads(output.read_text(encoding="utf-8"))
    assert coverage["summary"]["references_with_evidence"] == 1
    assert coverage["summary"]["references_missing_evidence"] == 1


def test_standards_coverage_cli_export(tmp_path):
    from rai_audit.core.cli import app

    input_path = tmp_path / "run.json"
    output_path = tmp_path / "coverage.md"
    _report().to_json(str(input_path))

    result = CliRunner().invoke(
        app,
        [
            "export",
            "standards-coverage",
            str(input_path),
            "--output",
            str(output_path),
            "--required-ref",
            "EU-AI-ACT-ART-10",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
