from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rai_audit.core.findings import AuditFinding, AuditReport

NON_COMPLIANCE_CLAIM = (
    "This report summarizes audit evidence mapped to selected standards references. "
    "It is not a certification, legal opinion, or claim of compliance."
)

STANDARDS_REGISTRY: dict[str, str] = {
    # EU AI Act
    "EU-AI-ACT-ART-9": "EU AI Act Article 9 — Risk management system",
    "EU-AI-ACT-ART-10": "EU AI Act Article 10 — Data and data governance",
    "EU-AI-ACT-ART-13": "EU AI Act Article 13 — Transparency and provision of information",
    "EU-AI-ACT-ART-14": "EU AI Act Article 14 — Human oversight",
    "EU-AI-ACT-ART-15": "EU AI Act Article 15 — Accuracy, robustness and cybersecurity",
    # NIST AI RMF
    "NIST-AI-RMF-GOVERN-1": "NIST AI RMF GOVERN 1 — Policies and processes for AI risk",
    "NIST-AI-RMF-MAP-1": "NIST AI RMF MAP 1 — Context is established for AI risk assessment",
    "NIST-AI-RMF-MEASURE-2.5": "NIST AI RMF MEASURE 2.5 — AI system fairness and bias",
    "NIST-AI-RMF-MEASURE-2.6": "NIST AI RMF MEASURE 2.6 — AI system robustness",
    "NIST-AI-RMF-MEASURE-2.7": "NIST AI RMF MEASURE 2.7 — AI system security",
    "NIST-AI-RMF-MANAGE-1": "NIST AI RMF MANAGE 1 — AI risk treatment",
    # ISO/IEC
    "ISO-42001-6.1": "ISO/IEC 42001 Clause 6.1 — AI risk assessment",
    "ISO-42001-8.4": "ISO/IEC 42001 Clause 8.4 — AI system impact assessment",
    "ISO-23894-6": "ISO/IEC 23894 Clause 6 — AI risk management process",
    # OWASP
    "OWASP-LLM-01": "OWASP LLM Top 10 2025 #1 — Prompt Injection",
    "OWASP-LLM-02": "OWASP LLM Top 10 2025 #2 — Sensitive Information Disclosure",
    "OWASP-LLM-03": "OWASP LLM Top 10 2025 #3 — Supply Chain",
    "OWASP-LLM-04": "OWASP LLM Top 10 2025 #4 — Data and Model Poisoning",
    "OWASP-LLM-05": "OWASP LLM Top 10 2025 #5 — Improper Output Handling",
    "OWASP-LLM-06": "OWASP LLM Top 10 2025 #6 — Excessive Agency",
    "OWASP-LLM-07": "OWASP LLM Top 10 2025 #7 — System Prompt Leakage",
    "OWASP-LLM-08": "OWASP LLM Top 10 2025 #8 — Vector and Embedding Weaknesses",
    "OWASP-LLM-09": "OWASP LLM Top 10 2025 #9 — Misinformation",
    "OWASP-LLM-10": "OWASP LLM Top 10 2025 #10 — Unbounded Consumption",
    "OWASP-ML-01": "OWASP ML Security Top 10 #1 — Input Manipulation Attack",
    "OWASP-ML-05": "OWASP ML Security Top 10 #5 — Model Inversion Attack",
    # OWASP Agentic Applications 2026
    "OWASP-ASI-01": "OWASP Agentic Top 10 2026 #1 - Agent Goal Hijack",
    "OWASP-ASI-02": "OWASP Agentic Top 10 2026 #2 - Tool Misuse and Exploitation",
    "OWASP-ASI-03": "OWASP Agentic Top 10 2026 #3 - Identity and Privilege Abuse",
    "OWASP-ASI-04": "OWASP Agentic Top 10 2026 #4 - Agentic Supply Chain Vulnerabilities",
    "OWASP-ASI-05": "OWASP Agentic Top 10 2026 #5 - Unexpected Code Execution",
    "OWASP-ASI-06": "OWASP Agentic Top 10 2026 #6 - Memory and Context Poisoning",
    "OWASP-ASI-07": "OWASP Agentic Top 10 2026 #7 - Insecure Inter-Agent Communication",
    "OWASP-ASI-08": "OWASP Agentic Top 10 2026 #8 - Cascading Failures",
    "OWASP-ASI-09": "OWASP Agentic Top 10 2026 #9 - Human-Agent Trust Exploitation",
    "OWASP-ASI-10": "OWASP Agentic Top 10 2026 #10 - Rogue Agents",
}


def describe_ref(ref: str) -> str:
    return STANDARDS_REGISTRY.get(ref, ref)


def describe_refs(refs: list[str]) -> list[str]:
    return [describe_ref(r) for r in refs]


def build_standards_crosswalk(findings: list[AuditFinding]) -> dict[str, dict[str, Any]]:
    """Summarize evidence mapped to standards without making a compliance claim."""
    crosswalk: dict[str, dict[str, Any]] = {}
    for finding in findings:
        for ref in finding.standards_refs:
            item = crosswalk.setdefault(
                ref,
                {
                    "description": describe_ref(ref),
                    "active_findings": [],
                    "passed_checks": [],
                },
            )
            target = (
                item["passed_checks"]
                if finding.severity.value == "passed"
                else item["active_findings"]
            )
            target.append(finding.check_id)
    for item in crosswalk.values():
        item["active_findings"] = sorted(set(item["active_findings"]))
        item["passed_checks"] = sorted(set(item["passed_checks"]))
        item["status"] = "findings_present" if item["active_findings"] else "evidence_recorded"
    return dict(sorted(crosswalk.items()))


def build_standards_coverage_report(
    report: AuditReport,
    required_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Build a dedicated evidence-coverage report without making a compliance claim."""
    required = sorted(set(required_refs or STANDARDS_REGISTRY))
    findings_by_ref: dict[str, list[AuditFinding]] = {}
    for finding in report.findings:
        for ref in finding.standards_refs:
            findings_by_ref.setdefault(ref, []).append(finding)

    coverage = []
    for ref in required:
        mapped_findings = findings_by_ref.get(ref, [])
        mapped_evidence = [
            {
                "check_id": finding.check_id,
                "title": finding.title,
                "severity": finding.severity.value,
                "evidence": finding.evidence,
            }
            for finding in mapped_findings
        ]
        active_findings = [
            finding.check_id
            for finding in mapped_findings
            if finding.severity.value != "passed"
        ]
        passed_checks = [
            finding.check_id
            for finding in mapped_findings
            if finding.severity.value == "passed"
        ]
        coverage.append(
            {
                "reference": ref,
                "description": describe_ref(ref),
                "status": (
                    "missing_evidence"
                    if not mapped_findings
                    else "findings_present"
                    if active_findings
                    else "evidence_recorded"
                ),
                "mapped_evidence": mapped_evidence,
                "active_findings": sorted(set(active_findings)),
                "passed_checks": sorted(set(passed_checks)),
            }
        )

    refs_with_evidence = sum(item["status"] != "missing_evidence" for item in coverage)
    return {
        "report_type": "standards_evidence_coverage",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_name": report.project_name,
        "audit_type": report.audit_type,
        "disclaimer": NON_COMPLIANCE_CLAIM,
        "summary": {
            "references_reviewed": len(coverage),
            "references_with_evidence": refs_with_evidence,
            "references_missing_evidence": len(coverage) - refs_with_evidence,
        },
        "coverage": coverage,
    }


def render_standards_coverage_markdown(coverage_report: dict[str, Any]) -> str:
    """Render a standards evidence-coverage report as Markdown."""
    summary = coverage_report["summary"]
    lines = [
        f"# Standards Evidence Coverage: {coverage_report['project_name']}",
        "",
        f"> {coverage_report['disclaimer']}",
        "",
        "## Summary",
        "",
        f"- References reviewed: {summary['references_reviewed']}",
        f"- References with mapped evidence: {summary['references_with_evidence']}",
        f"- References missing evidence: {summary['references_missing_evidence']}",
        "",
        "## Coverage",
        "",
        "| Reference | Status | Active Findings | Passed Checks |",
        "|-----------|--------|-----------------|---------------|",
    ]
    for item in coverage_report["coverage"]:
        lines.append(
            f"| `{item['reference']}` | {item['status']} "
            f"| {', '.join(item['active_findings']) or '-'} "
            f"| {', '.join(item['passed_checks']) or '-'} |"
        )

    missing = [
        item for item in coverage_report["coverage"] if item["status"] == "missing_evidence"
    ]
    if missing:
        lines.extend(["", "## Missing Evidence", ""])
        lines.extend(f"- `{item['reference']}`: {item['description']}" for item in missing)

    mapped = [
        item for item in coverage_report["coverage"] if item["status"] != "missing_evidence"
    ]
    if mapped:
        lines.extend(["", "## Mapped Evidence", ""])
        for item in mapped:
            lines.append(f"### `{item['reference']}`")
            lines.append("")
            for evidence in item["mapped_evidence"]:
                lines.append(
                    f"- `{evidence['check_id']}` [{evidence['severity'].upper()}]: "
                    f"{evidence['title']}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_standards_coverage_report(
    report: AuditReport,
    path: str | Path,
    *,
    required_refs: list[str] | None = None,
) -> None:
    """Write a JSON or Markdown standards evidence-coverage report."""
    output = Path(path)
    coverage = build_standards_coverage_report(report, required_refs)
    if output.suffix.lower() in {".md", ".markdown"}:
        output.write_text(render_standards_coverage_markdown(coverage), encoding="utf-8")
    else:
        output.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
