from __future__ import annotations

from rai_audit.core.findings import AuditFinding

OWASP_AGENTIC_TOP_10_2026 = {
    "OWASP-ASI-01": "Agent Goal Hijack",
    "OWASP-ASI-02": "Tool Misuse and Exploitation",
    "OWASP-ASI-03": "Identity and Privilege Abuse",
    "OWASP-ASI-04": "Agentic Supply Chain Vulnerabilities",
    "OWASP-ASI-05": "Unexpected Code Execution",
    "OWASP-ASI-06": "Memory and Context Poisoning",
    "OWASP-ASI-07": "Insecure Inter-Agent Communication",
    "OWASP-ASI-08": "Cascading Failures",
    "OWASP-ASI-09": "Human-Agent Trust Exploitation",
    "OWASP-ASI-10": "Rogue Agents",
}


def owasp_agentic_coverage(findings: list[AuditFinding]) -> dict[str, dict]:
    """Report mapped evidence and explicit gaps across OWASP Agentic Top 10 2026."""
    coverage = {}
    for reference, title in OWASP_AGENTIC_TOP_10_2026.items():
        mapped = [finding for finding in findings if reference in finding.standards_refs]
        coverage[reference] = {
            "title": title,
            "status": "mapped" if mapped else "missing_evidence",
            "checks": sorted({finding.check_id for finding in mapped}),
        }
    return coverage
