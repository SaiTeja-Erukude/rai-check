from __future__ import annotations

from collections import defaultdict

from rai_audit.core.findings import AuditFinding, CategoryRisk, RiskLevel, Severity

_SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.INFO: 0,
    Severity.PASSED: -1,
}

_SEVERITY_TO_RISK: dict[Severity, RiskLevel] = {
    Severity.CRITICAL: RiskLevel.CRITICAL,
    Severity.HIGH: RiskLevel.HIGH,
    Severity.MEDIUM: RiskLevel.MEDIUM,
    Severity.LOW: RiskLevel.LOW,
    Severity.INFO: RiskLevel.LOW,
}


def compute_risk_matrix(findings: list[AuditFinding]) -> list[CategoryRisk]:
    """Derive per-category risk levels from a list of findings."""
    by_category: dict[str, list[AuditFinding]] = defaultdict(list)
    for f in findings:
        by_category[f.category or "General"].append(f)

    result: list[CategoryRisk] = []
    for category in sorted(by_category):
        cat_findings = by_category[category]
        active = [f for f in cat_findings if f.severity != Severity.PASSED]
        passed = [f for f in cat_findings if f.severity == Severity.PASSED]

        if not active:
            risk_level = RiskLevel.LOW
        else:
            worst = max(active, key=lambda f: _SEVERITY_RANK[f.severity])
            risk_level = _SEVERITY_TO_RISK.get(worst.severity, RiskLevel.LOW)

        result.append(
            CategoryRisk(
                category=category,
                risk_level=risk_level,
                finding_count=len(active),
                passed_count=len(passed),
            )
        )
    return result


def gate_check(
    report_dict: dict,
    min_score: float | None = None,
    fail_on_critical: bool = True,
) -> tuple[bool, str]:
    """
    Returns (passed: bool, reason: str).
    Used by the CLI gate command and CI pipelines.
    """
    critical_count = sum(
        1 for f in report_dict.get("findings", []) if f.get("severity") == "critical"
    )
    score = report_dict.get("overall_score")

    if fail_on_critical and critical_count > 0:
        return False, f"{critical_count} critical finding(s)"

    if min_score is not None and score is not None and score < min_score:
        return False, f"score {score:.1f} is below minimum {min_score}"

    return True, "all gate conditions passed"
