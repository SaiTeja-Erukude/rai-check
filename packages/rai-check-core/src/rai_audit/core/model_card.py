"""Model card export for rai-check-core."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rai_audit.core.findings import AuditReport, RiskLevel, Severity

_RISK_BADGE = {
    RiskLevel.CRITICAL: "🔴 CRITICAL",
    RiskLevel.HIGH: "🟠 HIGH",
    RiskLevel.MEDIUM: "🟡 MEDIUM",
    RiskLevel.LOW: "🟢 LOW",
}


def render_model_card(
    report: AuditReport,
    *,
    model_name: str = "",
    model_version: str = "",
    author: str = "",
    license_id: str = "MIT",
    language: str = "en",
    tags: list[str] | None = None,
) -> str:
    """
    Render an AuditReport as a Markdown model card (HuggingFace-compatible format).

    Parameters
    ----------
    report:
        The AuditReport produced by any rai-check audit runner.
    model_name:
        Display name for the model. Defaults to report.project_name.
    model_version:
        Semantic version string, e.g. "1.2.0".
    author:
        Author or team name.
    license_id:
        SPDX license identifier (default "MIT").
    language:
        ISO 639-1 language code for the model card (default "en").
    tags:
        List of tags for the model card YAML frontmatter.
    """
    name = model_name or report.project_name
    version = model_version or str(report.metadata.get("model_version", ""))
    author = author or str(report.metadata.get("author", ""))
    resolved_tags = tags or ["responsible-ai", "audited", report.audit_type.replace("_", "-")]
    audit_date = str(
        report.metadata.get("audit_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    )

    lines: list[str] = []

    # ── YAML frontmatter ──────────────────────────────────────────────────────
    lines += [
        "---",
        f"model_name: {name}",
    ]
    if version:
        lines.append(f"model_version: {version}")
    lines += [
        f"license: {license_id}",
        f"language:",
        f"  - {language}",
        "tags:",
    ]
    for t in resolved_tags:
        lines.append(f"  - {t}")
    lines += ["---", ""]

    # ── Title ─────────────────────────────────────────────────────────────────
    lines.append(f"# Model Card: {name}")
    lines.append("")
    meta_row: list[str] = []
    if version:
        meta_row.append(f"**Version:** {version}")
    if author:
        meta_row.append(f"**Author:** {author}")
    meta_row.append(f"**Audit date:** {audit_date}")
    meta_row.append(f"**Audit type:** `{report.audit_type}`")
    lines.append("  \n".join(meta_row))
    lines.append("")

    badge = _RISK_BADGE.get(report.overall_risk_level, report.overall_risk_level.value.upper())
    lines.append(f"> **Overall risk level:** {badge}")
    lines.append("")

    # ── Model Details ─────────────────────────────────────────────────────────
    lines += ["## Model Details", ""]
    skip_keys = {"audit_date", "author", "model_version"}
    detail_lines = [
        f"- **{k.replace('_', ' ').title()}:** {v}"
        for k, v in report.metadata.items()
        if k not in skip_keys and not isinstance(v, (dict, list))
    ]
    if detail_lines:
        lines += detail_lines
    else:
        lines.append("*No additional metadata recorded.*")
    lines.append("")

    # ── Intended Use ──────────────────────────────────────────────────────────
    lines += [
        "## Intended Use",
        "",
        f"This model was audited under the **`{report.audit_type}`** audit type. "
        "Document the intended use cases, in-scope and out-of-scope applications, "
        "and primary users here.",
        "",
    ]

    # ── Evaluation Results ────────────────────────────────────────────────────
    perf = [f for f in report.findings if f.category == "Performance"]
    if perf:
        lines += ["## Evaluation Results", ""]
        for f in perf:
            lines.append(f"**{f.title}**")
            lines.append("")
            scalar = {k: v for k, v in f.evidence.items() if not isinstance(v, (dict, list))}
            if scalar:
                lines += ["| Metric | Value |", "|--------|-------|"]
                for k, v in scalar.items():
                    lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
                lines.append("")

    # ── Fairness Assessment ───────────────────────────────────────────────────
    fairness_active = [
        f for f in report.findings
        if f.category == "Fairness" and f.severity != Severity.PASSED
    ]
    fairness_passed = [
        f for f in report.findings
        if f.category == "Fairness" and f.severity == Severity.PASSED
    ]
    if fairness_active or fairness_passed:
        lines += ["## Fairness Assessment", ""]
        if fairness_active:
            lines += [
                "| Check ID | Severity | Group | Key Evidence |",
                "|----------|----------|-------|--------------|",
            ]
            for f in fairness_active:
                key_ev = _first_scalar_evidence(f.evidence)
                group = f.affected_group or ""
                lines.append(
                    f"| `{f.check_id}` | **{f.severity.value.upper()}** | {group} | {key_ev} |"
                )
            lines.append("")
        if fairness_passed:
            lines.append(f"**Passed fairness checks ({len(fairness_passed)}):**")
            for f in fairness_passed:
                lines.append(f"- ✓ `{f.check_id}` — {f.title}")
            lines.append("")

    # ── Risk Summary ──────────────────────────────────────────────────────────
    lines += ["## Risk Summary", ""]
    lines += [
        "| Category | Risk Level | Active Findings | Passed Checks |",
        "|----------|-----------|----------------|---------------|",
    ]
    for cat in report.risk_matrix:
        badge_cat = _RISK_BADGE.get(cat.risk_level, cat.risk_level.value.upper())
        lines.append(
            f"| {cat.category} | {badge_cat} | {cat.finding_count} | {cat.passed_count} |"
        )
    lines.append("")

    # ── Standards Compliance ──────────────────────────────────────────────────
    all_refs: set[str] = set()
    for f in report.findings:
        all_refs.update(f.standards_refs)
    if all_refs:
        lines += ["## Standards Compliance", ""]
        for ref in sorted(all_refs):
            lines.append(f"- `{ref}`")
        lines.append("")

    # ── Audit Methodology ─────────────────────────────────────────────────────
    active = [f for f in report.findings if f.severity != Severity.PASSED]
    passed = [f for f in report.findings if f.severity == Severity.PASSED]
    lines += ["## Audit Methodology", ""]
    lines += [
        f"- **Tool:** rai-check-kit",
        f"- **Audit type:** `{report.audit_type}`",
        f"- **Total checks run:** {len(report.findings)}",
        f"- **Active findings:** {len(active)}",
        f"- **Passed checks:** {len(passed)}",
    ]
    if report.overall_score is not None:
        lines.append(f"- **Heuristic score:** {report.overall_score:.1f}/100")
    lines.append("")
    lines.append(
        "> *Auto-generated by `rai-check export model-card`. "
        "Review and supplement before publishing.*"
    )
    lines.append("")

    return "\n".join(lines)


def _first_scalar_evidence(evidence: dict[str, Any]) -> str:
    for k, v in evidence.items():
        if isinstance(v, (int, float)):
            return f"{k}: {v}"
    return ""
