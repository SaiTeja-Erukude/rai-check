from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from rai_audit.core.schemas import SCHEMA_VERSION, prepare_document


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    PASSED = "passed"


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RemediationEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AuditFinding:
    check_id: str
    title: str
    severity: Severity
    description: str
    evidence: dict[str, Any]
    recommendation: str
    category: str = ""
    affected_group: str | None = None
    remediation_effort: RemediationEffort = RemediationEffort.MEDIUM
    standards_refs: list[str] = field(default_factory=list)
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "title": self.title,
            "severity": self.severity.value,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "category": self.category,
            "affected_group": self.affected_group,
            "remediation_effort": self.remediation_effort.value,
            "standards_refs": self.standards_refs,
            "timestamp": self.timestamp,
        }


@dataclass
class CategoryRisk:
    category: str
    risk_level: RiskLevel
    finding_count: int
    passed_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "risk_level": self.risk_level.value,
            "finding_count": self.finding_count,
            "passed_count": self.passed_count,
        }


@dataclass
class AuditReport:
    project_name: str
    audit_type: str
    risk_matrix: list[CategoryRisk]
    findings: list[AuditFinding]
    metadata: dict[str, Any]
    overall_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return prepare_document(
            "report",
            {
                "schema_version": SCHEMA_VERSION,
                "project_name": self.project_name,
                "audit_type": self.audit_type,
                "risk_matrix": [r.to_dict() for r in self.risk_matrix],
                "findings": [f.to_dict() for f in self.findings],
                "metadata": self.metadata,
                "overall_score": self.overall_score,
            },
        )

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_markdown(self, path: str) -> None:
        from rai_audit.core.report import render_markdown

        Path(path).write_text(render_markdown(self), encoding="utf-8")

    def to_html(self, path: str) -> None:
        from rai_audit.core.report import render_html

        Path(path).write_text(render_html(self), encoding="utf-8")

    def to_model_card(self, path: str, **kwargs) -> None:
        from rai_audit.core.model_card import render_model_card

        Path(path).write_text(render_model_card(self, **kwargs), encoding="utf-8")

    def to_sarif(self, path: str) -> None:
        from rai_audit.core.ci_formats import render_sarif

        Path(path).write_text(render_sarif(self), encoding="utf-8")

    def to_junit(self, path: str) -> None:
        from rai_audit.core.ci_formats import render_junit

        Path(path).write_text(render_junit(self), encoding="utf-8")

    def to_standards_coverage(
        self,
        path: str,
        *,
        required_refs: list[str] | None = None,
    ) -> None:
        from rai_audit.core.standards import write_standards_coverage_report

        write_standards_coverage_report(self, path, required_refs=required_refs)

    @property
    def critical_findings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def high_findings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == Severity.HIGH]

    @property
    def passed_findings(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == Severity.PASSED]

    @property
    def overall_risk_level(self) -> RiskLevel:
        for cat in self.risk_matrix:
            if cat.risk_level == RiskLevel.CRITICAL:
                return RiskLevel.CRITICAL
        for cat in self.risk_matrix:
            if cat.risk_level == RiskLevel.HIGH:
                return RiskLevel.HIGH
        for cat in self.risk_matrix:
            if cat.risk_level == RiskLevel.MEDIUM:
                return RiskLevel.MEDIUM
        return RiskLevel.LOW
