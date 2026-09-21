from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Any

from rai_audit.core.findings import AuditReport, Severity

_SARIF_LEVEL = {
    Severity.CRITICAL: "error",
    Severity.HIGH: "error",
    Severity.MEDIUM: "warning",
    Severity.LOW: "note",
}


def render_sarif(report: AuditReport) -> str:
    """Render active findings as SARIF 2.1.0 for CI and code-scanning systems."""
    active = [
        finding
        for finding in report.findings
        if finding.severity not in {Severity.PASSED, Severity.INFO}
    ]
    rules = {
        finding.check_id: {
            "id": finding.check_id,
            "name": finding.title,
            "shortDescription": {"text": finding.description},
            "properties": {
                "category": finding.category,
                "standards_refs": finding.standards_refs,
            },
        }
        for finding in active
    }
    results = [
        {
            "ruleId": finding.check_id,
            "level": _SARIF_LEVEL.get(finding.severity, "note"),
            "message": {"text": finding.description},
            "properties": {
                "category": finding.category,
                "severity": finding.severity.value,
                "evidence": finding.evidence,
                "recommendation": finding.recommendation,
                "standards_refs": finding.standards_refs,
            },
        }
        for finding in active
    ]
    document: dict[str, Any] = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "rai-check",
                        "informationUri": "https://pypi.org/project/rai-check-kit/",
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(document, indent=2)


def render_junit(report: AuditReport) -> str:
    """Render findings as JUnit XML for CI test-report systems."""
    active = [
        finding
        for finding in report.findings
        if finding.severity not in {Severity.PASSED, Severity.INFO}
    ]
    suite = ET.Element(
        "testsuite",
        {
            "name": report.project_name,
            "tests": str(len(report.findings)),
            "failures": str(len(active)),
        },
    )
    for finding in report.findings:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": finding.category or "General",
                "name": finding.check_id,
            },
        )
        if finding.severity not in {Severity.PASSED, Severity.INFO}:
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "message": finding.title,
                    "type": finding.severity.value,
                },
            )
            failure.text = finding.description
        output = ET.SubElement(case, "system-out")
        output.text = json.dumps(
            {
                "evidence": finding.evidence,
                "recommendation": finding.recommendation,
                "standards_refs": finding.standards_refs,
            },
            default=str,
        )
    ET.indent(suite)
    return ET.tostring(suite, encoding="unicode")
