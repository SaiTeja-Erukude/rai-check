from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rai_audit.core.findings import AuditFinding, RemediationEffort, Severity

if TYPE_CHECKING:
    import pandas as pd

_PII_COLUMN_PATTERNS: list[tuple[str, str]] = [
    (r"\bssn\b|social.?security", "SSN"),
    (r"\bemail\b|e.?mail", "email address"),
    (r"\bphone\b|mobile|telephone", "phone number"),
    (r"\bpassword\b|passwd\b|secret\b", "password/secret"),
    (r"\bcredit.?card\b|card.?number\b|cvv\b", "credit card"),
    (r"\bip.?address\b|ipaddr\b", "IP address"),
    (r"\bdob\b|date.?of.?birth|birthdate", "date of birth"),
    (r"\baddress\b|street\b|zipcode\b|postcode\b", "address"),
    (r"\bpassport\b|national.?id\b|driver.?licen", "government ID"),
]


def check_pii_columns(columns: list[str]) -> list[AuditFinding]:
    """Detect likely PII column names in a dataset."""
    findings: list[AuditFinding] = []
    flagged: list[tuple[str, str]] = []

    for col in columns:
        col_lower = col.lower()
        for pattern, label in _PII_COLUMN_PATTERNS:
            if re.search(pattern, col_lower):
                flagged.append((col, label))
                break

    if flagged:
        findings.append(
            AuditFinding(
                check_id="PRIV-001",
                title="Likely PII columns detected in audit input",
                severity=Severity.HIGH,
                description=(
                    f"{len(flagged)} column(s) appear to contain personally identifiable "
                    "information. Review before sharing audit artifacts."
                ),
                evidence={"flagged_columns": {col: label for col, label in flagged}},
                recommendation=(
                    "Remove or pseudonymise PII columns before running audits "
                    "or sharing reports. Use column aliases if group membership is needed."
                ),
                category="Privacy",
                remediation_effort=RemediationEffort.MEDIUM,
                standards_refs=["EU-AI-ACT-ART-10", "OWASP-ML-05"],
            )
        )
    else:
        findings.append(
            AuditFinding(
                check_id="PRIV-001",
                title="No obvious PII column names detected",
                severity=Severity.PASSED,
                description="Column names do not match known PII patterns.",
                evidence={"columns_checked": len(columns)},
                recommendation="",
                category="Privacy",
            )
        )

    return findings


def check_pii_in_dataframe(df: "pd.DataFrame") -> list[AuditFinding]:
    """Run PII column name check on a pandas DataFrame."""
    return check_pii_columns(list(df.columns))
