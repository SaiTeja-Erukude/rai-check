from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rai_audit.core.findings import AuditFinding, RemediationEffort, Severity

_PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
}


def data_quality_findings(
    df: pd.DataFrame,
    y_true: np.ndarray | None = None,
    max_missing_pct: float = 0.05,
    max_constant_pct: float = 0.99,
    min_class_balance: float = 0.10,
    max_outlier_pct: float = 0.05,
    outlier_iqr_multiplier: float = 1.5,
    max_leakage_predictability: float = 0.98,
    scan_pii: bool = True,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    n_rows, n_cols = df.shape

    # Missing values
    missing = df.isnull().mean()
    high_missing = missing[missing > max_missing_pct]
    if not high_missing.empty:
        findings.append(
            AuditFinding(
                check_id="DQ-001",
                title="High missing value rate in features",
                severity=Severity.MEDIUM,
                description=(
                    f"{len(high_missing)} column(s) have more than "
                    f"{max_missing_pct*100:.0f}% missing values."
                ),
                evidence={
                    "columns": {col: round(pct, 4) for col, pct in high_missing.items()},
                    "threshold": max_missing_pct,
                },
                recommendation=(
                    "Investigate missing value causes. Impute or drop columns as appropriate."
                ),
                category="Data Quality",
                remediation_effort=RemediationEffort.MEDIUM,
                standards_refs=["EU-AI-ACT-ART-10"],
            )
        )
    else:
        findings.append(
            AuditFinding(
                check_id="DQ-001",
                title="Missing values within acceptable range",
                severity=Severity.PASSED,
                description=(
                    f"All columns have less than {max_missing_pct*100:.0f}% missing values."
                ),
                evidence={"max_missing_pct": round(float(missing.max()), 4)},
                recommendation="",
                category="Data Quality",
            )
        )

    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    dup_pct = dup_count / n_rows if n_rows > 0 else 0
    if dup_pct > 0.01:
        findings.append(
            AuditFinding(
                check_id="DQ-002",
                title="Duplicate rows detected",
                severity=Severity.MEDIUM if dup_pct > 0.05 else Severity.LOW,
                description=f"{dup_count} duplicate rows found ({dup_pct*100:.1f}% of dataset).",
                evidence={"duplicate_count": dup_count, "duplicate_pct": round(dup_pct, 4)},
                recommendation="Review and remove duplicates before training or evaluation.",
                category="Data Quality",
                remediation_effort=RemediationEffort.LOW,
            )
        )
    else:
        findings.append(
            AuditFinding(
                check_id="DQ-002",
                title="No significant duplicate rows",
                severity=Severity.PASSED,
                description=f"{dup_count} duplicate rows ({dup_pct*100:.2f}%).",
                evidence={"duplicate_count": dup_count},
                recommendation="",
                category="Data Quality",
            )
        )

    # Constant / near-constant columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    near_constant = []
    for col in numeric_cols:
        value_counts = df[col].value_counts(normalize=True)
        if value_counts.empty:
            continue
        top_pct = value_counts.iloc[0]
        if top_pct > max_constant_pct:
            near_constant.append((col, round(top_pct, 4)))

    if near_constant:
        findings.append(
            AuditFinding(
                check_id="DQ-003",
                title="Near-constant columns detected",
                severity=Severity.LOW,
                description=(
                    f"{len(near_constant)} column(s) have a single value in more than "
                    f"{max_constant_pct*100:.0f}% of rows. These are unlikely to be predictive."
                ),
                evidence={"columns": {col: pct for col, pct in near_constant}},
                recommendation=(
                    "Consider removing near-constant columns unless they carry semantic meaning."
                ),
                category="Data Quality",
                remediation_effort=RemediationEffort.LOW,
            )
        )
    else:
        findings.append(
            AuditFinding(
                check_id="DQ-003",
                title="No near-constant columns detected",
                severity=Severity.PASSED,
                description="All numeric columns have adequate variance.",
                evidence={},
                recommendation="",
                category="Data Quality",
            )
        )

    # Class imbalance (classification only)
    if y_true is not None:
        classes, counts = np.unique(y_true, return_counts=True)
        class_pcts = counts / counts.sum()
        min_pct = float(class_pcts.min())

        if min_pct < min_class_balance:
            minority_class = classes[class_pcts.argmin()]
            findings.append(
                AuditFinding(
                    check_id="DQ-004",
                    title="Class imbalance detected",
                    severity=Severity.MEDIUM if min_pct < 0.05 else Severity.LOW,
                    description=(
                        f"Minority class '{minority_class}' represents only "
                        f"{min_pct*100:.1f}% of samples (threshold: {min_class_balance*100:.0f}%)."
                    ),
                    evidence={
                        "class_distribution": {
                            str(c): round(p, 4) for c, p in zip(classes, class_pcts)
                        },
                        "minority_class": str(minority_class),
                        "minority_pct": round(min_pct, 4),
                    },
                    recommendation=(
                        "Imbalanced classes can lead to biased metrics. "
                        "Consider resampling, class weights, or stratified evaluation."
                    ),
                    category="Data Quality",
                    remediation_effort=RemediationEffort.MEDIUM,
                    standards_refs=["EU-AI-ACT-ART-10"],
                )
            )
        else:
            findings.append(
                AuditFinding(
                    check_id="DQ-004",
                    title="Class balance within acceptable range",
                    severity=Severity.PASSED,
                    description=f"Minority class is {min_pct*100:.1f}% of samples.",
                    evidence={"minority_pct": round(min_pct, 4)},
                    recommendation="",
                    category="Data Quality",
                )
            )

    # Feature correlation (potential leakage warning)
    if y_true is not None:
        high_corr_cols = []
        for col in numeric_cols:
            try:
                corr = abs(float(np.corrcoef(df[col].fillna(0).values, y_true)[0, 1]))
                if corr > 0.95:
                    high_corr_cols.append((col, round(corr, 4)))
            except Exception:
                pass

        deterministic_features = _target_deterministic_features(
            df,
            np.asarray(y_true),
            max_leakage_predictability,
        )
        if high_corr_cols or deterministic_features:
            findings.append(
                AuditFinding(
                    check_id="DQ-005",
                    title="Potential target leakage detected",
                    severity=Severity.HIGH,
                    description=(
                        f"{len(high_corr_cols)} feature(s) have suspiciously high correlation "
                        f"with the target and {len(deterministic_features)} feature(s) can nearly "
                        "determine the target. This may indicate data leakage."
                    ),
                    evidence={
                        "high_correlation_features": {col: corr for col, corr in high_corr_cols},
                        "target_deterministic_features": deterministic_features,
                        "predictability_threshold": max_leakage_predictability,
                    },
                    recommendation=(
                        "Investigate whether these features could be derived from or contain "
                        "the target variable. Remove them if leakage is confirmed."
                    ),
                    category="Data Quality",
                    remediation_effort=RemediationEffort.HIGH,
                    standards_refs=["EU-AI-ACT-ART-10"],
                )
            )

    # Numeric outliers
    outlier_columns = {}
    for col in numeric_cols:
        values = df[col].dropna()
        if values.empty:
            continue
        lower_quartile, upper_quartile = values.quantile([0.25, 0.75])
        iqr = upper_quartile - lower_quartile
        if iqr <= 0:
            outlier_pct = float((values != values.median()).mean())
        else:
            lower = lower_quartile - outlier_iqr_multiplier * iqr
            upper = upper_quartile + outlier_iqr_multiplier * iqr
            outlier_pct = float(((values < lower) | (values > upper)).mean())
        if outlier_pct > max_outlier_pct:
            outlier_columns[col] = round(outlier_pct, 4)

    findings.append(
        AuditFinding(
            check_id="DQ-006",
            title=(
                "Numeric outliers detected"
                if outlier_columns
                else "No significant numeric outliers"
            ),
            severity=Severity.MEDIUM if outlier_columns else Severity.PASSED,
            description=(
                f"{len(outlier_columns)} numeric column(s) exceed the outlier-rate threshold "
                f"of {max_outlier_pct:.0%} using the {outlier_iqr_multiplier:g}x IQR rule."
            ),
            evidence={
                "columns": outlier_columns,
                "threshold": max_outlier_pct,
                "iqr_multiplier": outlier_iqr_multiplier,
            },
            recommendation=(
                "Review extreme values and decide whether to correct, cap, transform, or retain "
                "them based on domain meaning."
                if outlier_columns
                else ""
            ),
            category="Data Quality",
            remediation_effort=RemediationEffort.MEDIUM,
            standards_refs=["EU-AI-ACT-ART-10"],
        )
    )

    if scan_pii:
        pii_matches = _pii_matches(df)
        findings.append(
            AuditFinding(
                check_id="DQ-007",
                title=(
                    "Potential PII values detected"
                    if pii_matches
                    else "No common PII patterns detected"
                ),
                severity=Severity.HIGH if pii_matches else Severity.PASSED,
                description=(
                    f"Potential PII patterns were found in {len(pii_matches)} column(s)."
                    if pii_matches
                    else "No email address, phone number, or SSN patterns were found."
                ),
                evidence={"columns": pii_matches, "patterns_checked": sorted(_PII_PATTERNS)},
                recommendation=(
                    "Remove, tokenize, or explicitly approve PII fields before model training."
                    if pii_matches
                    else ""
                ),
                category="Privacy",
                remediation_effort=RemediationEffort.HIGH,
                standards_refs=["EU-AI-ACT-ART-10"],
            )
        )

    return findings


def split_data_quality_findings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_columns: list[str] | None = None,
) -> list[AuditFinding]:
    """Check entity overlap and exact duplicate rows across train and test splits."""
    if not isinstance(train, pd.DataFrame) or not isinstance(test, pd.DataFrame):
        raise TypeError("train and test must be pandas DataFrames.")
    id_columns = id_columns or []
    missing_id_columns = sorted(set(id_columns) - set(train.columns).intersection(test.columns))
    if missing_id_columns:
        raise ValueError(f"id_columns must exist in both splits: {missing_id_columns}")

    findings = []
    if id_columns:
        train_ids = _row_keys(train, id_columns)
        test_ids = _row_keys(test, id_columns)
        overlap = train_ids.intersection(test_ids)
        findings.append(
            AuditFinding(
                check_id="DQ-008",
                title=(
                    "Train/test entity overlap detected"
                    if overlap
                    else "No train/test entity overlap"
                ),
                severity=Severity.HIGH if overlap else Severity.PASSED,
                description=(
                    f"{len(overlap)} entity identifier(s) occur in both train and test splits."
                ),
                evidence={
                    "id_columns": id_columns,
                    "overlapping_entity_count": len(overlap),
                },
                recommendation=(
                    "Rebuild dataset splits so each entity belongs to only one split."
                    if overlap
                    else ""
                ),
                category="Data Quality",
                remediation_effort=RemediationEffort.HIGH,
                standards_refs=["EU-AI-ACT-ART-10"],
            )
        )
    else:
        findings.append(
            AuditFinding(
                check_id="DQ-008",
                title="Train/test entity overlap check skipped",
                severity=Severity.INFO,
                description="No id_columns were supplied for entity-level overlap detection.",
                evidence={"id_columns": []},
                recommendation="Provide stable entity identifiers when split leakage is a risk.",
                category="Data Quality",
            )
        )

    shared_columns = list(train.columns.intersection(test.columns))
    train_rows = _row_keys(train, shared_columns)
    test_rows = _row_keys(test, shared_columns)
    duplicate_rows = train_rows.intersection(test_rows)
    findings.append(
        AuditFinding(
            check_id="DQ-009",
            title=(
                "Cross-split duplicate rows detected"
                if duplicate_rows
                else "No cross-split duplicates"
            ),
            severity=Severity.HIGH if duplicate_rows else Severity.PASSED,
            description=(
                f"{len(duplicate_rows)} exact row value(s) occur in both train and test splits."
            ),
            evidence={
                "shared_columns": shared_columns,
                "cross_split_duplicate_count": len(duplicate_rows),
            },
            recommendation=(
                "Remove duplicate records and rebuild the split before evaluating the model."
                if duplicate_rows
                else ""
            ),
            category="Data Quality",
            remediation_effort=RemediationEffort.HIGH,
            standards_refs=["EU-AI-ACT-ART-10"],
        )
    )
    return findings


def _target_deterministic_features(
    df: pd.DataFrame,
    y_true: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    suspicious = {}
    target = pd.Series(y_true, index=df.index)
    for col in df.columns:
        values = df[col]
        valid = values.notna() & target.notna()
        if not valid.any():
            continue
        unique_count = values[valid].nunique()
        if unique_count < 2 or unique_count > max(20, int(valid.sum() ** 0.5)):
            continue
        grouped = pd.DataFrame({"feature": values[valid], "target": target[valid]}).groupby(
            "feature",
            dropna=False,
        )
        predictable = grouped["target"].value_counts().groupby(level=0).max().sum()
        predictability = float(predictable / valid.sum())
        if predictability >= threshold:
            suspicious[col] = round(predictability, 4)
    return suspicious


def _pii_matches(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    matches = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        pattern_counts = {
            name: int(df[col].fillna("").astype(str).str.contains(pattern).sum())
            for name, pattern in _PII_PATTERNS.items()
        }
        found = {name: count for name, count in pattern_counts.items() if count}
        if found:
            matches[col] = found
    return matches


def _row_keys(df: pd.DataFrame, columns: list[str]) -> set[int]:
    if not columns:
        return set()
    return set(pd.util.hash_pandas_object(df[columns], index=False).astype(int))
