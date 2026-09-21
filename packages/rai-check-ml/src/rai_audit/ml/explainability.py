from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rai_audit.core.findings import AuditFinding, RemediationEffort, Severity

if TYPE_CHECKING:
    import pandas as pd


def explainability_findings(
    feature_importances: dict[str, float],
    feature_names: list[str] | None = None,
    top_n: int = 10,
    max_concentration: float = 0.8,
) -> list[AuditFinding]:
    """
    Generate explainability findings from a feature importance dict.

    Args:
        feature_importances: {feature_name: importance_score}
        top_n: number of top features to report
        max_concentration: flag if top feature accounts for this fraction of total importance
    """
    findings: list[AuditFinding] = []

    if not feature_importances:
        return findings

    total = sum(abs(v) for v in feature_importances.values())
    if total == 0:
        return findings

    sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]

    top_importance = abs(sorted_features[0][1]) / total if total > 0 else 0
    severity = Severity.MEDIUM if top_importance > max_concentration else Severity.PASSED

    findings.append(
        AuditFinding(
            check_id="EXPL-001",
            title="Feature importance concentration",
            severity=severity,
            description=(
                f"Top feature '{sorted_features[0][0]}' accounts for "
                f"{top_importance*100:.1f}% of total feature importance. "
                + (
                    "High concentration may indicate over-reliance on a single feature."
                    if top_importance > max_concentration
                    else "Feature importance is reasonably distributed."
                )
            ),
            evidence={
                "top_feature": sorted_features[0][0],
                "top_feature_importance_pct": round(top_importance, 4),
                "top_features": {f: round(abs(v) / total, 4) for f, v in top_features},
            },
            recommendation=(
                "Investigate whether the dominant feature is appropriate. "
                "Check for proxy variables that could encode sensitive attributes."
            ) if top_importance > max_concentration else "",
            category="Transparency",
            remediation_effort=RemediationEffort.MEDIUM,
            standards_refs=["EU-AI-ACT-ART-13", "NIST-AI-RMF-MEASURE-2.5"],
        )
    )

    return findings


def shap_explainability_findings(
    model,
    X: "pd.DataFrame",
    top_n: int = 10,
    max_concentration: float = 0.8,
) -> list[AuditFinding]:
    """
    Generate explainability findings using SHAP values.
    Requires rai-check-ml[explainability] (shap>=0.44).
    """
    try:
        import shap
    except ImportError:
        return [
            AuditFinding(
                check_id="EXPL-SHAP-000",
                title="SHAP not installed",
                severity=Severity.INFO,
                description="Install rai-check-ml[explainability] to enable SHAP-based explainability.",
                evidence={},
                recommendation="pip install rai-check-ml[explainability]",
                category="Transparency",
            )
        ]

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    importances = {str(col): float(imp) for col, imp in zip(X.columns, mean_abs_shap)}
    return explainability_findings(importances, top_n=top_n, max_concentration=max_concentration)
