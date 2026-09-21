from __future__ import annotations

from rai_audit.core.findings import AuditFinding, Severity
from rai_audit.dl.image import ImageClassificationAudit

_SCIENTIFIC_STANDARDS = ["EU-AI-ACT-ART-10", "NIST-AI-RMF-MAP-1"]


class ScientificAIAudit(ImageClassificationAudit):
    """Image audit with scientific domain metadata for reproducible evaluations."""

    audit_type = "scientific_image_classification"

    def __init__(
        self,
        *args,
        scientific_domain: str = "scientific imaging",
        reproducibility_metadata: dict | None = None,
        quality_metrics: dict | None = None,
        **kwargs,
    ):
        self.scientific_domain = scientific_domain
        self.reproducibility_metadata = reproducibility_metadata or {}
        self.quality_metrics = quality_metrics or {}
        super().__init__(*args, **kwargs)

    def _additional_metadata(self) -> dict:
        return {
            "scientific_domain": self.scientific_domain,
            "reproducibility_metadata": self.reproducibility_metadata,
            "quality_metrics": self.quality_metrics,
        }

    def _additional_findings(self) -> list[AuditFinding]:
        required = ("dataset_version", "model_version", "random_seed", "environment")
        missing = [key for key in required if key not in self.reproducibility_metadata]
        invalid_quality = {
            name: value
            for name, value in self.quality_metrics.items()
            if not isinstance(value, (int, float)) or value < 0
        }
        return [
            AuditFinding(
                check_id="SCI-REPRO-001",
                title="Scientific reproducibility metadata is incomplete"
                if missing
                else "Scientific reproducibility metadata",
                severity=Severity.MEDIUM if missing else Severity.PASSED,
                description=f"{len(missing)} required reproducibility field(s) are missing.",
                evidence={"required_fields": list(required), "missing_fields": missing},
                recommendation=(
                    "Record dataset, model, seed, and environment versions for every run."
                ),
                category="Reproducibility",
                standards_refs=_SCIENTIFIC_STANDARDS,
            ),
            AuditFinding(
                check_id="SCI-DATA-001",
                title="Invalid scientific data-quality metrics"
                if invalid_quality
                else "Scientific data-quality metadata",
                severity=Severity.MEDIUM if invalid_quality else Severity.PASSED,
                description="Configured domain-specific data-quality metrics were validated.",
                evidence={
                    "quality_metrics": self.quality_metrics,
                    "invalid_metrics": invalid_quality,
                },
                recommendation=(
                    "Record non-negative domain quality metrics and define review thresholds."
                ),
                category="Data Quality",
                standards_refs=_SCIENTIFIC_STANDARDS,
            ),
        ]
