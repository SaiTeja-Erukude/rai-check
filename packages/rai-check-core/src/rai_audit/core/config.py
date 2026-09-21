from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rai_audit.core.evidence import (
    build_evidence_manifest,
    finalize_evidence_manifest,
    reproducibility_metadata,
    write_evidence_manifest,
)
from rai_audit.core.findings import AuditReport
from rai_audit.core.privacy import check_pii_in_dataframe
from rai_audit.core.reproducibility import check_reproducibility
from rai_audit.core.schemas import SchemaDocumentError, prepare_document
from rai_audit.core.scoring import compute_risk_matrix, gate_check
from rai_audit.core.standards import build_standards_crosswalk


class ConfigValidationError(ValueError):
    """Raised when an audit configuration cannot be loaded or executed."""


@dataclass(frozen=True)
class ConfigRunResult:
    report: AuditReport
    artifacts: dict[str, Path]
    manifest_path: Path
    gate_passed: bool
    gate_reason: str


def load_audit_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML audit configuration."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigValidationError(f"Could not read config {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigValidationError("audit config must be a mapping")
    try:
        return prepare_document("config", raw)
    except SchemaDocumentError as exc:
        raise ConfigValidationError(f"Invalid audit config: {exc}") from exc


def run_config(path: str | Path) -> ConfigRunResult:
    """Run an audit from YAML and write reports plus an evidence manifest."""
    config_path = Path(path)
    config = load_audit_config(config_path)
    project = _mapping(config.get("project", {}), "project")
    audit_config = _mapping(config.get("audit", {}), "audit")
    checks = _mapping(config.get("checks", {}), "checks")
    audit_type = str(audit_config.get("type", "classification"))
    project_name = str(project.get("name", "RAI Check"))
    metadata = {
        **_mapping(project.get("metadata", {}), "project.metadata"),
        **_mapping(audit_config.get("metadata", {}), "audit.metadata"),
    }
    thresholds = _thresholds(checks)

    report, inputs, frames = _dispatch(
        audit_type,
        audit_config,
        project_name=project_name,
        metadata=metadata,
        thresholds=thresholds,
        checks=checks,
        base_directory=config_path.parent,
    )
    manifest = build_evidence_manifest(
        report,
        config=config,
        inputs=inputs,
        metadata=metadata,
        working_directory=config_path.parent,
    )
    if _enabled(checks, "privacy", default=True):
        for frame in frames:
            report.findings.extend(check_pii_in_dataframe(frame))
    if _enabled(checks, "reproducibility", default=True):
        report.findings.extend(check_reproducibility(reproducibility_metadata(manifest, metadata)))
    report.risk_matrix = compute_risk_matrix(report.findings)
    report.metadata["evidence_manifest"] = manifest
    report.metadata["standards_crosswalk"] = build_standards_crosswalk(report.findings)

    output_directory = _resolve_path(
        config_path.parent,
        audit_config.get("output_dir", "./audit-report"),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts = _write_report_artifacts(
        report,
        output_directory,
        _string_list(audit_config.get("report_formats", ["html", "json"]), "audit.report_formats"),
    )
    manifest = finalize_evidence_manifest(manifest, artifacts)
    manifest_path = output_directory / "evidence-manifest.json"
    write_evidence_manifest(manifest, manifest_path)

    gate = _mapping(config.get("gate", {}), "gate")
    gate_passed, gate_reason = gate_check(
        report.to_dict(),
        min_score=gate.get("min_score"),
        fail_on_critical=bool(gate.get("fail_on_critical", True)),
    )
    return ConfigRunResult(
        report=report,
        artifacts=artifacts,
        manifest_path=manifest_path,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
    )


def _dispatch(
    audit_type: str,
    audit_config: dict[str, Any],
    *,
    project_name: str,
    metadata: dict[str, Any],
    thresholds: dict[str, Any],
    checks: dict[str, Any],
    base_directory: Path,
) -> tuple[AuditReport, dict[str, Path], list[Any]]:
    if audit_type in {"classification", "regression"}:
        return _tabular_audit(
            audit_type,
            audit_config,
            project_name,
            metadata,
            thresholds,
            checks,
            base_directory,
        )
    if audit_type == "drift":
        return _drift_audit(audit_config, project_name, metadata, thresholds, base_directory)
    if audit_type in {"llm", "rag", "rag-security"}:
        return _llm_audit(audit_type, audit_config, project_name, metadata, base_directory)
    if audit_type == "agents":
        return _agent_audit(audit_config, project_name, metadata, thresholds, base_directory)
    if audit_type in {"image", "medical", "scientific"}:
        return _image_audit(
            audit_type,
            audit_config,
            project_name,
            metadata,
            thresholds,
            base_directory,
        )
    raise ConfigValidationError(f"Unsupported audit.type: {audit_type}")


def _tabular_audit(
    audit_type: str,
    audit_config: dict[str, Any],
    project_name: str,
    metadata: dict[str, Any],
    thresholds: dict[str, Any],
    checks: dict[str, Any],
    base_directory: Path,
) -> tuple[AuditReport, dict[str, Path], list[Any]]:
    import pandas as pd
    from rai_audit.ml import ClassificationAudit, RegressionAudit

    path = _required_path(audit_config, "data", base_directory)
    frame = pd.read_csv(path)
    target = str(audit_config.get("target", "y_true"))
    prediction = str(audit_config.get("prediction", "y_pred"))
    _require_columns(frame, [target, prediction])
    sensitive_columns = _string_list(
        audit_config.get("sensitive_features", []),
        "sensitive_features",
    )
    _require_columns(frame, sensitive_columns)
    sensitive = (
        frame[sensitive_columns] if sensitive_columns and _enabled(checks, "fairness") else None
    )
    excluded = {target, prediction, *sensitive_columns}
    probability = audit_config.get("probability")
    if probability:
        probability = str(probability)
        _require_columns(frame, [probability])
        excluded.add(probability)
    features = frame[[column for column in frame.columns if column not in excluded]]

    common = {
        "y_true": frame[target].values,
        "y_pred": frame[prediction].values,
        "sensitive_features": sensitive,
        "data": features if _enabled(checks, "data_quality") else None,
        "project_name": project_name,
        "metadata": metadata,
        "thresholds": thresholds,
        "persist": bool(audit_config.get("persist", True)),
    }
    if audit_type == "classification":
        audit = ClassificationAudit(
            **common,
            y_prob=(
                frame[probability].values
                if probability and _enabled(checks, "robustness")
                else None
            ),
            positive_label=audit_config.get("positive_label"),
            include_intersections=bool(audit_config.get("include_intersections", False)),
            feature_importances=audit_config.get("feature_importances"),
        )
    else:
        audit = RegressionAudit(**common)
    return audit.run(), {"data": path}, [frame]


def _drift_audit(
    audit_config: dict[str, Any],
    project_name: str,
    metadata: dict[str, Any],
    thresholds: dict[str, Any],
    base_directory: Path,
) -> tuple[AuditReport, dict[str, Path], list[Any]]:
    import pandas as pd
    from rai_audit.ml import DriftAudit

    reference_path = _required_path(audit_config, "reference_data", base_directory)
    current_path = _required_path(audit_config, "current_data", base_directory)
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    sensitive_columns = _string_list(
        audit_config.get("sensitive_features", []),
        "sensitive_features",
    )
    _require_columns(reference, sensitive_columns)
    _require_columns(current, sensitive_columns)
    feature_columns = _string_list(
        audit_config.get("feature_columns", []),
        "audit.feature_columns",
    )
    if feature_columns:
        _require_columns(reference, feature_columns)
        _require_columns(current, feature_columns)
        reference_features = reference[feature_columns]
        current_features = current[feature_columns]
    else:
        excluded = set(sensitive_columns)
        excluded.update(
            str(value)
            for key in ("target", "prediction")
            if (value := audit_config.get(key))
        )
        reference_features = reference[
            [column for column in reference.columns if column not in excluded]
        ]
        current_features = current[[column for column in current.columns if column not in excluded]]
    target = audit_config.get("target")
    prediction = audit_config.get("prediction")
    if target:
        _require_columns(reference, [str(target)])
        _require_columns(current, [str(target)])
    if prediction:
        _require_columns(reference, [str(prediction)])
        _require_columns(current, [str(prediction)])
    audit = DriftAudit(
        reference=reference_features,
        current=current_features,
        reference_sensitive_features=reference[sensitive_columns] if sensitive_columns else None,
        current_sensitive_features=current[sensitive_columns] if sensitive_columns else None,
        y_true_ref=reference[str(target)].values if target else None,
        y_pred_ref=reference[str(prediction)].values if prediction else None,
        y_true_cur=current[str(target)].values if target else None,
        y_pred_cur=current[str(prediction)].values if prediction else None,
        project_name=project_name,
        metadata=metadata,
        thresholds=thresholds,
        persist=bool(audit_config.get("persist", True)),
    )
    return audit.run(), {"reference_data": reference_path, "current_data": current_path}, [
        reference,
        current,
    ]


def _llm_audit(
    audit_type: str,
    audit_config: dict[str, Any],
    project_name: str,
    metadata: dict[str, Any],
    base_directory: Path,
) -> tuple[AuditReport, dict[str, Path], list[Any]]:
    from rai_audit.genai.llm import LLMAudit, RAGAudit, RAGSecurityAudit, load_test_suite

    suite_path = _required_path(audit_config, "suite", base_directory)
    suite = load_test_suite(suite_path)
    audit_class = {"llm": LLMAudit, "rag": RAGAudit, "rag-security": RAGSecurityAudit}[audit_type]
    report = audit_class(
        suite,
        project_name=project_name,
        metadata=metadata,
        persist=bool(audit_config.get("persist", True)),
    ).run()
    return report, {"suite": suite_path}, []


def _agent_audit(
    audit_config: dict[str, Any],
    project_name: str,
    metadata: dict[str, Any],
    thresholds: dict[str, Any],
    base_directory: Path,
) -> tuple[AuditReport, dict[str, Path], list[Any]]:
    from rai_audit.genai.agents import AgentAudit, load_trace

    trace_path = _required_path(audit_config, "trace", base_directory)
    report = AgentAudit(
        load_trace(trace_path),
        allowed_tools=_string_list(audit_config.get("allowed_tools", []), "audit.allowed_tools")
        or None,
        project_name=project_name,
        metadata=metadata,
        thresholds=thresholds,
        persist=bool(audit_config.get("persist", True)),
    ).run()
    return report, {"trace": trace_path}, []


def _image_audit(
    audit_type: str,
    audit_config: dict[str, Any],
    project_name: str,
    metadata: dict[str, Any],
    thresholds: dict[str, Any],
    base_directory: Path,
) -> tuple[AuditReport, dict[str, Path], list[Any]]:
    import pandas as pd
    from rai_audit.dl import ImageClassificationAudit, MedicalImagingAudit, ScientificAIAudit

    path = _required_path(audit_config, "data", base_directory)
    frame = pd.read_csv(path)
    target = str(audit_config.get("target", "y_true"))
    prediction = str(audit_config.get("prediction", "y_pred"))
    _require_columns(frame, [target, prediction])
    transformed_prefix = str(audit_config.get("transformed_prefix", "transform_"))
    common = {
        "y_true": frame[target].values,
        "y_pred": frame[prediction].values,
        "transformed_predictions": {
            column.removeprefix(transformed_prefix): frame[column].values
            for column in frame.columns
            if column.startswith(transformed_prefix)
        },
        "project_name": project_name,
        "metadata": metadata,
        "thresholds": thresholds,
        "persist": bool(audit_config.get("persist", True)),
    }
    if audit_type == "image":
        audit = ImageClassificationAudit(**common)
    elif audit_type == "medical":
        patient_id = audit_config.get("patient_id")
        split = audit_config.get("split")
        site = audit_config.get("site")
        _require_columns(frame, [str(value) for value in (patient_id, split, site) if value])
        audit = MedicalImagingAudit(
            **common,
            patient_ids=frame[str(patient_id)].values if patient_id else None,
            splits=frame[str(split)].values if split else None,
            sites=frame[str(site)].values if site else None,
        )
    else:
        audit = ScientificAIAudit(
            **common,
            scientific_domain=str(audit_config.get("domain", "scientific imaging")),
        )
    return audit.run(), {"data": path}, [frame]


def _write_report_artifacts(
    report: AuditReport,
    output_directory: Path,
    formats: list[str],
) -> dict[str, Path]:
    artifacts = {}
    for format_name in formats:
        if format_name == "html":
            path = output_directory / "audit-report.html"
            report.to_html(str(path))
        elif format_name in {"markdown", "md"}:
            path = output_directory / "audit-report.md"
            report.to_markdown(str(path))
        elif format_name == "json":
            path = output_directory / "audit-report.json"
            report.to_json(str(path))
        elif format_name == "sarif":
            path = output_directory / "audit-report.sarif"
            report.to_sarif(str(path))
        elif format_name == "junit":
            path = output_directory / "audit-report.junit.xml"
            report.to_junit(str(path))
        elif format_name in {"standards", "standards-coverage"}:
            path = output_directory / "standards-coverage.json"
            report.to_standards_coverage(str(path))
        else:
            raise ConfigValidationError(f"Unsupported report format: {format_name}")
        artifacts[format_name] = path
    return artifacts


def _thresholds(checks: dict[str, Any]) -> dict[str, Any]:
    thresholds = {}
    for settings in checks.values():
        if isinstance(settings, dict):
            thresholds.update({key: value for key, value in settings.items() if key != "enabled"})
    return thresholds


def _enabled(checks: dict[str, Any], name: str, *, default: bool = True) -> bool:
    settings = checks.get(name, {})
    if isinstance(settings, dict):
        return bool(settings.get("enabled", default))
    if isinstance(settings, bool):
        return settings
    return default


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{label} must be a mapping")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigValidationError(f"{label} must be a list of strings")
    return value


def _resolve_path(base_directory: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_directory / path


def _required_path(config: dict[str, Any], key: str, base_directory: Path) -> Path:
    if not config.get(key):
        raise ConfigValidationError(f"audit.{key} is required")
    path = _resolve_path(base_directory, config[key])
    if not path.exists():
        raise ConfigValidationError(f"{path} not found")
    return path


def _require_columns(frame: Any, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ConfigValidationError(f"Columns not found: {missing}")
