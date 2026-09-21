from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from rai_audit.core.schemas import SchemaDocumentError, prepare_document
from rai_audit.genai.llm.models import SUPPORTED_CHECKS, LLMTestCase, LLMTestSuite, RAGContext

_TYPE_CHECKS = {
    "prompt_injection": ("prompt_injection",),
    "unsafe_output": ("unsafe_output", "toxicity"),
    "rag": (
        "rag_faithfulness",
        "rag_citation",
        "rag_security",
        "rag_retrieval",
        "rag_provenance",
        "rag_tenant_isolation",
        "rag_stale_context",
        "rag_poisoned_document",
    ),
    "rag_faithfulness": ("rag_faithfulness",),
    "rag_citation": ("rag_citation",),
    "rag_security": ("rag_security",),
    "rag_retrieval": ("rag_retrieval",),
    "rag_provenance": ("rag_provenance",),
    "rag_tenant_isolation": ("rag_tenant_isolation",),
    "rag_stale_context": ("rag_stale_context",),
    "rag_poisoned_document": ("rag_poisoned_document",),
}


class SuiteValidationError(ValueError):
    """Raised when an LLM YAML suite does not match the supported schema."""


def load_test_suite(path: str | Path) -> LLMTestSuite:
    """Load and validate an LLM audit suite from YAML."""
    suite_path = Path(path)
    try:
        raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SuiteValidationError(f"Could not read suite {suite_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SuiteValidationError(f"Invalid YAML in {suite_path}: {exc}") from exc

    try:
        root = prepare_document("suite", raw)
    except SchemaDocumentError as exc:
        raise SuiteValidationError(f"Invalid suite schema: {exc}") from exc
    name = _required_text(root, "name", "suite")
    project_name = _optional_text(root.get("project_name"), "suite.project_name")
    defaults = _mapping(root.get("defaults", {}), "suite.defaults")
    default_forbidden = _text_tuple(
        defaults.get("forbidden_terms", ()), "suite.defaults.forbidden_terms"
    )
    default_max_context_age_days = _optional_positive_int(
        defaults.get("max_context_age_days"),
        "suite.defaults.max_context_age_days",
    )
    default_min_retrieval_recall = _fraction(
        defaults.get("min_retrieval_recall", 1.0),
        "suite.defaults.min_retrieval_recall",
    )
    default_require_context_provenance = _bool(
        defaults.get("require_context_provenance", True),
        "suite.defaults.require_context_provenance",
    )

    raw_cases = root.get("cases", root.get("tests"))
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteValidationError("suite.cases must be a non-empty list")

    cases: list[LLMTestCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_cases):
        label = f"suite.cases[{index}]"
        case = _mapping(item, label)
        case_id = _required_text(case, "id", label)
        if case_id in seen_ids:
            raise SuiteValidationError(f"Duplicate test case id: {case_id}")
        seen_ids.add(case_id)

        checks = _checks(case, label)
        contexts = _contexts(case.get("contexts", ()), label)
        if "rag_faithfulness" in checks and not contexts:
            raise SuiteValidationError(f"{label}.contexts is required for rag_faithfulness")

        cases.append(
            LLMTestCase(
                id=case_id,
                prompt=_required_text(case, "prompt", label),
                checks=checks,
                response=_optional_text(case.get("response"), f"{label}.response"),
                expected_refusal=_bool(
                    case.get("expected_refusal", True), f"{label}.expected_refusal"
                ),
                forbidden_terms=default_forbidden
                + _text_tuple(case.get("forbidden_terms", ()), f"{label}.forbidden_terms"),
                contexts=contexts,
                expected_citations=_text_tuple(
                    case.get("expected_citations", ()), f"{label}.expected_citations"
                ),
                relevant_sources=_text_tuple(
                    case.get("relevant_sources", ()), f"{label}.relevant_sources"
                ),
                retrieval_k=_optional_positive_int(case.get("retrieval_k"), f"{label}.retrieval_k"),
                tenant_id=_optional_text(case.get("tenant_id"), f"{label}.tenant_id"),
                max_context_age_days=_optional_positive_int(
                    case.get("max_context_age_days", default_max_context_age_days),
                    f"{label}.max_context_age_days",
                ),
                min_retrieval_recall=_fraction(
                    case.get("min_retrieval_recall", default_min_retrieval_recall),
                    f"{label}.min_retrieval_recall",
                ),
                require_context_provenance=_bool(
                    case.get(
                        "require_context_provenance",
                        default_require_context_provenance,
                    ),
                    f"{label}.require_context_provenance",
                ),
                evaluated_at=_optional_text(case.get("evaluated_at"), f"{label}.evaluated_at"),
                judge_result=case.get("judge_result"),
                output_schema=_optional_mapping(
                    case.get("output_schema"), f"{label}.output_schema"
                ),
                max_latency_ms=_optional_positive_number(
                    case.get("max_latency_ms"), f"{label}.max_latency_ms"
                ),
                max_total_tokens=_optional_positive_int(
                    case.get("max_total_tokens"), f"{label}.max_total_tokens"
                ),
                max_cost_usd=_optional_positive_number(
                    case.get("max_cost_usd"), f"{label}.max_cost_usd"
                ),
                metadata=_mapping(case.get("metadata", {}), f"{label}.metadata"),
            )
        )

    return LLMTestSuite(
        name=name,
        project_name=project_name,
        cases=tuple(cases),
        metadata=_mapping(root.get("metadata", {}), "suite.metadata"),
    )


def _checks(case: Mapping[str, Any], label: str) -> tuple[str, ...]:
    raw_checks = case.get("checks")
    if raw_checks is None:
        case_type = _required_text(case, "type", label)
        if case_type not in _TYPE_CHECKS:
            supported = ", ".join(sorted(_TYPE_CHECKS))
            raise SuiteValidationError(f"{label}.type must be one of: {supported}")
        return _TYPE_CHECKS[case_type]

    checks = _text_tuple(raw_checks, f"{label}.checks")
    unknown = sorted(set(checks) - SUPPORTED_CHECKS)
    if unknown:
        raise SuiteValidationError(
            f"{label}.checks contains unsupported checks: {', '.join(unknown)}"
        )
    if not checks:
        raise SuiteValidationError(f"{label}.checks must not be empty")
    return checks


def _contexts(raw: Any, label: str) -> tuple[RAGContext, ...]:
    if not isinstance(raw, (list, tuple)):
        raise SuiteValidationError(f"{label}.contexts must be a list")
    contexts: list[RAGContext] = []
    for index, item in enumerate(raw):
        context_label = f"{label}.contexts[{index}]"
        if isinstance(item, str):
            contexts.append(RAGContext(source=f"context-{index + 1}", content=item))
            continue
        context = _mapping(item, context_label)
        contexts.append(
            RAGContext(
                source=_required_text(context, "source", context_label),
                content=_required_text(context, "content", context_label),
                trusted=_bool(context.get("trusted", False), f"{context_label}.trusted"),
                document_id=_optional_text(
                    context.get("document_id"),
                    f"{context_label}.document_id",
                ),
                tenant_id=_optional_text(context.get("tenant_id"), f"{context_label}.tenant_id"),
                updated_at=_optional_text(
                    context.get("updated_at"),
                    f"{context_label}.updated_at",
                ),
                poisoned=_bool(context.get("poisoned", False), f"{context_label}.poisoned"),
                metadata=_mapping(context.get("metadata", {}), f"{context_label}.metadata"),
            )
        )
    return tuple(contexts)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SuiteValidationError(f"{label} must be a mapping")
    return value


def _optional_mapping(value: Any, label: str) -> Mapping[str, Any] | None:
    return None if value is None else _mapping(value, label)


def _required_text(value: Mapping[str, Any], key: str, label: str) -> str:
    text = _optional_text(value.get(key), f"{label}.{key}")
    if text is None:
        raise SuiteValidationError(f"{label}.{key} is required")
    return text


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SuiteValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _text_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise SuiteValidationError(f"{label} must be a list")
    result: list[str] = []
    for item in value:
        text = _optional_text(item, label)
        if text is None:
            raise SuiteValidationError(f"{label} entries must be non-empty strings")
        result.append(text)
    return tuple(result)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SuiteValidationError(f"{label} must be a boolean")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SuiteValidationError(f"{label} must be a positive integer")
    return value


def _fraction(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SuiteValidationError(f"{label} must be a number between 0 and 1")
    fraction = float(value)
    if not 0 <= fraction <= 1:
        raise SuiteValidationError(f"{label} must be between 0 and 1")
    return fraction


def _optional_positive_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise SuiteValidationError(f"{label} must be a positive number")
    return float(value)
