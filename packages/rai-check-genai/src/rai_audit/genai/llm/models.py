from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from rai_audit.core.schemas import SCHEMA_VERSION

SUPPORTED_CHECKS = frozenset(
    {
        "prompt_injection",
        "prompt_leakage",
        "pii_redaction",
        "refusal_overblocking",
        "structured_output",
        "rate_limit",
        "latency",
        "token_budget",
        "unsafe_output",
        "toxicity",
        "rag_faithfulness",
        "rag_citation",
        "rag_poisoned_document",
        "rag_provenance",
        "rag_retrieval",
        "rag_security",
        "rag_stale_context",
        "rag_tenant_isolation",
    }
)


@dataclass(frozen=True)
class RAGContext:
    source: str
    content: str
    trusted: bool = False
    document_id: str | None = None
    tenant_id: str | None = None
    updated_at: str | None = None
    poisoned: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "content": self.content,
            "trusted": self.trusted,
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "updated_at": self.updated_at,
            "poisoned": self.poisoned,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMTestCase:
    id: str
    prompt: str
    checks: tuple[str, ...]
    response: str | None = None
    expected_refusal: bool = True
    forbidden_terms: tuple[str, ...] = ()
    contexts: tuple[RAGContext, ...] = ()
    expected_citations: tuple[str, ...] = ()
    relevant_sources: tuple[str, ...] = ()
    retrieval_k: int | None = None
    tenant_id: str | None = None
    max_context_age_days: int | None = None
    min_retrieval_recall: float = 1.0
    require_context_provenance: bool = True
    evaluated_at: str | None = None
    judge_result: Mapping[str, Any] | bool | float | None = None
    output_schema: Mapping[str, Any] | None = None
    max_latency_ms: float | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "checks": list(self.checks),
            "response": self.response,
            "expected_refusal": self.expected_refusal,
            "forbidden_terms": list(self.forbidden_terms),
            "contexts": [context.to_dict() for context in self.contexts],
            "expected_citations": list(self.expected_citations),
            "relevant_sources": list(self.relevant_sources),
            "retrieval_k": self.retrieval_k,
            "tenant_id": self.tenant_id,
            "max_context_age_days": self.max_context_age_days,
            "min_retrieval_recall": self.min_retrieval_recall,
            "require_context_provenance": self.require_context_provenance,
            "evaluated_at": self.evaluated_at,
            "judge_result": self.judge_result,
            "output_schema": dict(self.output_schema) if self.output_schema else None,
            "max_latency_ms": self.max_latency_ms,
            "max_total_tokens": self.max_total_tokens,
            "max_cost_usd": self.max_cost_usd,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMTestSuite:
    name: str
    cases: tuple[LLMTestCase, ...]
    project_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "project_name": self.project_name,
            "cases": [case.to_dict() for case in self.cases],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized response and operational metrics from a live provider."""

    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    rate_limited: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "rate_limited": self.rate_limited,
            "metadata": dict(self.metadata),
        }


ResponseProvider = Callable[[LLMTestCase], str | ProviderResponse]
FaithfulnessJudge = Callable[[LLMTestCase, str], Mapping[str, Any] | bool | float]
