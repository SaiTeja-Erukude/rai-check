from __future__ import annotations

from datetime import datetime, timezone

from rai_audit.core.engine import BaseAudit
from rai_audit.core.findings import AuditFinding, AuditReport, Severity
from rai_audit.core.history import save_run
from rai_audit.core.scoring import compute_risk_matrix
from rai_audit.genai.llm.checks import (
    check_latency,
    check_pii_redaction,
    check_prompt_injection,
    check_prompt_leakage,
    check_rag_citations,
    check_rag_faithfulness,
    check_rag_poisoned_documents,
    check_rag_provenance,
    check_rag_retrieval,
    check_rag_security,
    check_rag_stale_context,
    check_rag_tenant_isolation,
    check_rate_limit,
    check_refusal_overblocking,
    check_structured_output,
    check_token_budget,
    check_toxicity,
    check_unsafe_output,
)
from rai_audit.genai.llm.models import (
    FaithfulnessJudge,
    LLMTestCase,
    LLMTestSuite,
    ProviderResponse,
    ResponseProvider,
)

_RAG_CHECKS = frozenset(
    {
        "rag_faithfulness",
        "rag_citation",
        "rag_security",
        "rag_retrieval",
        "rag_provenance",
        "rag_tenant_isolation",
        "rag_stale_context",
        "rag_poisoned_document",
    }
)
_RAG_SECURITY_CHECKS = frozenset({"rag_security", "rag_tenant_isolation", "rag_poisoned_document"})


class LLMAudit(BaseAudit):
    """Audit captured or live LLM responses using a validated YAML test suite."""

    selected_checks: frozenset[str] | None = None
    audit_type = "llm_application"

    def __init__(
        self,
        suite: LLMTestSuite,
        responder: ResponseProvider | None = None,
        faithfulness_judge: FaithfulnessJudge | None = None,
        project_name: str | None = None,
        metadata: dict | None = None,
        persist: bool = True,
    ):
        self.suite = suite
        self.responder = responder
        self.faithfulness_judge = faithfulness_judge
        self.project_name = project_name or suite.project_name or suite.name
        self.metadata = metadata or {}
        self.persist = persist

    def run(self) -> AuditReport:
        findings: list[AuditFinding] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        audited_cases = 0
        response_metrics = []

        for case in self.suite.cases:
            checks = self._checks_for_case(case)
            if not checks:
                continue
            audited_cases += 1
            response = case.response
            provider_response = None
            if response is None and self.responder is not None:
                response = self.responder(case)
            if isinstance(response, ProviderResponse):
                provider_response = response
                response_metrics.append({"test_case": case.id, **response.to_dict()})
                response = response.text
            if response is None:
                findings.append(self._missing_response_finding(case))
                continue
            if not isinstance(response, str):
                raise TypeError(f"Responder for test case '{case.id}' must return a string")

            for check in checks:
                findings.append(self._run_check(check, case, response, provider_response))

        for finding in findings:
            finding.timestamp = timestamp
        report = AuditReport(
            project_name=self.project_name,
            audit_type=self.audit_type,
            risk_matrix=compute_risk_matrix(findings),
            findings=findings,
            metadata={
                **self.suite.metadata,
                **self.metadata,
                "suite": self.suite.name,
                "suite_cases": len(self.suite.cases),
                "audited_cases": audited_cases,
                "response_metrics": response_metrics,
            },
        )
        if self.persist:
            save_run(report.to_dict())
        return report

    def _checks_for_case(self, case: LLMTestCase) -> tuple[str, ...]:
        if self.selected_checks is None:
            return case.checks
        return tuple(check for check in case.checks if check in self.selected_checks)

    def _run_check(
        self, check: str, case: LLMTestCase, response: str, provider_response=None
    ) -> AuditFinding:
        if check == "prompt_injection":
            return check_prompt_injection(case, response)
        if check == "unsafe_output":
            return check_unsafe_output(case, response)
        if check == "toxicity":
            return check_toxicity(case, response)
        if check == "pii_redaction":
            return check_pii_redaction(case, response)
        if check == "prompt_leakage":
            return check_prompt_leakage(case, response)
        if check == "refusal_overblocking":
            return check_refusal_overblocking(case, response)
        if check == "structured_output":
            return check_structured_output(case, response)
        if check == "rate_limit":
            return check_rate_limit(case, provider_response)
        if check == "latency":
            return check_latency(case, provider_response)
        if check == "token_budget":
            return check_token_budget(case, provider_response)
        if check == "rag_faithfulness":
            return check_rag_faithfulness(case, response, self.faithfulness_judge)
        if check == "rag_citation":
            return check_rag_citations(case, response)
        if check == "rag_security":
            return check_rag_security(case, response)
        if check == "rag_retrieval":
            return check_rag_retrieval(case)
        if check == "rag_provenance":
            return check_rag_provenance(case)
        if check == "rag_tenant_isolation":
            return check_rag_tenant_isolation(case)
        if check == "rag_stale_context":
            return check_rag_stale_context(case)
        if check == "rag_poisoned_document":
            return check_rag_poisoned_documents(case)
        raise ValueError(f"Unsupported check: {check}")

    @staticmethod
    def _missing_response_finding(case: LLMTestCase) -> AuditFinding:
        return AuditFinding(
            check_id=f"LLM-INPUT-{case.id}",
            title="Test case has no response",
            severity=Severity.HIGH,
            description=(
                "The suite case has no captured response and no responder callback was provided."
            ),
            evidence={"test_case": case.id},
            recommendation="Capture a response in YAML or configure a responder callback.",
            category="Audit Inputs",
        )


class RAGAudit(LLMAudit):
    """Run RAG faithfulness, citation, and retrieval security checks."""

    selected_checks = _RAG_CHECKS
    audit_type = "rag_application"


class RAGSecurityAudit(LLMAudit):
    """Run only retrieval security checks for RAG test cases."""

    selected_checks = _RAG_SECURITY_CHECKS
    audit_type = "rag_security"
