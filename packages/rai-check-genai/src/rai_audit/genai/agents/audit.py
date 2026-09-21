from __future__ import annotations

from datetime import datetime, timezone

from rai_audit.genai.agents.checks import (
    control_flow_findings,
    handoff_findings,
    identity_and_credential_findings,
    memory_findings,
    memory_poisoning_findings,
    permission_findings,
    prompt_injection_findings,
    resource_budget_findings,
    reversible_action_findings,
    tool_argument_policy_findings,
    tool_manifest_findings,
    tool_use_findings,
    trust_and_alignment_findings,
)
from rai_audit.genai.agents.models import AgentTrace
from rai_audit.core.engine import BaseAudit
from rai_audit.core.findings import AuditReport
from rai_audit.core.history import save_run
from rai_audit.core.scoring import compute_risk_matrix


class AgentAudit(BaseAudit):
    """Audit an agent execution trace for tool, memory, permission, and injection risks."""

    def __init__(
        self,
        trace: AgentTrace,
        *,
        allowed_tools: list[str] | tuple[str, ...] | None = None,
        project_name: str | None = None,
        metadata: dict | None = None,
        thresholds: dict | None = None,
        tool_argument_policies: dict | None = None,
        persist: bool = True,
    ):
        self.trace = trace
        self.allowed_tools = allowed_tools
        self.project_name = project_name or trace.workflow_name
        self.metadata = metadata or {}
        self.thresholds = thresholds or {}
        self.tool_argument_policies = tool_argument_policies or {}
        self.persist = persist

    def run(self) -> AuditReport:
        findings = [
            *tool_use_findings(
                self.trace,
                allowed_tools=self.allowed_tools,
                max_tool_errors=self.thresholds.get("max_tool_errors", 0),
            ),
            *tool_argument_policy_findings(self.trace, policies=self.tool_argument_policies),
            *identity_and_credential_findings(self.trace),
            *tool_manifest_findings(self.trace),
            *memory_findings(self.trace),
            *memory_poisoning_findings(self.trace),
            *permission_findings(self.trace),
            *prompt_injection_findings(self.trace),
            *resource_budget_findings(
                self.trace,
                max_tool_calls=self.thresholds.get("max_tool_calls", 50),
                max_consecutive_tool_calls=self.thresholds.get("max_consecutive_tool_calls", 10),
            ),
            *control_flow_findings(
                self.trace,
                max_recursion_depth=self.thresholds.get("max_recursion_depth", 8),
                max_retries=self.thresholds.get("max_retries", 3),
            ),
            *reversible_action_findings(self.trace),
            *handoff_findings(self.trace),
            *trust_and_alignment_findings(self.trace),
        ]
        timestamp = datetime.now(timezone.utc).isoformat()
        for finding in findings:
            finding.timestamp = timestamp
        report = AuditReport(
            project_name=self.project_name,
            audit_type="agent_trace",
            risk_matrix=compute_risk_matrix(findings),
            findings=findings,
            metadata={
                **self.trace.metadata,
                **self.metadata,
                "trace_id": self.trace.trace_id,
                "group_id": self.trace.group_id,
                "workflow_name": self.trace.workflow_name,
                "event_count": len(self.trace.events),
                "owasp_agentic_top_10_2026": self._owasp_coverage(findings),
            },
        )
        if self.persist:
            save_run(report.to_dict())
        return report

    @staticmethod
    def _owasp_coverage(findings):
        from rai_audit.genai.agents.owasp import owasp_agentic_coverage

        return owasp_agentic_coverage(findings)
