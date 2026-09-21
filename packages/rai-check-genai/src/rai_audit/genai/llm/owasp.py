from __future__ import annotations

OWASP_LLM_TOP_10_2025: dict[str, str] = {
    "OWASP-LLM-01": "Prompt Injection",
    "OWASP-LLM-02": "Sensitive Information Disclosure",
    "OWASP-LLM-03": "Supply Chain",
    "OWASP-LLM-04": "Data and Model Poisoning",
    "OWASP-LLM-05": "Improper Output Handling",
    "OWASP-LLM-06": "Excessive Agency",
    "OWASP-LLM-07": "System Prompt Leakage",
    "OWASP-LLM-08": "Vector and Embedding Weaknesses",
    "OWASP-LLM-09": "Misinformation",
    "OWASP-LLM-10": "Unbounded Consumption",
}

CHECK_OWASP_REFS: dict[str, list[str]] = {
    "prompt_injection": ["OWASP-LLM-01", "OWASP-LLM-07"],
    "unsafe_output": ["OWASP-LLM-02", "OWASP-LLM-05"],
    "toxicity": ["OWASP-LLM-05"],
    "pii_redaction": ["OWASP-LLM-02"],
    "prompt_leakage": ["OWASP-LLM-07"],
    "refusal_overblocking": ["OWASP-LLM-05"],
    "structured_output": ["OWASP-LLM-05"],
    "rate_limit": ["OWASP-LLM-10"],
    "latency": ["OWASP-LLM-10"],
    "token_budget": ["OWASP-LLM-10"],
    "rag_faithfulness": ["OWASP-LLM-09"],
    "rag_citation": ["OWASP-LLM-08", "OWASP-LLM-09"],
    "rag_poisoned_document": ["OWASP-LLM-04", "OWASP-LLM-08"],
    "rag_provenance": ["OWASP-LLM-03", "OWASP-LLM-08"],
    "rag_retrieval": ["OWASP-LLM-08", "OWASP-LLM-09"],
    "rag_security": ["OWASP-LLM-01", "OWASP-LLM-04", "OWASP-LLM-08"],
    "rag_stale_context": ["OWASP-LLM-08", "OWASP-LLM-09"],
    "rag_tenant_isolation": ["OWASP-LLM-02", "OWASP-LLM-08"],
}


def owasp_refs_for_check(check: str) -> list[str]:
    """Return OWASP LLM Top 10 2025 references associated with a check."""
    return list(CHECK_OWASP_REFS.get(check, ()))
