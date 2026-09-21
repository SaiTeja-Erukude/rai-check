# rai-check-genai

Generative AI audit package for LLM, RAG, and agentic AI systems.

It includes:

- LLM safety, refusal, structured-output, latency, token-budget, and provider checks
- RAG faithfulness, citation, provenance, retrieval, tenant-isolation, stale-context, and poisoned-document checks
- Agent trace checks for tool allowlists, permissions, memory use, handoffs, untrusted content, and prompt injection

## CLI

```bash
rai-check genai llm run \
  --suite packages/rai-check-genai/examples/llm_audit_suite.yml \
  --format html

rai-check genai agents run \
  --trace packages/rai-check-genai/examples/customer_support_trace.json \
  --allowed-tools lookup_order,refund_order \
  --format html
```

## Python API

```python
from rai_audit.genai import LLMAudit, load_test_suite

suite = load_test_suite("packages/rai-check-genai/examples/llm_audit_suite.yml")
report = LLMAudit(suite, persist=False).run()
report.to_html("llm_audit_report.html")
```

```python
from rai_audit.genai import AgentAudit, load_trace

trace = load_trace("packages/rai-check-genai/examples/customer_support_trace.json")
report = AgentAudit(trace, allowed_tools=["lookup_order"], persist=False).run()
report.to_html("agent_audit_report.html")
```

Framework adapters normalize traces from common agent runtimes:

```python
from rai_audit.genai import (
    adapt_autogen_messages,
    adapt_langgraph_events,
    adapt_openai_agents_trace,
    adapt_otel_spans,
)
```

See also:

- [OpenAI Agents SDK tracing](https://openai.github.io/openai-agents-python/tracing/)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
