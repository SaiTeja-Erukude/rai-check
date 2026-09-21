import rai_audit.genai.agents as agents_pkg


def test_agents_public_api():
    assert "AgentAudit" in agents_pkg.__all__
    assert "AgentTrace" in agents_pkg.__all__
    assert "TraceEvent" in agents_pkg.__all__
    assert "adapt_langgraph_events" in agents_pkg.__all__
    assert "adapt_openai_agents_trace" in agents_pkg.__all__
    assert "adapt_autogen_messages" in agents_pkg.__all__
