from types import SimpleNamespace

from rai_audit.genai.llm.adapters import OpenAIResponder
from rai_audit.genai.llm.models import LLMTestCase


def test_openai_adapter_normalizes_metrics_and_cost():
    response = SimpleNamespace(
        output_text="Hello",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
    responder = OpenAIResponder("test-model", client=client, pricing={"test-model": (1, 2)})

    result = responder(LLMTestCase(id="test", prompt="Hello", checks=()))

    assert result.text == "Hello"
    assert result.total_tokens == 15
    assert result.cost_usd == 0.00002
