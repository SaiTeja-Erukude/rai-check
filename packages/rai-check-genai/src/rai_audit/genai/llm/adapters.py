from __future__ import annotations

import importlib
from collections.abc import Mapping
from time import perf_counter
from typing import Any

from rai_audit.genai.llm.models import LLMTestCase, ProviderResponse


class OpenAIResponder:
    """Callable OpenAI Responses API adapter with normalized operational metrics."""

    def __init__(
        self,
        model: str,
        *,
        client=None,
        pricing: Mapping[str, tuple[float, float]] | None = None,
        **request_options,
    ):
        self.model = model
        self.client = client or _require("openai", "openai").OpenAI()
        self.pricing = pricing or {}
        self.request_options = request_options

    def __call__(self, case: LLMTestCase) -> ProviderResponse:
        started = perf_counter()
        try:
            response = self.client.responses.create(
                model=self.model, input=case.prompt, **self.request_options
            )
        except Exception as exc:
            if _is_rate_limit(exc):
                return ProviderResponse(
                    "",
                    "openai",
                    self.model,
                    _elapsed(started),
                    rate_limited=True,
                    metadata={"error": str(exc)},
                )
            raise
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return ProviderResponse(
            text=str(getattr(response, "output_text", "")),
            provider="openai",
            model=self.model,
            latency_ms=_elapsed(started),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost(self.model, input_tokens, output_tokens, self.pricing),
        )


class AnthropicResponder:
    """Callable Anthropic Messages API adapter with normalized operational metrics."""

    def __init__(
        self,
        model: str,
        *,
        client=None,
        max_tokens: int = 1024,
        pricing: Mapping[str, tuple[float, float]] | None = None,
        **request_options,
    ):
        self.model = model
        self.client = client or _require("anthropic", "anthropic").Anthropic()
        self.max_tokens = max_tokens
        self.pricing = pricing or {}
        self.request_options = request_options

    def __call__(self, case: LLMTestCase) -> ProviderResponse:
        started = perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": case.prompt}],
                **self.request_options,
            )
        except Exception as exc:
            if _is_rate_limit(exc):
                return ProviderResponse(
                    "",
                    "anthropic",
                    self.model,
                    _elapsed(started),
                    rate_limited=True,
                    metadata={"error": str(exc)},
                )
            raise
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        text = "".join(
            str(getattr(block, "text", "")) for block in getattr(response, "content", ())
        )
        return ProviderResponse(
            text,
            "anthropic",
            self.model,
            _elapsed(started),
            input_tokens,
            output_tokens,
            _cost(self.model, input_tokens, output_tokens, self.pricing),
        )


def rubric_judge(responder, *, rubric: str, pass_score: float = 0.8):
    """Build a configurable JSON rubric judge from a provider responder."""
    import json

    def judge(case: LLMTestCase, response: str):
        prompt = (
            f"{rubric}\nReturn JSON with score from 0 to 1 and reasoning.\n"
            f"Question: {case.prompt}\nContexts: {[context.content for context in case.contexts]}\n"
            f"Response: {response}"
        )
        verdict = responder(LLMTestCase(id=f"judge-{case.id}", prompt=prompt, checks=()))
        text = verdict.text if isinstance(verdict, ProviderResponse) else verdict
        parsed = json.loads(text)
        parsed["passed"] = float(parsed["score"]) >= pass_score
        return parsed

    return judge


def _cost(model, input_tokens, output_tokens, pricing):
    rates = pricing.get(model)
    if rates is None:
        return None
    return round((input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000, 8)


def _elapsed(started):
    return (perf_counter() - started) * 1000


def _is_rate_limit(exc):
    return "rate" in type(exc).__name__.lower() and "limit" in type(exc).__name__.lower()


def _require(module_name: str, extra_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Install rai-check-genai[{extra_name}] to use this adapter.") from exc
