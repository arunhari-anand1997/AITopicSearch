"""DeepSeek provider specifics.

Covers the three ways it differs from Anthropic, each verified against the
live API before being encoded here:
  - no server-side schema enforcement, so the schema goes in the prompt
  - temperature is accepted (and is therefore the real lever)
  - caching is automatic and reported as prompt_cache_hit_tokens
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, Field

from frontier_scout.llm.base import LLMRequest, LLMUnavailable
from frontier_scout.llm.deepseek_provider import (
    PRICING,
    DeepSeekError,
    DeepSeekProvider,
    EmptyCompletion,
    MalformedOutput,
    _schema_instruction,
    _strip_fences,
    estimate_cost,
)
from frontier_scout.models import Claim


class Out(BaseModel):
    label: str
    evidence: list[Claim] = Field(default_factory=list)


def _provider(settings, handler) -> DeepSeekProvider:
    settings.deepseek_api_key = "sk-test"
    provider = DeepSeekProvider(settings)
    provider._client = httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        transport=httpx.MockTransport(handler),
    )
    return provider


def _body(content: str, *, finish: str = "stop", usage: dict | None = None) -> dict:
    return {
        "model": "deepseek-flash",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": usage or {
            "prompt_tokens": 100, "completion_tokens": 20,
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 100,
        },
    }


def _request(**kw) -> LLMRequest:
    return LLMRequest(
        agent="trend_scout", instructions="be useful", task="go",
        schema=Out, prompt_version="v1", **kw,
    )


def test_missing_key_is_unavailable_not_a_crash(settings):
    settings.deepseek_api_key = None
    with pytest.raises(LLMUnavailable):
        DeepSeekProvider(settings)


async def test_schema_is_injected_into_the_prompt(settings):
    """DeepSeek cannot enforce a schema server-side, so it must be in the text."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_body('{"label": "x", "evidence": []}'))

    provider = _provider(settings, handler)
    await provider.complete(_request())
    await provider.aclose()

    system = captured["messages"][0]["content"]
    assert "JSON Schema" in system
    assert '"label"' in system
    assert captured["response_format"] == {"type": "json_object"}
    # json_object mode requires the literal word "json" in the prompt.
    assert "json" in system.lower()


async def test_shared_context_comes_first_for_prefix_caching(settings):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_body('{"label": "x", "evidence": []}'))

    provider = _provider(settings, handler)
    await provider.complete(_request(shared_context="DOSSIER-MARKER"))
    await provider.aclose()

    system = captured["messages"][0]["content"]
    assert system.startswith("DOSSIER-MARKER"), "shared prefix must lead"


async def test_temperature_is_sent_per_role(settings):
    """Unlike current Claude models, DeepSeek accepts sampling parameters."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_body('{"label": "x", "evidence": []}'))

    provider = _provider(settings, handler)
    await provider.complete(_request(metadata={"role": "generator"}))
    await provider.aclose()

    assert captured["temperature"] == pytest.approx(0.9)


async def test_empty_content_raises_so_the_runner_retries(settings):
    """A documented DeepSeek quirk; it must not validate as an empty result."""
    provider = _provider(settings, lambda r: httpx.Response(200, json=_body("   ")))
    with pytest.raises(EmptyCompletion):
        await provider.complete(_request())
    await provider.aclose()


async def test_truncated_output_is_reported_as_malformed(settings):
    provider = _provider(
        settings, lambda r: httpx.Response(200, json=_body('{"label": "x"', finish="length"))
    )
    with pytest.raises(MalformedOutput):
        await provider.complete(_request())
    await provider.aclose()


async def test_http_error_surfaces_with_context(settings):
    provider = _provider(
        settings, lambda r: httpx.Response(400, text="response_format unavailable")
    )
    with pytest.raises(DeepSeekError, match="400"):
        await provider.complete(_request())
    await provider.aclose()


async def test_code_fences_are_stripped(settings):
    provider = _provider(
        settings,
        lambda r: httpx.Response(
            200, json=_body('```json\n{"label": "fenced", "evidence": []}\n```')
        ),
    )
    result = await provider.complete(_request())
    await provider.aclose()
    assert result.data["label"] == "fenced"


def test_strip_fences_leaves_plain_json_alone():
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


async def test_cache_hits_are_counted_and_priced_cheaply(settings):
    usage = {
        "prompt_tokens": 10_000, "completion_tokens": 500,
        "prompt_cache_hit_tokens": 9_000, "prompt_cache_miss_tokens": 1_000,
    }
    provider = _provider(
        settings,
        lambda r: httpx.Response(200, json=_body('{"label":"x","evidence":[]}', usage=usage)),
    )
    result = await provider.complete(_request())
    await provider.aclose()

    assert result.cache_read_tokens == 9_000
    assert result.input_tokens == 1_000, "billed input is the cache MISS portion"
    uncached = estimate_cost("deepseek-flash", 10_000, 500, 0)
    assert result.cost_usd < uncached


def test_pricing_uses_peak_rates_so_budgets_are_not_optimistic():
    flash_in, flash_out, flash_cache = PRICING["deepseek-flash"]
    pro_in, pro_out, _ = PRICING["deepseek-v4-pro"]
    assert flash_cache < flash_in < flash_out
    assert pro_in > flash_in and pro_out > flash_out


def test_deepseek_is_far_cheaper_than_the_anthropic_default():
    """The reason a second provider was worth wiring up."""
    from frontier_scout.llm.anthropic_provider import estimate_cost as claude_cost

    ds = estimate_cost("deepseek-v4-pro", 50_000, 5_000)
    claude = claude_cost("claude-opus-5", 50_000, 5_000)
    assert ds < claude / 4


def test_schema_instruction_is_valid_json_schema():
    text = _schema_instruction(Out)
    payload = text.split("```")[1].strip()
    schema = json.loads(payload)
    assert "label" in schema["properties"]
