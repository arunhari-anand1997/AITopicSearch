"""DeepSeek provider.

The second provider, and the reason the LLM layer was abstracted in the first
place. Three differences from Anthropic shape this file, all verified against
the live API rather than assumed:

1. NO SERVER-SIDE SCHEMA ENFORCEMENT. `response_format: {"type": "json_schema"}`
   returns 400 ("This response_format type is unavailable now"); only
   `{"type": "json_object"}` works. So the schema is injected into the prompt
   and validated client-side, which makes the runner's repair loop load-bearing
   rather than a safety net. It also requires the word "json" to appear in the
   prompt, and the docs warn that content is occasionally returned EMPTY -- both
   handled below.

2. TEMPERATURE WORKS. Unlike current Claude models, which reject sampling
   parameters with a 400, DeepSeek is OpenAI-compatible and accepts
   `temperature`. The `legacy_temperature` block in config/editorial.yaml
   becomes live again here, so the original "low for extraction, higher for
   generation" plan is expressible after all.

3. CACHING IS AUTOMATIC. There is no `cache_control`; DeepSeek caches common
   prefixes on its own and reports `prompt_cache_hit_tokens`. Putting the shared
   source dossier first still pays -- it just happens without being asked for.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..config import Settings
from .base import LLMProvider, LLMRequest, LLMResponse, LLMUnavailable

BASE_URL = "https://api.deepseek.com"

#: Peak per-million-token rates. Peak (the higher of the published peak/off-peak
#: pair) is used deliberately: a budget ceiling should never be optimistic.
#: (cache_miss_input, output, cache_hit_input)
PRICING: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.30, 1.20, 0.006),
    "deepseek-v4-pro": (1.32, 3.96, 0.044),
}
DEFAULT_PRICING = (1.32, 3.96, 0.044)

#: Cheap model for high-volume extraction, capable model for judgement.
DEFAULT_MODELS: dict[str, str] = {
    "scout": "deepseek-flash",
    "clusterer": "deepseek-v4-pro",
    "translator": "deepseek-v4-pro",
    "generator": "deepseek-v4-pro",
    "evaluator": "deepseek-v4-pro",
    "pairwise": "deepseek-flash",
    "red_team": "deepseek-v4-pro",
    "extractor": "deepseek-flash",
}

DEFAULT_TEMPERATURE: dict[str, float] = {
    "scout": 0.2,
    "clusterer": 0.3,
    "translator": 0.8,
    "generator": 0.9,
    "evaluator": 0.0,
    "pairwise": 0.0,
    "red_team": 0.2,
    "extractor": 0.0,
}


class DeepSeekProvider(LLMProvider):
    name = "deepseek"

    def __init__(self, settings: Settings):
        self.settings = settings
        key = settings.deepseek_api_key
        if not key:
            raise LLMUnavailable(
                "DEEPSEEK_API_KEY is not set. The collection and trend layers "
                "still run without it; only the agent layer needs a key."
            )
        self.api_key = key
        configured = settings.section("deepseek_models", {}) or {}
        self._models = {**DEFAULT_MODELS, **configured}
        self._temperature = {
            **DEFAULT_TEMPERATURE,
            **(settings.section("legacy_temperature", {}) or {}),
        }
        self._client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(300.0, connect=30.0),
        )

    def model_for(self, role: str) -> str:
        return self._models.get(role, DEFAULT_MODELS.get(role, "deepseek-flash"))

    def effort_for(self, role: str) -> str:
        """DeepSeek has no effort parameter; temperature is the lever instead."""
        return "medium"

    def temperature_for(self, role: str) -> float:
        return float(self._temperature.get(role, 0.3))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        role = request.agent_role()
        model = self.model_for(role)
        started = time.monotonic()

        # Shared context FIRST so DeepSeek's automatic prefix caching can reuse
        # it across the eight scouts that read the same dossier.
        system_parts = []
        if request.shared_context:
            system_parts.append(request.shared_context)
        system_parts.append(request.instructions)
        system_parts.append(_schema_instruction(request.schema))

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "\n\n".join(system_parts)},
                {"role": "user", "content": f"{request.task}\n\nRespond with JSON only."},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": min(request.max_tokens, 32000),
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self.temperature_for(role)
            ),
            "stream": False,
        }

        response = await self._client.post("/chat/completions", json=payload)
        if response.status_code != 200:
            raise DeepSeekError(
                f"{model} returned HTTP {response.status_code}: {response.text[:300]}"
            )
        body = response.json()
        latency_ms = int((time.monotonic() - started) * 1000)

        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason")

        # Documented DeepSeek behaviour: json_object mode occasionally returns
        # empty content. Raise so the runner retries rather than letting an
        # empty dict validate as a silently empty result.
        if not text.strip():
            raise EmptyCompletion(
                f"{model} returned empty content (finish_reason={finish}); retrying"
            )
        if finish == "length":
            raise MalformedOutput(
                f"{model} hit the token limit before closing its JSON "
                f"(finish_reason=length); retrying"
            )

        try:
            data = json.loads(_strip_fences(text))
        except json.JSONDecodeError as exc:
            raise MalformedOutput(f"{model} returned unparseable JSON: {exc}") from exc

        usage = body.get("usage") or {}
        cache_hit = int(usage.get("prompt_cache_hit_tokens") or 0)
        cache_miss = int(
            usage.get("prompt_cache_miss_tokens")
            or max(0, int(usage.get("prompt_tokens") or 0) - cache_hit)
        )
        output_tokens = int(usage.get("completion_tokens") or 0)

        return LLMResponse(
            data=data,
            model=model,
            input_tokens=cache_miss,
            output_tokens=output_tokens,
            cache_read_tokens=cache_hit,
            cost_usd=estimate_cost(model, cache_miss, output_tokens, cache_hit),
            latency_ms=latency_ms,
            stop_reason=finish,
            raw_text=text,
        )


class DeepSeekError(RuntimeError):
    """Non-200 from the API."""


class EmptyCompletion(RuntimeError):
    """json_object mode returned empty content; documented and retryable."""


class MalformedOutput(ValueError):
    """The model returned something that is not valid JSON."""


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cache_hit_tokens: int = 0
) -> float:
    """Estimate USD, using peak rates so a budget is never optimistic."""
    in_rate, out_rate, cache_rate = PRICING.get(model, DEFAULT_PRICING)
    return (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_hit_tokens * cache_rate
    ) / 1_000_000


def _strip_fences(text: str) -> str:
    """Remove ```json fences some models add despite json_object mode."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _schema_instruction(schema: type) -> str:
    """Put the schema in the prompt, since the server will not enforce it.

    The literal word "json" is required by DeepSeek's json_object mode, and is
    present here by construction.
    """
    rendered = json.dumps(schema.model_json_schema(), indent=2)
    return (
        "# OUTPUT FORMAT\n\n"
        "Return a single json object conforming EXACTLY to this JSON Schema. "
        "Include every required field. Use null for an unknown optional value "
        "and [] for an empty list. Emit no prose, no markdown and no code "
        "fences -- json only.\n\n"
        f"```\n{rendered}\n```"
    )
