"""Anthropic provider.

Two API facts shape this file and are worth stating plainly, because both
contradict what the original design assumed:

1. `temperature` is REJECTED with a 400 on current models (Opus 5, Sonnet 5,
   Opus 4.7/4.8, Fable). The "low temperature for extraction, higher for
   generation" plan cannot be implemented with a sampling parameter any more.
   `output_config.effort` replaces it as the quality/spend lever, so this
   provider maps each pipeline role to an effort level and only sends
   `temperature` to models that still accept it.

2. Prompt caching is a prefix match. The eight scouts all reason over the same
   source dossier for a given cluster, so the dossier goes in the FIRST system
   block with `cache_control`, and the agent-specific instructions follow it.
   Ordered the other way round, every scout would pay full price for the same
   text.
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import Settings
from .base import LLMProvider, LLMRequest, LLMResponse, LLMUnavailable

# Per-million-token pricing, used for the run cost report.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
}
DEFAULT_PRICING = (5.00, 25.00)

# Models that still accept sampling parameters. Everything current rejects them
# with a 400, so the default is to omit temperature entirely.
SAMPLING_MODELS = frozenset({"claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5"})

# Effort replaces temperature as the per-role lever. Extraction and judgement
# run cheaper; generation and clustering get room to think.
DEFAULT_EFFORT: dict[str, str] = {
    "scout": "medium",
    "clusterer": "high",
    "translator": "high",
    "generator": "high",
    "evaluator": "medium",
    "pairwise": "medium",
    "red_team": "medium",
}

DEFAULT_MODELS: dict[str, str] = {
    "scout": "claude-sonnet-5",
    "clusterer": "claude-opus-5",
    "translator": "claude-opus-5",
    "generator": "claude-opus-5",
    "evaluator": "claude-sonnet-5",
    "pairwise": "claude-sonnet-5",
    "red_team": "claude-sonnet-5",
}


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.anthropic_api_key:
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not set. The collection and trend layers "
                "still run without it; only the agent layer needs a key."
            )
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise LLMUnavailable(f"anthropic SDK not installed: {exc}") from exc

        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=3)
        self._models = {**DEFAULT_MODELS, **(settings.models or {})}
        self._effort = {**DEFAULT_EFFORT, **(settings.section("effort", {}) or {})}

    def model_for(self, role: str) -> str:
        return self._models.get(role, DEFAULT_MODELS.get(role, "claude-sonnet-5"))

    def effort_for(self, role: str) -> str:
        return self._effort.get(role, "medium")

    async def complete(self, request: LLMRequest) -> LLMResponse:
        model = self.model_for(request.agent_role())
        started = time.monotonic()

        system_blocks: list[dict[str, Any]] = []
        if request.shared_context:
            # Shared first, and cached: all scouts on a cluster reuse this.
            system_blocks.append(
                {
                    "type": "text",
                    "text": request.shared_context,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        system_blocks.append({"type": "text", "text": request.instructions})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "system": system_blocks,
            "messages": [{"role": "user", "content": request.task}],
            "output_config": {
                "effort": request.effort,
                "format": {
                    "type": "json_schema",
                    "schema": _strict_schema(request.schema),
                },
            },
        }
        # Only models that still support sampling get a temperature; the rest
        # would return a 400.
        if request.temperature is not None and model in SAMPLING_MODELS:
            kwargs["temperature"] = request.temperature

        response = await self.client.messages.create(**kwargs)
        latency_ms = int((time.monotonic() - started) * 1000)

        # A safety decline is a real outcome, not a parse failure.
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            raise LLMRefusal(
                f"{model} declined this request"
                + (f" (category: {detail.category})" if detail else "")
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedOutput(f"{model} returned unparseable JSON: {exc}") from exc

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return LLMResponse(
            data=data,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=estimate_cost(model, usage.input_tokens, usage.output_tokens,
                                   cache_read, cache_write),
            latency_ms=latency_ms,
            stop_reason=response.stop_reason,
            raw_text=text,
        )


class MalformedOutput(ValueError):
    """The model returned something that is not valid JSON."""


class LLMRefusal(RuntimeError):
    """The model declined the request on safety grounds."""


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Estimate USD for one call.

    Cached reads bill at roughly a tenth of input, cache writes at ~1.25x.
    """
    in_rate, out_rate = PRICING.get(model, DEFAULT_PRICING)
    return (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_read * in_rate * 0.1
        + cache_write * in_rate * 1.25
    ) / 1_000_000


def _strict_schema(model: type) -> dict[str, Any]:
    """Build a strict JSON Schema from a Pydantic model.

    Structured outputs require `additionalProperties: false` throughout and no
    unresolved `$ref`s, neither of which Pydantic emits by default.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    resolved = _inline_refs(schema, defs)
    return _harden(resolved)


def _inline_refs(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    if depth > 12:  # guard against a self-referential model
        return {"type": "object"}
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            target = defs.get(name, {})
            merged = {k: v for k, v in node.items() if k != "$ref"}
            return _inline_refs({**target, **merged}, defs, depth + 1)
        return {k: _inline_refs(v, defs, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs, depth + 1) for v in node]
    return node


def _harden(node: Any) -> Any:
    """Force additionalProperties:false and make every property required."""
    if isinstance(node, dict):
        out = {k: _harden(v) for k, v in node.items()}
        if out.get("type") == "object" or "properties" in out:
            out.setdefault("additionalProperties", False)
            props = out.get("properties")
            if isinstance(props, dict):
                # Structured outputs require every property in `required`;
                # optionality is expressed by allowing null in the type.
                out["required"] = list(props.keys())
        return out
    if isinstance(node, list):
        return [_harden(v) for v in node]
    return node
