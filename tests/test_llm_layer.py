"""Caching, cost tracking, retries and the citation guarantee."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from frontier_scout.llm.anthropic_provider import _strict_schema, estimate_cost
from frontier_scout.llm.base import LLMRequest
from frontier_scout.llm.cache import CallCache
from frontier_scout.llm.runner import AgentRunner, BudgetExhausted
from frontier_scout.models import Claim, ClaimKind

from .conftest import FakeProvider


class Findings(BaseModel):
    label: str
    evidence: list[Claim] = Field(default_factory=list)


def _request(agent: str = "trend_scout", task: str = "go") -> LLMRequest:
    return LLMRequest(
        agent=agent, instructions="be useful", task=task,
        schema=Findings, prompt_version="v1",
    )


async def test_cache_prevents_a_second_call(settings, db):
    provider = FakeProvider({"trend_scout": {"label": "x", "evidence": []}})
    runner = AgentRunner(provider, settings, db, "run1")

    first = await runner.run(_request())
    second = await runner.run(_request())

    assert first.label == "x" and second.label == "x"
    assert len(provider.calls) == 1, "identical input must be served from cache"
    assert runner.stats.cache_hits == 1


def test_cache_key_changes_with_prompt_version():
    a = _request()
    b = _request()
    b.prompt_version = "v2"
    assert a.cache_key() != b.cache_key()


def test_cache_survives_corrupt_entries(tmp_path):
    cache = CallCache(tmp_path / "c")
    cache.put("abc123", {"data": {"ok": True}})
    path = cache._path("abc123")
    path.write_text("{ not json")
    assert cache.get("abc123") is None


async def test_malformed_output_is_retried_then_succeeds(settings, db):
    provider = FakeProvider({
        "trend_scout": [
            {"wrong_field": 1},                      # fails schema
            {"label": "recovered", "evidence": []},  # repaired
        ]
    })
    runner = AgentRunner(provider, settings, db, "run2")
    result = await runner.run(_request())

    assert result is not None and result.label == "recovered"
    assert runner.stats.retries == 1


async def test_persistent_malformed_output_returns_none(settings, db):
    provider = FakeProvider({"trend_scout": [{"bad": 1}, {"bad": 2}, {"bad": 3}]})
    runner = AgentRunner(provider, settings, db, "run3")

    assert await runner.run(_request()) is None
    assert runner.stats.failures


async def test_fabricated_citation_triggers_one_repair(settings, db):
    """The provenance guarantee, enforced mechanically."""
    provider = FakeProvider({
        "trend_scout": [
            {"label": "a", "evidence": [
                {"text": "claimed", "kind": "SOURCE_FACT",
                 "source_ids": ["ghost_999"], "confidence": 0.8}
            ]},
            {"label": "a", "evidence": [
                {"text": "claimed", "kind": "SOURCE_FACT",
                 "source_ids": ["real_1"], "confidence": 0.8}
            ]},
        ]
    })
    runner = AgentRunner(provider, settings, db, "run4")
    result = await runner.run(_request(), allowed_source_ids={"real_1"})

    assert result.evidence[0].source_ids == ["real_1"]
    assert runner.stats.citation_repairs == 1
    assert "not provided to you" in provider.calls[1].task


async def test_unrepaired_citation_is_dropped_and_demoted(settings, db):
    """Never publish a fabricated citation, even after retries are spent."""
    bad = {"label": "a", "evidence": [
        {"text": "claimed", "kind": "SOURCE_FACT",
         "source_ids": ["ghost_1"], "confidence": 0.9}
    ]}
    provider = FakeProvider({"trend_scout": [bad, dict(bad), dict(bad)]})
    runner = AgentRunner(provider, settings, db, "run5")
    result = await runner.run(_request(), allowed_source_ids={"real_1"})

    claim = result.evidence[0]
    assert claim.source_ids == []
    assert claim.kind == ClaimKind.INFERENCE, "an unsourced fact must be demoted"
    assert claim.confidence <= 0.3
    assert runner.stats.dropped_claims == 1


async def test_budget_ceiling_stops_spending(settings, db):
    provider = FakeProvider({"trend_scout": {"label": "x", "evidence": []}})
    runner = AgentRunner(provider, settings, db, "run6")
    runner.budget_usd = 0.0005          # below one call's cost
    runner.stats.spend_usd = 0.001      # already over

    with pytest.raises(BudgetExhausted):
        await runner.run(_request(task="fresh task"))


async def test_costs_are_recorded_to_the_database(settings, db):
    provider = FakeProvider({"trend_scout": {"label": "x", "evidence": []}})
    runner = AgentRunner(provider, settings, db, "run7")
    await runner.run(_request(task="unique task for cost test"))

    report = db.run_cost("run7")
    assert report["n"] == 1
    assert report["usd"] > 0
    assert runner.cost_report()["estimated_cost_usd"] > 0


def test_cost_estimate_rewards_cache_reads():
    full = estimate_cost("claude-opus-5", 50_000, 5_000)
    cached = estimate_cost("claude-opus-5", 5_000, 5_000, cache_read=45_000)
    assert cached < full / 2


def test_strict_schema_is_structured_output_ready():
    schema = _strict_schema(Findings)
    assert schema["additionalProperties"] is False
    assert "$defs" not in schema
    assert set(schema["required"]) == {"label", "evidence"}
    nested = schema["properties"]["evidence"]["items"]
    assert nested["additionalProperties"] is False, "nested objects must be strict too"
