"""Durability: a run that dies mid-flight must not lose completed work.

A real discovery run died after the clusterer stage and lost everything,
because clusters persisted only once their group's generator returned and
pairwise judgements only once the whole tournament finished. These tests pin
the fix by actually failing a stage and asserting the earlier work survived.
"""

from __future__ import annotations

import asyncio

import pytest

from frontier_scout.evals.pairwise import PairwiseTournament
from frontier_scout.llm.runner import AgentRunner
from frontier_scout.models import CandidateVideo
from frontier_scout.pipeline.discovery import DiscoveryPipeline

from .conftest import FakeProvider
from .test_integration import _scripted


def _candidate(title: str) -> CandidateVideo:
    return CandidateVideo(
        working_title=title, central_question="q", one_sentence_premise="p",
        real_current_trigger="t", why_now="now", technical_mechanism="m",
        old_mental_model="old", new_mental_model="new",
        human_consequence="c", credible_disagreement="d",
    )


async def test_scout_signals_survive_a_later_stage_dying(db, settings, make_source_item):
    """Scouts finished; the clusterer then exploded. The signals must remain."""
    items = []
    for i in range(6):
        item = make_source_item(
            external_id=f"dur{i}",
            text="verification bottleneck and agent harness keep coming up here",
            author=f"author_{i}", publication=f"Pub {i}", days_ago=i,
        )
        db.insert_source_item(item)
        items.append(item)

    scripted = _scripted(items[0].id)
    scripted["clusterer"] = RuntimeError("clusterer died mid-run")

    pipeline = DiscoveryPipeline(
        db, settings, FakeProvider(scripted), run_id="dying", use_cache=False
    )
    await pipeline.run(top_n=1, coverage_blocks={})

    stored = db.conn.execute(
        "SELECT COUNT(*) c FROM signals WHERE run_id='dying'"
    ).fetchone()["c"]
    assert stored > 0, "scout signals must be persisted before the next stage runs"

    joins = db.conn.execute("SELECT COUNT(*) c FROM signal_sources").fetchone()["c"]
    assert joins > 0, "provenance joins must persist with the signals"


async def test_stage_checkpoints_record_where_a_run_reached(
    db, settings, make_source_item
):
    item = make_source_item(external_id="stg", text="agent harness discussion here",
                            author="a1")
    db.insert_source_item(item)
    for i in range(5):
        db.insert_source_item(
            make_source_item(external_id=f"stg{i}", text="agent harness talk",
                             author=f"a{i}", publication=f"P{i}", days_ago=i)
        )

    pipeline = DiscoveryPipeline(
        db, settings, FakeProvider(_scripted(item.id)), run_id="staged", use_cache=False
    )
    await pipeline.run(top_n=1, coverage_blocks={})

    stages = {r["stage"]: r["status"] for r in db.stages("staged")}
    assert stages.get("scouts") == "done"
    assert "cluster_and_generate" in stages


async def test_pairwise_judgements_persist_as_they_land(db, settings):
    """Hundreds of comparisons must not be lost to a death near the end."""
    saved: list = []
    candidates = [_candidate(f"Title {i}") for i in range(3)]
    provider = FakeProvider(
        {"pairwise": {"winner": "A", "confidence": 0.8, "explanation": "clearer"}}
    )
    runner = AgentRunner(provider, settings, db, "tourney", use_cache=False)

    result = await PairwiseTournament(runner).run(candidates, sink=saved.append)

    assert len(saved) == len(result.results) > 0
    assert saved[0].question_key


async def test_call_timeout_does_not_hang_the_run(settings, db):
    """A request that never returns must fail the call, not stall the run."""

    class Hanging(FakeProvider):
        async def complete(self, request):
            await asyncio.sleep(30)

    runner = AgentRunner(Hanging(), settings, db, "hang", use_cache=False)
    runner.call_timeout = 0.2

    from frontier_scout.llm.base import LLMRequest
    from frontier_scout.agents.scouts import TrendScoutOutput

    result = await asyncio.wait_for(
        runner.run(
            LLMRequest(agent="trend_scout", instructions="i", task="t",
                       schema=TrendScoutOutput, prompt_version="v1"),
            max_attempts=2,
        ),
        timeout=5,
    )
    assert result is None
    assert any("within" in f for f in runner.stats.failures)
