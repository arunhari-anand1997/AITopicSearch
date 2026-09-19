"""Persistence, ingestion, scoring, tournament ranking and graceful degradation."""

from __future__ import annotations

import json

import pytest

from frontier_scout.db import Database
from frontier_scout.evals.pairwise import TournamentResult
from frontier_scout.evals.red_team import severity_penalty
from frontier_scout.ingest.base import canonicalize, extract_arxiv_ids, parse_date
from frontier_scout.ingest.files import ingest_file
from frontier_scout.models import (
    CandidateVideo,
    ConnectorStatus,
    Evaluation,
    HumanReview,
    PairwiseResult,
    RedTeamFinding,
)
from frontier_scout.pipeline.discovery import DiscoveryPipeline
from frontier_scout.trends.opportunity import binary_gate, score_candidate

WEIGHTS = {
    "insider_momentum": 0.20, "consequence": 0.25, "public_confusion": 0.20,
    "story_potential": 0.15, "technical_substance": 0.20,
}


# -- persistence -----------------------------------------------------------


def test_source_item_insert_is_idempotent(db, make_source_item):
    item = make_source_item(external_id="dup1", text="hello world models")
    assert db.insert_source_item(item) is True
    assert db.insert_source_item(item) is False, "re-inserting must not duplicate"
    assert db.conn.execute("SELECT COUNT(*) c FROM source_items").fetchone()["c"] == 1


def test_engagement_is_appended_as_a_series(db, make_source_item):
    """Acceleration needs at least two observations to exist."""
    item = make_source_item(external_id="eng1", text="text", score=10)
    db.insert_source_item(item)
    item.engagement.score = 55
    db.insert_source_item(item)

    rows = db.conn.execute(
        "SELECT score FROM engagement_snapshots WHERE source_item_id=? ORDER BY id",
        (item.id,),
    ).fetchall()
    assert [r["score"] for r in rows] == [10, 55]


def test_candidate_and_review_round_trip(db):
    cand = CandidateVideo(
        working_title="Who Checks the AI?", central_question="q",
        one_sentence_premise="p", real_current_trigger="t", why_now="now",
        technical_mechanism="verification", old_mental_model="old",
        new_mental_model="new", human_consequence="c",
        credible_disagreement="d", source_ids=[],
    )
    db.start_run("r1", "discover")
    db.save_candidate(cand, "r1")
    db.save_evaluation(
        Evaluation(candidate_id=cand.id, scores={"stakes": 8, "novelty": 6}), "r1"
    )
    db.save_red_team(RedTeamFinding(candidate_id=cand.id, attack_key="fake_stakes",
                                    verdict="WOUNDED", severity=6), "r1")
    db.save_human_review(
        HumanReview(candidate_id=cand.id, decision="APPROVE",
                    personal_excitation_score=9, would_spend_two_weeks=True),
        run_id="r1",
    )

    row = db.conn.execute(
        "SELECT c.working_title, e.stakes, h.personal_excitation_score,"
        " h.would_spend_two_weeks, r.verdict"
        " FROM candidates c"
        " JOIN evaluations e ON e.candidate_id=c.id"
        " JOIN human_reviews h ON h.candidate_id=c.id"
        " JOIN red_team r ON r.candidate_id=c.id WHERE c.id=?",
        (cand.id,),
    ).fetchone()
    assert row["working_title"] == "Who Checks the AI?"
    assert row["stakes"] == 8
    assert row["personal_excitation_score"] == 9
    assert row["would_spend_two_weeks"] == 1
    assert row["verdict"] == "WOUNDED"


def test_video_outcomes_table_exists_for_future_learning(db):
    """Schema must be ready to join published performance to predictions."""
    cols = {
        r["name"]
        for r in db.conn.execute("PRAGMA table_info(video_outcomes)").fetchall()
    }
    assert {"ctr", "views_30d", "avg_view_duration_s", "candidate_id"} <= cols


# -- ingestion helpers -----------------------------------------------------


def test_canonicalize_strips_tracking_so_shared_origins_match():
    a = canonicalize("https://WWW.Example.com/post/?utm_source=twitter&id=3")
    b = canonicalize("https://example.com/post?id=3")
    assert a == b


def test_extract_arxiv_ids():
    text = "see https://arxiv.org/abs/2501.12345 and arxiv.org/pdf/2402.00001"
    assert extract_arxiv_ids(text) == ["2402.00001", "2501.12345"]


def test_parse_date_handles_rss_and_iso():
    assert parse_date("Sat, 19 Sep 2026 15:42:20 GMT") is not None
    assert parse_date("2026-09-19T15:42:20Z") is not None
    assert parse_date("nonsense") is None


def test_file_ingest_jsonl(tmp_path, db):
    path = tmp_path / "items.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"external_id": "m1", "text": "a claim about world models",
                 "source_type": "x", "author": "someone", "likes": 12},
                {"external_id": "m2", "title": "essay", "text": "another",
                 "source_type": "substack"},
            ]
        )
    )
    result = ingest_file(path)
    assert result.status == ConnectorStatus.OK
    assert len(result.items) == 2
    assert result.items[0].source_type == "x"
    assert result.items[0].engagement.likes == 12


def test_missing_file_reports_unavailable_not_crash(tmp_path):
    result = ingest_file(tmp_path / "nope.jsonl")
    assert result.status == ConnectorStatus.SOURCE_UNAVAILABLE
    assert not result.ok


# -- scoring ---------------------------------------------------------------


def test_geometric_mean_collapses_on_a_weak_dimension():
    strong = score_candidate(metrics=None, evaluation=None, saturation=None, weights=WEIGHTS)
    assert strong.value == 0.0  # nothing to score

    from frontier_scout.trends.opportunity import weighted_geometric_mean

    balanced = weighted_geometric_mean({k: 0.8 for k in WEIGHTS}, WEIGHTS)
    crippled = weighted_geometric_mean({**{k: 0.8 for k in WEIGHTS}, "consequence": 0.02},
                                       WEIGHTS)
    assert crippled < balanced / 1.5


def test_unknown_coverage_applies_no_adjustment_and_says_so():
    ev = Evaluation(scores={"stakes": 7, "human_relevance": 7})
    score = score_candidate(metrics=None, evaluation=ev, saturation=None, weights=WEIGHTS)
    assert score.coverage_adjustment == 1.0
    assert score.coverage_known is False
    assert any("coverage unknown" in n for n in score.notes)


def test_known_saturation_penalises():
    ev = Evaluation(scores={"stakes": 7, "human_relevance": 7})
    covered = score_candidate(metrics=None, evaluation=ev, saturation=1.0, weights=WEIGHTS)
    assert covered.coverage_adjustment < 0.5


def test_binary_gate_rejects_jargon_dependent_candidates():
    ev = Evaluation(cares_before_knowing_term=False, interests_non_ai_person=True)
    passed, note = binary_gate(ev)
    assert passed is False and "insider term" in note


def test_red_team_fatal_finding_dominates():
    fatal = severity_penalty([RedTeamFinding(attack_key="a", verdict="FATAL", severity=9)])
    wounded = severity_penalty([RedTeamFinding(attack_key="a", verdict="WOUNDED", severity=5)])
    clean = severity_penalty([RedTeamFinding(attack_key="a", verdict="SURVIVES", severity=1)])
    assert fatal < wounded < clean == 1.0


# -- tournament ------------------------------------------------------------


def test_tournament_ranking_is_by_win_rate():
    t = TournamentResult()
    for winner, loser in [("a", "b"), ("a", "c"), ("b", "c")]:
        for cid in (winner, loser):
            t.played[cid] += 1
        t.wins[winner] += 1

    ranking = t.ranking()
    assert ranking[0][0] == "a"
    assert ranking[0][1] == 1.0
    assert ranking[-1][0] == "c"
    assert t.record_for("a") == "2.0/2"


def test_pairwise_rows_are_stored_per_question(db):
    db.start_run("r2", "discover")
    for question in ("publish_next_week", "share_with_friend", "mission_fit"):
        db.save_pairwise(
            PairwiseResult(candidate_a="a", candidate_b="b", question_key=question,
                           winner="a", confidence=0.7),
            "r2",
        )
    rows = db.conn.execute(
        "SELECT DISTINCT question_key FROM pairwise_results WHERE run_id='r2'"
    ).fetchall()
    assert len(rows) == 3, "each question must be judged and stored separately"


# -- graceful degradation --------------------------------------------------


async def test_pipeline_without_llm_still_produces_metrics(db, settings, make_source_item):
    """No API key must degrade to the quantitative layer, not crash."""
    for i in range(6):
        db.insert_source_item(
            make_source_item(
                external_id=f"deg{i}",
                text="world models and test time compute are discussed widely",
                author=f"author_{i}", publication=f"Pub {i}", days_ago=i,
            )
        )
    pipeline = DiscoveryPipeline(db, settings, provider=None, run_id="degraded")
    result = await pipeline.run()

    assert result.llm_available is False
    assert result.metrics, "momentum table must still be produced"
    assert result.candidates == []
    assert any("ANTHROPIC_API_KEY" in n for n in result.notes)
