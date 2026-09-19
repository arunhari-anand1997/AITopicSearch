"""Full pipeline integration, driven by a scripted provider.

Proves the whole chain runs: scouts -> clusterer -> translator -> generator ->
evaluator -> pairwise -> red team -> finalists -> brief, with provenance held
all the way through. No network, no API key, deterministic.
"""

from __future__ import annotations

import json

from frontier_scout.pipeline.brief import write_daily_brief, write_run_summary
from frontier_scout.pipeline.discovery import DiscoveryPipeline, write_artifacts

from .conftest import FakeProvider


def _scripted(source_id: str) -> dict:
    """Responses for every agent the pipeline invokes."""
    cite = [{"text": "writers report this", "kind": "SOURCE_FACT",
             "source_ids": [source_id], "confidence": 0.8}]
    candidate = {
        "working_title": "Who Checks the AI?",
        "alternative_titles": ["The End of Prompting"],
        "central_question": "If machines make more than we can review, what happens to checking?",
        "one_sentence_premise": "Verification is becoming scarcer than generation.",
        "real_current_trigger": "several newsletters argued this independently this month",
        "why_now": "the argument appeared across unrelated publications in weeks",
        "viewer_problem_solved": "why AI output still needs people",
        "technical_mechanism": "verification: how a system knows its output is right",
        "old_mental_model": "AI is a thing you ask",
        "new_mental_model": "AI is a process you supervise",
        "technical_concepts_to_teach": ["verification", "agent loops"],
        "stakes": "who is accountable for unreviewed machine work",
        "human_consequence": "the reviewable unit of work changes for ordinary jobs",
        "economic_consequence": "review capacity becomes the constraint",
        "societal_consequence": None,
        "human_angle": "the people who now check instead of make",
        "credible_disagreement": "builders say scaffolding is the gain; skeptics say it is oversold",
        "story_engine": "a race between making and checking",
        "people_orgs": ["Interconnects"],
        "evidence_available": "newsletters, HN threads, one arXiv paper",
        "coverage_gap": "unknown; YouTube not configured",
        "visual_opportunities": ["a queue of unreviewed output"],
        "strongest_counterargument": "this may be an essayists' argument, not a measured result",
        "what_we_need_to_investigate": ["is there a measured verification cost?"],
        "possible_ending_insight": "progress may hinge on cheap checking, not clever generating",
        "source_ids": [source_id],
    }
    return {
        "trend_scout": {
            "observations": [{
                "label": "verification", "movement_kind": "NEW_IDEA",
                "what_is_moving": "verification talk", "underlying_movement":
                "the constraint is moving from making to checking",
                "older_name": None, "sense_shift_note": None,
                "evidence": cite, "confidence": 0.7,
            }],
            "decaying_terms": ["loop engineering"], "overall_read": "real movement",
        },
        "frontier_scout": {
            "changes": [{
                "what_changed": "agents run long enough that scaffolding dominates",
                "change_kind": "workflow",
                "is_real_change_or_just_language": "REAL_CHANGE",
                "reasoning": "multiple independent reports", "evidence": cite,
                "what_evidence_would_settle_it": "a measured verification cost",
                "confidence": 0.6,
            }],
            "summary": "workflow shift",
        },
        "disagreement_scout": {
            "disagreements": [{
                "question_at_issue": "is scaffolding where the gains are?",
                "views": [
                    {"label": "A", "position": "scaffolding is the gain",
                     "proponents": ["builders"], "source_ids": [source_id],
                     "assumptions": ["models are good enough"],
                     "supporting_evidence": ["shipping systems"],
                     "what_would_weaken_it": "a model jump that removes the need"},
                    {"label": "B", "position": "oversold",
                     "proponents": ["Interconnects"], "source_ids": [source_id],
                     "assumptions": ["the loop is not the bottleneck"],
                     "supporting_evidence": ["skeptical essay"],
                     "what_would_weaken_it": "measured throughput gains"},
                ],
                "is_empirical_or_values": "EMPIRICAL",
                "why_it_matters_publicly": "it decides where effort goes",
                "confidence": 0.7,
            }],
            "consensus_note": "",
        },
        "human_consequence_scout": {
            "consequences": [{
                "domain": "work", "who_is_affected": "knowledge workers",
                "concrete_change": "people check machine output instead of producing it",
                "conditional_on": "if verification stays expensive",
                "time_horizon": "1-2_years", "is_second_order": True,
                "evidence": cite, "confidence": 0.6,
            }],
            "strongest_single_consequence": "review becomes the job",
        },
        "mechanism_scout": {
            "primary_mechanism": {
                "name": "verification",
                "plain_explanation": "how a system knows whether its answer is right",
                "why_it_matters_here": "it is the actual constraint",
                "analogy_candidates": ["proofreading"],
                "where_the_analogy_breaks": "proofreading does not scale with volume",
                "common_misconception": "that generating and checking cost the same",
                "difficulty": "moderate",
            },
            "supporting_mechanisms": [],
            "the_real_concept_note": "people argue about harnesses but need verification",
        },
        "capital_scout": {"signals": [], "note": "no clear capital signal in these sources"},
        "story_scout": {
            "possibilities": [{
                "story": "a skeptic publishes against the consensus",
                "story_kind": "bet_against_consensus",
                "who_is_involved": ["Interconnects"],
                "what_is_in_motion": "the consensus is being contested in public",
                "visual_potential": "side-by-side argument",
                "evidence": cite,
            }],
            "is_currently_only_an_argument": False, "research_needed": [],
        },
        "coverage_scout": {
            "coverage_data_available": False, "assessments": [],
            "general_audience_saturation": None,
            "gap_description": "no YouTube data supplied; gap unknown",
            "demand_signal_note": "",
        },
        "clusterer": {
            "cluster_name": "checking beats making",
            "underlying_change": "the scarce thing is shifting from producing output to confirming it",
            "raw_terms": ["verification", "agent harness"],
            "excluded_terms": ["loop engineering"],
            "old_mental_model": "prompting is the skill",
            "evidence": cite, "disagreements": [],
            "potential_public_relevance": "it changes what work looks like",
            "breadth_warning": False,
        },
        "translator": {
            "translations": [{
                "insider_term": "verification bottleneck",
                "mechanism": "checking output is harder than making it",
                "consequence": "people supervise instead of produce",
                "public_questions": ["Who Checks the AI?"],
            }],
            "best_public_questions": ["Who Checks the AI?", "The End of Prompting"],
        },
        "candidate_generator": {"candidates": [candidate, {**candidate,
                                "working_title": "The End of Prompting",
                                "central_question": "Are we done instructing machines step by step?"}]},
        "evaluator": {
            "buzz_timeliness": 7, "public_curiosity": 8, "stakes": 8, "novelty": 6,
            "technical_depth": 7, "accessibility": 7, "human_relevance": 8,
            "serious_disagreement": 7, "story_motion": 5, "visual_potential": 5,
            "evidence_availability": 6, "coverage_gap": 5, "retellability": 7,
            "sponsor_audience_quality": 6, "expansion_potential": 7,
            "cares_before_knowing_term": True, "interests_non_ai_person": True,
            "strongest_failure_reason": "the claim rests on essays, not measurements",
            "viewer_understanding_delta": "why checking is now the constraint",
            "old_model": "AI is a thing you ask",
            "new_model": "AI is a process you supervise",
        },
        "pairwise": {"winner": "A", "confidence": 0.7, "explanation": "clearer stake"},
        "red_team": {
            "attacks": [{
                "attack_key": "title_broader_than_evidence", "verdict": "WOUNDED",
                "argument": "the evidence is argument, not measurement",
                "severity": 5,
                "abandon_evidence": "if a measured verification cost shows it is cheap",
            }],
            "overall_verdict": "survives, wounded",
            "single_biggest_risk": "evidence burden",
        },
        "red_team_revision": {
            "should_revise": True,
            "revised_working_title": "Who Checks the AI?",
            "revised_central_question": "If machines outproduce our ability to review, what breaks?",
            "revised_premise": "Verification may be becoming the real constraint.",
            "what_changed": "narrowed the promise to match the evidence",
            "why_unchanged": "",
        },
    }


async def test_full_pipeline_produces_finalists_and_brief(db, settings, make_source_item):
    items = []
    for i in range(8):
        item = make_source_item(
            external_id=f"int{i}",
            text=(
                "verification is becoming the bottleneck for agent harness work; "
                "world models and test time compute keep coming up as well"
            ),
            author=f"author_{i}", publication=f"Pub {i}", days_ago=i % 5,
        )
        db.insert_source_item(item)
        items.append(item)

    provider = FakeProvider(_scripted(items[0].id))
    pipeline = DiscoveryPipeline(db, settings, provider, run_id="integration")
    result = await pipeline.run(top_n=1, coverage_blocks={})

    assert result.llm_available is True
    assert result.signals, "scouts must produce signals"
    assert result.clusters, "clusterer must produce a cluster"
    assert result.candidates, "generator must produce candidates"
    assert result.evaluations, "evaluator must score them"
    assert result.tournament and result.tournament.results, "pairwise must run"
    assert result.red_team, "red team must attack"
    assert result.finalists, "finalists must be assembled"

    # Provenance survives the whole chain.
    cited = {s for sig in result.signals for s in sig.source_ids}
    assert cited, "signals must carry source ids"
    assert cited <= {i.id for i in items}, "no fabricated ids may survive"

    # The red team's wound must actually cost the candidate something.
    top = result.finalists[0]
    assert top.opportunity is not None
    assert top.opportunity.red_team_penalty < 1.0
    assert any("coverage unknown" in n for n in top.opportunity.notes)

    # Every stage is persisted and inspectable.
    out = write_artifacts(result, settings.runs_dir)
    brief = write_daily_brief(db, result, settings.runs_dir)
    write_run_summary(db, result, settings.runs_dir)

    for name in ("02_signals.jsonl", "03_clusters.json", "04_candidates.json",
                 "05_evaluations.json", "06_pairwise.json", "07_red_team.json",
                 "08_finalists.json"):
        assert (out / name).stat().st_size > 2, f"{name} must not be empty"

    text = brief.read_text()
    assert "Who Checks the AI?" in text
    assert "Red team" in text or "red_team" in text
    assert result.cost["calls"] > 0

    # A revision was produced and stored as a distinct row.
    revisions = [c for c in result.candidates if c.revision_of]
    assert revisions, "red team revision pass must produce a revised candidate"
    assert revisions[0].id != revisions[0].revision_of


async def test_pipeline_survives_an_agent_failure(db, settings, make_source_item):
    """One dead scout must not take the run down."""
    items = []
    for i in range(5):
        item = make_source_item(
            external_id=f"f{i}",
            text="verification bottleneck and agent harness work keep coming up",
            author=f"author_{i}", publication=f"Pub {i}", days_ago=i,
        )
        db.insert_source_item(item)
        items.append(item)
    item = items[0]

    scripted = _scripted(item.id)
    scripted["trend_scout"] = RuntimeError("model exploded")

    provider = FakeProvider(scripted)
    pipeline = DiscoveryPipeline(db, settings, provider, run_id="failure")
    result = await pipeline.run(top_n=1, coverage_blocks={})

    agents = {s.agent for s in result.signals}
    assert "trend_scout" not in agents
    assert "mechanism_scout" in agents, "other scouts must still have run"
    assert result.candidates, "the run must still reach candidates"
