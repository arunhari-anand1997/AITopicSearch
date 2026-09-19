"""Concept registry and trend arithmetic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from frontier_scout.trends.engine import TrendEngine, squash
from frontier_scout.trends.registry import ConceptRegistry, is_specific_enough, normalize


def test_normalize_handles_unicode_apostrophes():
    """Smart quotes split contractions and produce phantom bigrams like 'don t'."""
    assert normalize("Don’t worry") == "don't worry"


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("don t", False),          # contraction fragment
        ("claude s", False),       # possessive fragment
        ("5 1", False),            # version number
        ("large language", False), # all-generic tokens
        ("ai models", False),
        ("world models", True),
        ("verification bottleneck", True),
        ("graph engineering", True),
    ],
)
def test_phrase_specificity_filter(phrase, expected):
    assert is_specific_enough(phrase) is expected


def test_registry_matches_aliases_to_one_stable_concept(db):
    registry = ConceptRegistry(db)
    registry.seed([
        {
            "canonical_label": "agent harness",
            "kind": "term",
            "aliases": ["agent harness", "harness engineering", "agent scaffolding"],
        }
    ])
    hits_a = registry.find_mentions("teams are investing in harness engineering now")
    hits_b = registry.find_mentions("the agent scaffolding matters more than the prompt")
    assert hits_a and hits_b
    assert hits_a[0][0] == hits_b[0][0], "aliases must resolve to one concept id"


def test_registry_keeps_context_for_sense_checking(db):
    """A literal match is not a same-sense match; the snippet is how we tell."""
    registry = ConceptRegistry(db)
    registry.seed([{"canonical_label": "loop engineering", "aliases": ["loop engineering"]}])
    hits = registry.find_mentions(
        "in 2012 control loop engineering for industrial plants was routine work"
    )
    assert hits
    _cid, _offset, context = hits[0]
    assert "control" in context and "industrial" in context


def test_longest_alias_wins(db):
    registry = ConceptRegistry(db)
    registry.seed([
        {"canonical_label": "compute", "aliases": ["compute"]},
        {"canonical_label": "test-time compute", "aliases": ["test time compute"]},
    ])
    hits = registry.find_mentions("scaling test time compute changes the economics")
    labels = {registry.label(cid) for cid, _, _ in hits}
    assert "test-time compute" in labels


def test_squash_compresses_and_caps():
    assert squash(0) == 0.0
    assert squash(None) is None
    assert squash(1000, cap=10) == 1.0
    assert squash(2) < squash(6) < 1.0


def test_velocity_is_insufficient_history_not_zero(db, settings, make_source_item):
    """A new concept must not be rankable as 'not accelerating'."""
    registry = ConceptRegistry(db)
    registry.seed([{"canonical_label": "world models", "aliases": ["world models"]}])
    for i in range(4):
        item = make_source_item(
            external_id=f"wm{i}", text="world models are interesting",
            author=f"a{i}", days_ago=i,
        )
        db.insert_source_item(item)
    registry.backfill_mentions()
    registry.update_first_observed()

    engine = TrendEngine(db, settings, registry)
    metrics = engine.compute_concept(registry.concept_ids[0])

    assert metrics.velocity_ratio is None
    assert metrics.components.normalized_velocity is None
    assert any("INSUFFICIENT_HISTORY" in n for n in metrics.notes)


def test_missing_components_renormalise_weights(db, settings, make_source_item):
    """Momentum must not be dragged down merely because a component is absent."""
    registry = ConceptRegistry(db)
    registry.seed([{"canonical_label": "world models", "aliases": ["world models"]}])
    for i in range(5):
        item = make_source_item(
            external_id=f"r{i}", text="world models everywhere",
            author=f"author_{i}", publication=f"Pub {i}", days_ago=i,
        )
        db.insert_source_item(item)
    registry.backfill_mentions()
    registry.update_first_observed()

    engine = TrendEngine(db, settings, registry)
    metrics = engine.compute_concept(registry.concept_ids[0])

    assert metrics.insider_momentum_raw > 0, "must score on available components alone"
    assert metrics.data_completeness < 1.0
    assert metrics.missing_sources


def test_independence_discounts_momentum(db, settings, make_source_item):
    registry = ConceptRegistry(db)
    registry.seed([{"canonical_label": "world models", "aliases": ["world models"]}])
    origin = "https://example.com/origin"
    for i in range(4):
        item = make_source_item(
            external_id=f"casc{i}", text="world models, following the original post",
            author=f"author_{i}", publication=f"Pub {i}", links=[origin], days_ago=i,
        )
        db.insert_source_item(item)
    registry.backfill_mentions()
    registry.update_first_observed()

    engine = TrendEngine(db, settings, registry)
    metrics = engine.compute_concept(registry.concept_ids[0])

    assert metrics.independence_ratio < 1.0
    assert metrics.insider_momentum < metrics.insider_momentum_raw
    assert any("single-origin cascade" in n for n in metrics.notes)


def test_nominated_phrases_are_staged_not_tracked(db, make_source_item):
    """Auto-discovered phrases must not silently become gate candidates."""
    registry = ConceptRegistry(db)
    for i in range(8):
        db.insert_source_item(
            make_source_item(
                external_id=f"nom{i}",
                text="the quiet revolution in provenance ledgers is accelerating fast",
                author=f"author_{i}", publication=f"Pub {i % 4}", days_ago=1,
            )
        )
    registry.nominate_phrases(min_authors=3, min_count=3)
    for cid in registry.concept_ids:
        if registry.status(cid) == "nominated":
            assert cid not in registry.tracked_ids
            break
