"""The claim layer: extraction, persistence and the queries it unlocks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from frontier_scout.agents.claim_extractor import (
    ClaimExtractionOutput,
    ClaimExtractor,
    ExtractedClaim,
    ExtractedEntity,
    ExtractedPosition,
)
from frontier_scout.llm.runner import AgentRunner
from frontier_scout.trends import contradictions
from frontier_scout.trends.registry import ConceptRegistry

from .conftest import FakeProvider

SEED = [
    {"canonical_label": "world models", "aliases": ["world models", "world model"]},
    {"canonical_label": "agent harness", "aliases": ["agent harness"]},
]


def _extractor(db, settings, responses=None) -> ClaimExtractor:
    registry = ConceptRegistry(db)
    registry.seed(SEED)
    provider = FakeProvider(responses or {})
    runner = AgentRunner(provider, settings, db, "claims", use_cache=False)
    return ClaimExtractor(db, runner, registry)


def _article(db, make_source_item, *, ext: str, text: str, author: str, days_ago: int = 1):
    item = make_source_item(
        external_id=ext, text=text, author=author,
        publication=f"Pub {author}", days_ago=days_ago,
    )
    db.insert_source_item(item)
    return db.conn.execute(
        "SELECT id,title,text,source_name,published_at,author_id,source_type,"
        "primary_or_secondary FROM source_items WHERE id=?", (item.id,)
    ).fetchone()


# -- persistence -----------------------------------------------------------


def test_extraction_persists_claims_positions_and_entities(db, settings, make_source_item):
    extractor = _extractor(db, settings)
    row = _article(db, make_source_item, ext="a1",
                   text="world models are becoming practical", author="writer")

    output = ClaimExtractionOutput(
        claims=[ExtractedClaim(
            proposition="world models now run in real time",
            stance="asserts", evidence_type="benchmark", subject="world models",
            quote="runs at 24fps",
        )],
        positions=[ExtractedPosition(
            concept="world models", stance="advocating", rationale="calls it practical"
        )],
        entities=[ExtractedEntity(name="DeepMind", entity_type="org")],
    )
    claims, positions = extractor.persist(row, output)

    assert (claims, positions) == (1, 1)
    stored = db.conn.execute("SELECT * FROM article_claims").fetchone()
    assert stored["proposition"] == "world models now run in real time"
    assert stored["evidence_type"] == "benchmark"
    assert stored["concept_id"] is not None, "should resolve to the tracked concept"
    assert db.conn.execute("SELECT COUNT(*) c FROM article_entities").fetchone()["c"] == 1


def test_positions_naming_untracked_concepts_are_dropped(db, settings, make_source_item):
    """A model naming a concept we do not track must not create a phantom row."""
    extractor = _extractor(db, settings)
    row = _article(db, make_source_item, ext="a2", text="some text here", author="w")

    extractor.persist(row, ClaimExtractionOutput(
        positions=[ExtractedPosition(concept="quantum blockchain", stance="advocating")]
    ))
    assert db.conn.execute("SELECT COUNT(*) c FROM article_positions").fetchone()["c"] == 0


def test_articles_are_not_re_extracted(db, settings, make_source_item):
    extractor = _extractor(db, settings)
    row = _article(db, make_source_item, ext="a3",
                   text="x" * 600, author="w")

    assert any(r["id"] == row["id"] for r in extractor.pending())
    extractor.persist(row, ClaimExtractionOutput())
    assert not any(r["id"] == row["id"] for r in extractor.pending())


def test_zero_claim_articles_are_marked_not_retried_forever(db, settings, make_source_item):
    """A link roundup yields nothing; it must not be re-paid for every run."""
    extractor = _extractor(db, settings)
    row = _article(db, make_source_item, ext="a4", text="y" * 600, author="w")

    extractor.persist(row, ClaimExtractionOutput())
    marker = db.conn.execute("SELECT * FROM claim_extractions").fetchone()
    assert marker["status"] == "empty"
    assert not extractor.pending()


def test_failed_extraction_is_recorded(db, settings, make_source_item):
    extractor = _extractor(db, settings)
    row = _article(db, make_source_item, ext="a5", text="z" * 600, author="w")

    extractor.persist(row, None)
    marker = db.conn.execute("SELECT * FROM claim_extractions").fetchone()
    assert marker["status"] == "failed"


def test_pending_is_selective_by_length_and_type(db, settings, make_source_item):
    extractor = _extractor(db, settings)
    _article(db, make_source_item, ext="short", text="tiny", author="w")
    _article(db, make_source_item, ext="long", text="w" * 3000, author="w")

    assert len(extractor.pending(min_chars=1000)) == 1
    assert extractor.pending(source_types=["arxiv"]) == []


async def test_extract_one_sends_tracked_concepts_and_json_instruction(
    db, settings, make_source_item
):
    provider_responses = {"claim_extractor": {"claims": [], "positions": [], "entities": []}}
    extractor = _extractor(db, settings, provider_responses)
    row = _article(db, make_source_item, ext="a6", text="world models discussion", author="w")

    await extractor.extract_one(row)
    request = extractor.runner.provider.calls[0]

    assert "world models" in request.task
    assert "json" in request.task.lower()   # required by DeepSeek json_object mode
    assert request.metadata["role"] == "extractor"


# -- the queries the layer unlocks ----------------------------------------


def _position(db, item_id, concept_id, author_id, stance, when):
    with db.tx() as c:
        c.execute(
            "INSERT OR REPLACE INTO article_positions(source_item_id,concept_id,author_id,"
            "stance,rationale,confidence,observed_at) VALUES(?,?,?,?,?,?,?)",
            (item_id, concept_id, author_id, stance, f"{stance} because", 0.8, when),
        )


def test_dispute_needs_real_people_on_both_sides(db, settings, make_source_item):
    """The brief forbids manufacturing balance, so a one-sided split says so."""
    extractor = _extractor(db, settings)
    concept_id = extractor.registry.concept_ids[0]
    now = datetime.now(timezone.utc)

    rows = [
        _article(db, make_source_item, ext=f"d{i}", text="argument", author=f"author_{i}")
        for i in range(4)
    ]
    for i, row in enumerate(rows):
        stance = "advocating" if i < 3 else "skeptical"
        _position(db, row["id"], concept_id, row["author_id"], stance,
                  (now - timedelta(days=i)).isoformat())

    disputes = contradictions.find_disputes(db, concept_id=concept_id)
    assert disputes, "opposing stances must be detected"
    assert disputes[0].is_genuine is False, "1 skeptic is not a genuine split"
    assert "one-sided" in disputes[0].render()


def test_genuine_dispute_is_flagged(db, settings, make_source_item):
    extractor = _extractor(db, settings)
    concept_id = extractor.registry.concept_ids[0]
    now = datetime.now(timezone.utc)

    for i in range(4):
        row = _article(db, make_source_item, ext=f"g{i}", text="argument", author=f"auth_{i}")
        _position(db, row["id"], concept_id, row["author_id"],
                  "advocating" if i % 2 == 0 else "skeptical",
                  (now - timedelta(days=i)).isoformat())

    dispute = contradictions.find_disputes(db, concept_id=concept_id)[0]
    assert dispute.is_genuine is True
    assert {s.stance for s in dispute.sides} == {"advocating", "skeptical"}


def test_author_reversal_is_detected(db, settings, make_source_item):
    """Same author, opposite stance, months apart -- invisible before this."""
    extractor = _extractor(db, settings)
    concept_id = extractor.registry.concept_ids[0]
    now = datetime.now(timezone.utc)

    old = _article(db, make_source_item, ext="r1", text="early view",
                   author="changer", days_ago=200)
    new = _article(db, make_source_item, ext="r2", text="later view",
                   author="changer", days_ago=5)
    _position(db, old["id"], concept_id, old["author_id"], "skeptical",
              (now - timedelta(days=200)).isoformat())
    _position(db, new["id"], concept_id, new["author_id"], "advocating",
              (now - timedelta(days=5)).isoformat())

    reversals = contradictions.find_reversals(db)
    assert any(r.author == "changer" and r.from_stance == "skeptical"
               and r.to_stance == "advocating" for r in reversals)


def test_explicit_self_correction_surfaces_as_a_reversal(db, settings, make_source_item):
    extractor = _extractor(db, settings)
    row = _article(db, make_source_item, ext="sc1", text="I was wrong about this",
                   author="honest")
    extractor.persist(row, ClaimExtractionOutput(claims=[ExtractedClaim(
        proposition="I previously argued the opposite and was wrong",
        stance="asserts", is_self_correction=True, subject="world models",
    )]))

    reversals = contradictions.find_reversals(db)
    assert any("wrong" in r.rationale for r in reversals)


def test_surprising_claims_are_queryable(db, settings, make_source_item):
    extractor = _extractor(db, settings)
    row = _article(db, make_source_item, ext="s1", text="contrarian piece", author="w")
    extractor.persist(row, ClaimExtractionOutput(claims=[
        ExtractedClaim(proposition="the consensus is wrong", stance="asserts",
                       is_surprising=True, subject="world models"),
        ExtractedClaim(proposition="ordinary observation", stance="asserts",
                       subject="world models"),
    ]))

    surprises = contradictions.find_surprises(db)
    assert len(surprises) == 1
    assert surprises[0]["proposition"] == "the consensus is wrong"


def test_dossier_section_is_empty_when_nothing_extracted(db, settings):
    extractor = _extractor(db, settings)
    assert contradictions.render_for_dossier(db, extractor.registry.concept_ids[0]) == ""


def test_dossier_section_includes_claims_via_article_provenance(
    db, settings, make_source_item
):
    """A claim need not name the concept if its article was about it."""
    extractor = _extractor(db, settings)
    registry = extractor.registry
    concept_id = registry.concept_ids[0]

    row = _article(db, make_source_item, ext="p1",
                   text="world models are the topic of this whole piece", author="w")
    registry.backfill_mentions()
    extractor.persist(row, ClaimExtractionOutput(claims=[ExtractedClaim(
        proposition="simulators now predict action consequences",
        stance="asserts", evidence_type="benchmark", subject="simulation",
    )]))

    rendered = contradictions.render_for_dossier(db, concept_id)
    assert "simulators now predict" in rendered


def test_org_bylines_are_not_treated_as_one_mind_changing(db, settings, make_source_item):
    """A company blog is many writers; two differing posts is not a reversal.

    Live data produced "Anthropic moved from skeptical to advocating on evals"
    across two posts one day apart. That is an artifact of one byline covering
    many authors, and it would be a false editorial lead.
    """
    from frontier_scout.models import SourceItem, SourceRole, SourceType

    extractor = _extractor(db, settings)
    concept_id = extractor.registry.concept_ids[0]
    now = datetime.now(timezone.utc)

    ids = []
    for i, stance in enumerate(("skeptical", "advocating")):
        item = SourceItem(
            source_type=SourceType.COMPANY_BLOG, source_name="BigLab",
            external_id=f"org{i}", title="post", text="an argument about world models",
            author_handle="BigLab", publication="BigLab",
            published_at=now - timedelta(days=200 - i * 150),
            source_role=SourceRole.PRIMARY_EVIDENCE, primary_or_secondary="PRIMARY",
        )
        db.insert_source_item(item)
        ids.append(item.id)
        row = db.conn.execute(
            "SELECT author_id FROM source_items WHERE id=?", (item.id,)
        ).fetchone()
        _position(db, item.id, concept_id, row["author_id"], stance,
                  (now - timedelta(days=200 - i * 150)).isoformat())

    assert not [r for r in contradictions.find_reversals(db) if r.author == "BigLab"]


def test_same_week_stance_flip_is_not_a_reversal(db, settings, make_source_item):
    """Changing your mind takes longer than a day."""
    extractor = _extractor(db, settings)
    concept_id = extractor.registry.concept_ids[0]
    now = datetime.now(timezone.utc)

    for i, stance in enumerate(("skeptical", "advocating")):
        row = _article(db, make_source_item, ext=f"fast{i}", text="a view",
                       author="hasty", days_ago=10 - i)
        _position(db, row["id"], concept_id, row["author_id"], stance,
                  (now - timedelta(days=10 - i)).isoformat())

    assert not [r for r in contradictions.find_reversals(db) if r.author == "hasty"]


def test_failed_extractions_are_retried_but_settled_ones_are_not(
    db, settings, make_source_item
):
    """A token overrun is transient; an empty article is a settled answer."""
    extractor = _extractor(db, settings)
    failed = _article(db, make_source_item, ext="f1", text="a" * 900, author="w")
    empty = _article(db, make_source_item, ext="e1", text="b" * 900, author="w")

    extractor.persist(failed, None)                      # -> failed
    extractor.persist(empty, ClaimExtractionOutput())    # -> empty

    pending_ids = {r["id"] for r in extractor.pending()}
    assert failed["id"] in pending_ids, "a failed extraction must be retried"
    assert empty["id"] not in pending_ids, "an empty article must not be re-paid for"
