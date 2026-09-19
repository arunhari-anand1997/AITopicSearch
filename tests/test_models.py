"""Schema validation and the provenance contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from frontier_scout.models import (
    Claim,
    ClaimKind,
    ConceptMetrics,
    Evaluation,
    RedTeamFinding,
    SourceItem,
    SourceRole,
    SourceType,
    TrendComponents,
    validate_citations,
)


def test_source_item_id_is_stable_and_content_addressed():
    a = SourceItem(
        source_type=SourceType.SUBSTACK, source_name="P", external_id="x1",
        source_role=SourceRole.EARLY_SIGNAL,
    )
    b = SourceItem(
        source_type=SourceType.SUBSTACK, source_name="different name",
        external_id="x1", source_role=SourceRole.COMMUNITY_REACTION,
    )
    assert a.id == b.id, "same source_type + external_id must yield the same id"
    assert a.id.startswith("substack_")


def test_source_fact_requires_a_citation():
    with pytest.raises(ValidationError):
        Claim(text="unsourced", kind=ClaimKind.SOURCE_FACT, source_ids=[])


def test_inference_needs_no_citation():
    claim = Claim(text="my reasoning", kind=ClaimKind.INFERENCE)
    assert claim.source_ids == []


def test_validate_citations_reports_fabricated_ids():
    claims = [
        Claim(text="ok", kind=ClaimKind.SOURCE_FACT, source_ids=["real_1"]),
        Claim(text="bad", kind=ClaimKind.SOURCE_FACT, source_ids=["real_1", "ghost_9"]),
    ]
    assert validate_citations(claims, {"real_1"}) == ["ghost_9"]


def test_confidence_is_clamped():
    assert Claim(text="t", kind=ClaimKind.INFERENCE, confidence=5.0).confidence == 1.0
    assert Claim(text="t", kind=ClaimKind.INFERENCE, confidence=-2.0).confidence == 0.0


def test_unavailable_trend_components_are_none_not_zero():
    """The distinction the whole trend engine rests on."""
    comps = TrendComponents(normalized_velocity=None, novelty=0.5)
    available = comps.available()
    assert "normalized_velocity" not in available
    assert available["novelty"] == 0.5
    assert 0.0 not in [comps.normalized_velocity]


def test_evaluation_rejects_unknown_dimensions_and_clamps_range():
    ev = Evaluation(scores={"stakes": 99, "not_a_dimension": 5, "novelty": -3})
    assert ev.scores["stakes"] == 10
    assert ev.scores["novelty"] == 1
    assert "not_a_dimension" not in ev.scores


def test_red_team_severity_clamped():
    assert RedTeamFinding(attack_key="k", severity=50).severity == 10


def test_metrics_default_google_trends_to_not_available():
    """Absent Trends access must never look like a measured zero."""
    m = ConceptMetrics(concept_id=1)
    assert m.google_search_change == "NOT_AVAILABLE"
