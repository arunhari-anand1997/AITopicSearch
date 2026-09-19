"""SerpAPI public-interest readings and the X-citation signal."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from frontier_scout.ingest.serpapi_trends import (
    SOURCE_LABEL,
    SerpApiTrendsConnector,
    _trajectory,
)
from frontier_scout.models import ConnectorStatus, SourceType
from frontier_scout.trends import x_citations


def _timeline(values: list[int]) -> dict:
    return {
        "interest_over_time": {
            "timeline_data": [
                {"date": f"day {i}", "values": [{"extracted_value": v}]}
                for i, v in enumerate(values)
            ]
        }
    }


def _connector(settings, handler, *, enabled: bool = True) -> SerpApiTrendsConnector:
    settings.serpapi_key = "test-key"
    settings.raw = {**settings.raw, "google_trends": {"enabled": enabled, "max_searches_per_run": 2}}
    conn = SerpApiTrendsConnector(settings)
    conn._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return conn


# -- Google Trends via SerpAPI --------------------------------------------


async def test_missing_key_reports_not_available_not_a_guess(settings):
    settings.serpapi_key = None
    conn = SerpApiTrendsConnector(settings)
    result = await conn.interest("world models")
    assert result.status == ConnectorStatus.NOT_CONFIGURED
    assert "NOT_AVAILABLE" in result.summary()


async def test_disabled_in_config_is_respected(settings):
    conn = _connector(settings, lambda r: httpx.Response(200, json=_timeline([1] * 20)),
                      enabled=False)
    result = await conn.interest("x")
    await conn._client.aclose()
    assert result.status == ConnectorStatus.NOT_CONFIGURED


async def test_reading_is_labelled_as_serpapi_not_official_google(settings):
    """Provenance must never imply first-party Google data."""
    conn = _connector(settings, lambda r: httpx.Response(200, json=_timeline([50] * 20)))
    result = await conn.interest("world models")
    await conn._client.aclose()
    assert result.source == SOURCE_LABEL
    assert SOURCE_LABEL in result.summary()


async def test_per_run_search_ceiling_is_enforced(settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_timeline(list(range(20))))

    conn = _connector(settings, handler)  # ceiling is 2
    for query in ("a", "b"):
        assert (await conn.interest(query)).status == ConnectorStatus.OK
    third = await conn.interest("c")
    await conn._client.aclose()

    assert third.status == ConnectorStatus.BUDGET_EXHAUSTED
    assert calls["n"] == 2, "must not call the paid API past the ceiling"


async def test_results_are_cached_to_protect_quota(settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_timeline(list(range(20))))

    conn = _connector(settings, handler)
    await conn.interest("repeat me")
    await conn.interest("repeat me")
    await conn._client.aclose()
    assert calls["n"] == 1, "weekly data must not be re-fetched within the cache window"


async def test_api_error_is_surfaced_not_swallowed(settings):
    conn = _connector(settings, lambda r: httpx.Response(200, json={"error": "bad query"}))
    result = await conn.interest("x")
    await conn._client.aclose()
    assert result.status == ConnectorStatus.SOURCE_UNAVAILABLE
    assert "NOT_AVAILABLE" in result.summary()


@pytest.mark.parametrize(
    "values,expected",
    [
        ([10] * 8 + [40] * 4, "rising"),
        ([40] * 8 + [10] * 4, "falling"),
        ([20] * 12, "flat"),
    ],
)
def test_trajectory_classification(values, expected):
    change, direction = _trajectory([(str(i), v) for i, v in enumerate(values)])
    assert direction == expected


def test_trajectory_needs_enough_history():
    change, direction = _trajectory([("a", 1), ("b", 2)])
    assert change is None and direction is None


# -- X via citations -------------------------------------------------------


def _cite(db, make_source_item, *, ext: str, author: str, url: str, pub: str = "Pub"):
    item = make_source_item(
        external_id=ext, text="discussion", author=author, publication=pub, links=[url]
    )
    db.insert_source_item(item)
    return item


def test_citation_signal_ranks_by_distinct_authors(db, make_source_item):
    popular = "https://twitter.com/karpathy/status/111"
    lonely = "https://twitter.com/someone/status/222"
    for i in range(4):
        _cite(db, make_source_item, ext=f"p{i}", author=f"author_{i}",
              url=popular, pub=f"Pub {i}")
    _cite(db, make_source_item, ext="l0", author="author_9", url=lonely)

    signal = x_citations.compute(db, days=90)
    assert signal.top_posts[0].url == popular
    assert signal.top_posts[0].citing_authors == 4
    assert signal.top_posts[0].is_independently_endorsed is True
    assert signal.top_posts[1].is_independently_endorsed is False


def test_product_surfaces_are_not_treated_as_accounts(db, make_source_item):
    """/i/, /search and /intent are X's own pages, not people."""
    for i, url in enumerate(
        [
            "https://twitter.com/i/status/1",
            "https://x.com/search?q=ai",
            "https://x.com/intent/tweet",
            "https://x.com/karpathy/status/9",
        ]
    ):
        _cite(db, make_source_item, ext=f"n{i}", author=f"a{i}", url=url)

    signal = x_citations.compute(db, days=90)
    handles = {h for h, _, _ in signal.top_accounts}
    assert handles == {"karpathy"}


def test_signal_can_be_scoped_to_one_concept(db, make_source_item):
    from frontier_scout.trends.registry import ConceptRegistry

    registry = ConceptRegistry(db)
    registry.seed([{"canonical_label": "world models", "aliases": ["world models"]}])

    on_topic = make_source_item(
        external_id="on", text="world models are the future", author="a1",
        links=["https://twitter.com/someone/status/1"],
    )
    off_topic = make_source_item(
        external_id="off", text="unrelated chatter", author="a2",
        links=["https://twitter.com/other/status/2"],
    )
    db.insert_source_item(on_topic)
    db.insert_source_item(off_topic)
    registry.backfill_mentions()

    concept_id = registry.concept_ids[0]
    scoped = x_citations.compute(db, concept_id=concept_id, days=90)
    urls = {p.url for p in scoped.top_posts}
    assert "https://twitter.com/someone/status/1" in urls
    assert "https://twitter.com/other/status/2" not in urls


def test_render_always_states_the_lagging_caveat(db, make_source_item):
    _cite(db, make_source_item, ext="c1", author="a1",
          url="https://twitter.com/karpathy/status/5")
    rendered = x_citations.compute(db, days=90).render()
    assert "No X API was used" in rendered
    assert "LAGGING" in rendered


def test_empty_corpus_renders_nothing(db):
    assert x_citations.compute(db, days=90).render() == ""
