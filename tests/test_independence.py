"""The independence graph.

These are the highest-value tests in the suite: they encode the difference
between genuine convergence and an echo, which is the judgement the whole
system exists to make.
"""

from __future__ import annotations

from frontier_scout.trends.independence import compute_independence


def test_five_posts_one_author_is_one_source(db, make_source_item):
    ids = []
    for i in range(5):
        item = make_source_item(external_id=f"a{i}", text="world models are the future",
                                author="same_person")
        db.insert_source_item(item)
        ids.append(item.id)

    result = compute_independence(db, ids)
    assert result.independent_source_count == 1
    assert result.distinct_authors == 1
    assert result.independence_ratio == 1.0  # 1 component / 1 author


def test_four_writers_citing_one_origin_collapse_to_one_source(db, make_source_item):
    """The trap: four voices that look like consensus but share an origin."""
    origin = "https://example.com/the-original-essay"
    ids = []
    for i in range(4):
        item = make_source_item(
            external_id=f"echo{i}",
            text=f"writer {i} responding to the essay",
            author=f"writer_{i}",
            publication=f"Pub {i}",
            links=[origin],
        )
        db.insert_source_item(item)
        ids.append(item.id)

    result = compute_independence(db, ids)
    assert result.distinct_authors == 4
    assert result.independent_source_count == 1, "shared origin must collapse them"
    assert result.independence_ratio == 0.25
    assert result.is_single_origin_cascade is True


def test_four_independent_writers_stay_independent(db, make_source_item):
    """The buy signal: genuine convergence with no shared origin."""
    ids = []
    topics = [
        "verification is becoming the real constraint on agent systems",
        "checking machine output costs more than producing it these days",
        "the bottleneck moved from generation to confirming correctness",
        "nobody can review what the models now produce at this volume",
    ]
    for i, text in enumerate(topics):
        item = make_source_item(
            external_id=f"indep{i}", text=text,
            author=f"author_{i}", publication=f"Pub {i}",
            links=[f"https://unrelated-{i}.example.com/page"],
        )
        db.insert_source_item(item)
        ids.append(item.id)

    result = compute_independence(db, ids)
    assert result.distinct_authors == 4
    assert result.independent_source_count == 4
    assert result.independence_ratio == 1.0
    assert result.is_single_origin_cascade is False


def test_shared_paper_reference_links_items(db, make_source_item):
    ids = []
    for i in range(3):
        item = make_source_item(
            external_id=f"paper{i}", text=f"discussion {i}",
            author=f"author_{i}", arxiv=["2501.12345"],
        )
        db.insert_source_item(item)
        ids.append(item.id)

    result = compute_independence(db, ids)
    assert result.independent_source_count == 1
    assert any("2501.12345" in reason for reason in result.join_reasons)


def test_generic_links_do_not_collapse_unrelated_writers(db, make_source_item):
    """Everyone links GitHub; joining on it would hide real convergence."""
    ids = []
    for i in range(3):
        item = make_source_item(
            external_id=f"gen{i}", text=f"entirely different argument number {i}",
            author=f"author_{i}", links=["https://github.com/some/repo"],
        )
        db.insert_source_item(item)
        ids.append(item.id)

    result = compute_independence(db, ids)
    assert result.independent_source_count == 3


def test_near_duplicate_text_is_treated_as_copied(db, make_source_item):
    shared = (
        "the shift from prompting to harness design is the most important "
        "change in how engineers actually work with these models this year "
        "and it has happened almost entirely without public notice at all"
    )
    ids = []
    for i in range(2):
        item = make_source_item(
            external_id=f"dup{i}", text=shared, author=f"author_{i}",
        )
        db.insert_source_item(item)
        ids.append(item.id)

    result = compute_independence(db, ids)
    assert result.independent_source_count == 1


def test_empty_input_is_safe(db):
    result = compute_independence(db, [])
    assert result.independent_source_count == 0
    assert result.independence_ratio == 1.0
