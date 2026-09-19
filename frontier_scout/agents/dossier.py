"""Building the source dossier that agents actually read.

Scouts never reason from a summary. They get real excerpts with real source
IDs, authors, dates and URLs, plus the computed trend statistics, and they are
required to cite the IDs. This module assembles that packet.

The dossier is identical for all eight scouts on a given cluster, which is why
it goes into the cached system block: eight calls, one paid copy of the text.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from ..db import Database
from ..models import ConceptMetrics, SignalPropagation

EXCERPT_CHARS = 1400
MAX_ITEMS_PER_CONCEPT = 14


@dataclass
class ConceptGroup:
    """A pre-cluster: concepts that co-occur, scored together."""

    key: str
    concept_ids: list[int] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    metrics: list[ConceptMetrics] = field(default_factory=list)
    propagation: list[SignalPropagation] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)

    @property
    def momentum(self) -> float:
        """Group momentum is its strongest member, not its average.

        Averaging would let one hot concept be diluted by its quieter
        neighbours, which is the opposite of what a gate should do.
        """
        return max((m.insider_momentum for m in self.metrics), default=0.0)

    @property
    def title(self) -> str:
        return " / ".join(self.labels[:4])


def _fmt_dt(value: Any) -> str:
    if not value:
        return "undated"
    return str(value)[:10]


def build_dossier(db: Database, group: ConceptGroup, *, max_items: int = 40) -> str:
    """Render the evidence packet for one concept group.

    Every excerpt carries its source ID so the agent can cite it, and the
    provenance header states plainly which parts are observation and which the
    agent is allowed to infer.
    """
    lines: list[str] = []
    lines.append("# SOURCE MATERIAL")
    lines.append(
        "Everything in this section is OBSERVED DATA collected from real sources. "
        "Each item has a SOURCE ID in [brackets]. When you state something you "
        "read here, cite its source ID. When you reason beyond what is written "
        "here, label it INFERENCE and cite nothing."
    )
    lines.append("")

    # -- measured signal ---------------------------------------------------
    lines.append("## MEASURED TREND SIGNAL (computed, not estimated)")
    lines.append("")
    for m in group.metrics:
        lines.append(f"### {m.concept_label}")
        lines.append(
            f"- mentions: {m.mentions_24h} in 24h, {m.mentions_7d} in 7d, "
            f"{m.mentions_30d} in 30d"
        )
        lines.append(
            f"- unique authors: {m.unique_authors_7d} in 7d "
            f"(previous 7d: {m.unique_authors_prev_7d}); "
            f"publications: {m.unique_publications_7d}"
        )
        lines.append(
            f"- INDEPENDENT sources: {m.independent_source_count} "
            f"(independence ratio {m.independence_ratio:.2f} -- "
            f"1.0 means genuinely separate voices, low means they share an origin)"
        )
        if m.velocity_ratio is not None:
            lines.append(
                f"- velocity: {m.velocity_ratio:.2f}x the {m.baseline_daily:.1f}/day baseline"
            )
        else:
            lines.append(
                "- velocity: NOT COMPUTED (insufficient baseline history). "
                "Do not describe this concept as accelerating or decaying on "
                "velocity grounds."
            )
        lines.append(
            f"- platforms: {m.cross_platform_count}, evidence roles: {m.cross_role_count}, "
            f"papers: {m.paper_count}"
        )
        lines.append(f"- momentum: {m.insider_momentum:.3f} (raw {m.insider_momentum_raw:.3f})")
        lines.append(f"- public search interest: {m.google_search_change}")
        for note in m.notes:
            lines.append(f"- NOTE: {note}")
        lines.append("")

    # -- propagation -------------------------------------------------------
    if group.propagation:
        lines.append("## PROPAGATION (which platform saw it first)")
        for prop in group.propagation:
            if not prop.platform_sequence:
                continue
            label = next(
                (m.concept_label for m in group.metrics if m.concept_id == prop.concept_id),
                "concept",
            )
            chain = " -> ".join(
                f"{step['platform']} ({_fmt_dt(step['first_seen'])})"
                for step in prop.platform_sequence
            )
            lines.append(f"- {label}: {chain}")
        lines.append("")

    # -- the actual sources ------------------------------------------------
    lines.append("## SOURCE ITEMS")
    lines.append("")
    rows = _fetch_items(db, group, max_items)
    if not rows:
        lines.append("(no source items available)")
    for row in rows:
        lines.append(f"[{row['id']}]")
        lines.append(f"- type: {row['source_type']} | role: {row['source_role']}")
        lines.append(f"- publication: {row['source_name']}")
        if row["author"]:
            lines.append(f"- author: {row['author']}")
        lines.append(f"- date: {_fmt_dt(row['published_at'])}")
        if row["url"]:
            lines.append(f"- url: {row['url']}")
        if row["engagement"]:
            lines.append(f"- engagement: {row['engagement']}")
        if row["primary_or_secondary"] == "PRIMARY":
            lines.append(
                "- NOTE: PRIMARY SOURCE. This is evidence of what this "
                "organisation CLAIMS, not evidence the claim is correct."
            )
        if row["paywalled"]:
            lines.append("- NOTE: paywalled; only the public excerpt is available.")
        if row["title"]:
            lines.append(f"- title: {row['title']}")
        excerpt = (row["text"] or "").strip().replace("\n", " ")
        if excerpt:
            lines.append(f"- excerpt: {excerpt[:EXCERPT_CHARS]}")
        lines.append("")

    return "\n".join(lines)


def _fetch_items(db: Database, group: ConceptGroup, max_items: int) -> list[sqlite3.Row]:
    """Pick the items worth showing.

    Balanced across concepts and biased to recent, high-engagement material, so
    one loud concept cannot crowd the others out of the dossier.
    """
    collected: dict[str, sqlite3.Row] = {}
    per_concept = max(3, max_items // max(len(group.concept_ids), 1))

    for concept_id in group.concept_ids:
        rows = db.conn.execute(
            "SELECT DISTINCT s.id, s.source_type, s.source_name, s.source_role,"
            " s.primary_or_secondary, s.title, s.text, s.url, s.published_at,"
            " s.paywalled, a.handle AS author,"
            " (SELECT COALESCE(MAX(COALESCE(e.score,0)+COALESCE(e.likes,0)"
            "   +COALESCE(e.comment_count,0)),0) FROM engagement_snapshots e"
            "   WHERE e.source_item_id = s.id) AS engagement"
            " FROM concept_mentions m"
            " JOIN source_items s ON s.id = m.source_item_id"
            " LEFT JOIN authors a ON a.id = s.author_id"
            " WHERE m.concept_id = ?"
            " ORDER BY s.published_at DESC, engagement DESC"
            " LIMIT ?",
            (concept_id, min(per_concept, MAX_ITEMS_PER_CONCEPT)),
        ).fetchall()
        for row in rows:
            collected.setdefault(row["id"], row)

    ordered = sorted(
        collected.values(),
        key=lambda r: (str(r["published_at"] or ""), r["engagement"] or 0),
        reverse=True,
    )
    return ordered[:max_items]


def source_ids_for(db: Database, group: ConceptGroup, max_items: int = 40) -> set[str]:
    """The set of IDs an agent is allowed to cite for this group."""
    return {row["id"] for row in _fetch_items(db, group, max_items)}


def build_groups(
    db: Database,
    metrics: list[ConceptMetrics],
    *,
    max_groups: int = 12,
    overlap_threshold: float = 0.35,
    min_independent_sources: int = 3,
    tracked_only: bool = True,
) -> list[ConceptGroup]:
    """Pre-cluster concepts by co-occurrence, deterministically.

    This runs before any LLM call. Grouping related concepts means the scouts
    see a whole family at once -- both cheaper and better reasoning than eight
    separate calls about five words for the same idea.

    Two filters guard the gate, because everything past it costs money:

    `tracked_only` keeps auto-nominated phrases out. They are measured and
    reported, but a phrase the system discovered on its own is a suggestion for
    the editor, not a licence to spend. Without this, thin junk n-grams
    ("use claude", "across three") outrank real concepts -- they are maximally
    novel by construction and their tiny denominators inflate every ratio.

    `min_independent_sources` drops concepts whose evidence is too thin to
    reason about at all. Two source items cannot support a documentary.
    """
    if not metrics:
        return []

    eligible = [
        m for m in metrics
        if (not tracked_only or m.status == "tracked")
        and m.independent_source_count >= min_independent_sources
    ]
    metrics = eligible or metrics  # never return nothing at all

    items_by_concept: dict[int, set[str]] = {}
    for m in metrics:
        rows = db.conn.execute(
            "SELECT DISTINCT source_item_id FROM concept_mentions WHERE concept_id=?",
            (m.concept_id,),
        ).fetchall()
        items_by_concept[m.concept_id] = {r["source_item_id"] for r in rows}

    ordered = sorted(metrics, key=lambda m: m.insider_momentum, reverse=True)
    groups: list[ConceptGroup] = []
    assigned: set[int] = set()

    for m in ordered:
        if m.concept_id in assigned:
            continue
        group = ConceptGroup(
            key=f"grp_{m.concept_id}",
            concept_ids=[m.concept_id],
            labels=[m.concept_label],
            metrics=[m],
        )
        assigned.add(m.concept_id)
        mine = items_by_concept[m.concept_id]

        for other in ordered:
            if other.concept_id in assigned:
                continue
            theirs = items_by_concept[other.concept_id]
            if not mine or not theirs:
                continue
            # Jaccard over the documents mentioning each concept.
            overlap = len(mine & theirs) / len(mine | theirs)
            if overlap >= overlap_threshold:
                group.concept_ids.append(other.concept_id)
                group.labels.append(other.concept_label)
                group.metrics.append(other)
                assigned.add(other.concept_id)

        group.item_ids = sorted(
            set().union(*(items_by_concept[c] for c in group.concept_ids))
        )
        groups.append(group)
        if len(groups) >= max_groups:
            break

    return groups
