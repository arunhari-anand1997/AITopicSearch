"""Disagreement as a query, not a guess.

Today the Disagreement Scout skims the ~40 excerpts that fit in a prompt and
forms a view. Over a claim store the same question becomes a database query
across EVERY article, and the answer is auditable: each side of a dispute comes
back with the source IDs, authors and verbatim quotes behind it.

Three things this finds that prose-skimming cannot:

1. DISPUTES  -- independent authors taking opposing stances on the same concept.
2. REVERSALS -- one author changing their own mind over time. Editorially
   valuable and completely invisible before now.
3. SURPRISES -- claims that explicitly contradict what their own author
   presents as conventional wisdom.

Independence is enforced throughout: two posts by the same author are one
voice, and a "dispute" needs genuinely different people on each side.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..db import Database

OPPOSING = {"advocating": "skeptical", "skeptical": "advocating"}


@dataclass
class Side:
    stance: str
    authors: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    rationales: list[str] = field(default_factory=list)

    @property
    def author_count(self) -> int:
        return len(set(self.authors))


@dataclass
class Dispute:
    concept_id: int
    concept_label: str
    sides: list[Side] = field(default_factory=list)

    @property
    def is_genuine(self) -> bool:
        """Both sides must have real, distinct people on them.

        The brief forbids manufacturing balance, so a 'dispute' where one side
        is a single voice is reported as such rather than dressed up as a split.
        """
        return len(self.sides) >= 2 and all(s.author_count >= 2 for s in self.sides)

    def render(self) -> str:
        lines = [f"### {self.concept_label}"]
        for side in self.sides:
            lines.append(
                f"- **{side.stance}** — {side.author_count} distinct authors "
                f"({', '.join(sorted(set(side.authors))[:4])})"
            )
            for rationale in side.rationales[:2]:
                if rationale:
                    lines.append(f"  - {rationale[:190]}")
        if not self.is_genuine:
            lines.append("  - NOTE: one-sided; not yet a genuine split.")
        return "\n".join(lines)


@dataclass
class Reversal:
    author: str
    concept_label: str
    from_stance: str
    to_stance: str
    from_date: str
    to_date: str
    from_source: str
    to_source: str
    rationale: str = ""

    def render(self) -> str:
        return (
            f"- **{self.author}** moved from *{self.from_stance}* "
            f"({self.from_date[:10]}) to *{self.to_stance}* ({self.to_date[:10]}) "
            f"on {self.concept_label}"
            + (f" — {self.rationale[:160]}" if self.rationale else "")
        )


def find_disputes(
    db: Database, *, concept_id: int | None = None, days: int = 120, min_authors: int = 2
) -> list[Dispute]:
    """Concepts where independent authors hold opposing positions."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    sql = """
        SELECT p.concept_id, c.canonical_label, p.stance, p.rationale,
               p.source_item_id, COALESCE(a.handle, a.display_name, 'unknown') author
        FROM article_positions p
        JOIN concepts c ON c.id = p.concept_id
        LEFT JOIN authors a ON a.id = p.author_id
        WHERE p.stance IN ('advocating','skeptical')
          AND COALESCE(p.observed_at,'') >= ?
    """
    params: list[Any] = [cutoff]
    if concept_id is not None:
        sql += " AND p.concept_id = ?"
        params.append(concept_id)

    grouped: defaultdict[tuple[int, str], defaultdict[str, Side]] = defaultdict(
        lambda: defaultdict(lambda: Side(stance=""))
    )
    for row in db.conn.execute(sql, params).fetchall():
        key = (row["concept_id"], row["canonical_label"])
        side = grouped[key][row["stance"]]
        side.stance = row["stance"]
        side.authors.append(row["author"])
        side.source_ids.append(row["source_item_id"])
        if row["rationale"]:
            side.rationales.append(row["rationale"])

    disputes: list[Dispute] = []
    for (cid, label), sides in grouped.items():
        if len(sides) < 2:
            continue
        dispute = Dispute(concept_id=cid, concept_label=label, sides=list(sides.values()))
        if any(s.author_count >= min_authors for s in dispute.sides):
            disputes.append(dispute)

    disputes.sort(
        key=lambda d: min((s.author_count for s in d.sides), default=0), reverse=True
    )
    return disputes


def find_reversals(
    db: Database, *, days: int = 400, min_days_between: int = 21
) -> list[Reversal]:
    """Authors who changed their own mind.

    Two paths: an explicit self-correction flagged during extraction, or an
    observed stance flip on the same concept over time.

    Two guards, both learned from false positives on live data:

    - ORGANISATIONS ARE EXCLUDED. A company blog is many writers under one
      byline, so two posts taking different stances is not one mind changing.
      Live data produced "Anthropic moved from skeptical to advocating on
      evals" between two posts one day apart, which is an artifact, not a
      reversal.
    - A MINIMUM GAP is required. Changing your mind takes longer than a day;
      anything faster is two different pieces on different aspects, or an
      extraction wobble.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    rows = db.conn.execute(
        """
        SELECT p.concept_id, c.canonical_label, p.stance, p.rationale,
               p.observed_at, p.source_item_id,
               COALESCE(a.handle, a.display_name) author
        FROM article_positions p
        JOIN concepts c ON c.id = p.concept_id
        JOIN authors a ON a.id = p.author_id
        JOIN source_items s ON s.id = p.source_item_id
        WHERE p.stance IN ('advocating','skeptical')
          AND COALESCE(p.observed_at,'') >= ?
          AND s.primary_or_secondary <> 'PRIMARY'
          AND s.source_type <> 'company_blog'
        ORDER BY p.author_id, p.concept_id, p.observed_at
        """,
        (cutoff,),
    ).fetchall()

    by_author: defaultdict[tuple[str, int], list[Any]] = defaultdict(list)
    for row in rows:
        if row["author"]:
            by_author[(row["author"], row["concept_id"])].append(row)

    reversals: list[Reversal] = []
    for (author, _cid), entries in by_author.items():
        if len(entries) < 2:
            continue
        first, last = entries[0], entries[-1]
        if OPPOSING.get(first["stance"]) != last["stance"]:
            continue
        gap = _gap_days(first["observed_at"], last["observed_at"])
        if gap is not None and gap < min_days_between:
            continue
        if True:
            reversals.append(
                Reversal(
                    author=author,
                    concept_label=first["canonical_label"],
                    from_stance=first["stance"],
                    to_stance=last["stance"],
                    from_date=str(first["observed_at"] or ""),
                    to_date=str(last["observed_at"] or ""),
                    from_source=first["source_item_id"],
                    to_source=last["source_item_id"],
                    rationale=last["rationale"] or "",
                )
            )

    explicit = db.conn.execute(
        """
        SELECT cl.proposition, cl.quote, s.published_at, cl.source_item_id,
               COALESCE(a.handle, a.display_name) author
        FROM article_claims cl
        JOIN source_items s ON s.id = cl.source_item_id
        LEFT JOIN authors a ON a.id = s.author_id
        WHERE cl.is_self_correction = 1 AND COALESCE(s.published_at,'') >= ?
        ORDER BY s.published_at DESC LIMIT 25
        """,
        (cutoff,),
    ).fetchall()
    for row in explicit:
        reversals.append(
            Reversal(
                author=row["author"] or "unknown",
                concept_label="(stated self-correction)",
                from_stance="previous view",
                to_stance="revised view",
                from_date="",
                to_date=str(row["published_at"] or ""),
                from_source=row["source_item_id"],
                to_source=row["source_item_id"],
                rationale=row["proposition"],
            )
        )
    return reversals


def find_surprises(db: Database, *, concept_id: int | None = None, days: int = 90,
                   limit: int = 15) -> list[Any]:
    """Claims that explicitly contradict their author's stated conventional view."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    sql = """
        SELECT cl.proposition, cl.quote, cl.evidence_type, cl.subject,
               s.title, s.url, s.published_at, s.source_name,
               COALESCE(a.handle, a.display_name) author
        FROM article_claims cl
        JOIN source_items s ON s.id = cl.source_item_id
        LEFT JOIN authors a ON a.id = s.author_id
        WHERE cl.is_surprising = 1 AND COALESCE(s.published_at,'') >= ?
    """
    params: list[Any] = [cutoff]
    if concept_id is not None:
        sql += " AND cl.concept_id = ?"
        params.append(concept_id)
    sql += " ORDER BY s.published_at DESC LIMIT ?"
    params.append(limit)
    return db.conn.execute(sql, params).fetchall()


def render_for_dossier(db: Database, concept_id: int, *, days: int = 120) -> str:
    """The claim-layer section of a scout's dossier.

    This is where the layer pays for itself: instead of forty excerpts chosen by
    recency, the scout gets the distilled positions of every article that
    engaged with the concept, with the disagreement already located.
    """
    parts: list[str] = []

    disputes = find_disputes(db, concept_id=concept_id, days=days)
    if disputes:
        parts.append("## DISAGREEMENT FOUND IN THE CLAIM RECORD")
        parts.append(
            "Computed across every extracted article, not sampled. Each side "
            "lists the distinct authors holding it."
        )
        parts.append("")
        for dispute in disputes[:3]:
            parts.append(dispute.render())
            parts.append("")

    surprises = find_surprises(db, concept_id=concept_id, days=days, limit=6)
    if surprises:
        parts.append("## CLAIMS THAT CONTRADICT CONVENTIONAL WISDOM")
        parts.append("")
        for row in surprises:
            parts.append(
                f"- {row['proposition']} "
                f"({row['author'] or row['source_name']}, "
                f"{str(row['published_at'])[:10]}, evidence: {row['evidence_type']})"
            )
        parts.append("")

    # A claim resolves to a concept either directly (its own wording matched an
    # alias) or by provenance (it came from an article that mentions the
    # concept). Direct matching alone resolves only about a third of claims,
    # which would leave most of the layer invisible to the dossier.
    claims = db.conn.execute(
        """
        SELECT DISTINCT cl.id, cl.proposition, cl.stance, cl.evidence_type,
               cl.hedging, cl.concept_id,
               COALESCE(a.handle, a.display_name) author, s.published_at
        FROM article_claims cl
        JOIN source_items s ON s.id = cl.source_item_id
        LEFT JOIN authors a ON a.id = s.author_id
        WHERE COALESCE(s.published_at,'') >= ?
          AND (cl.concept_id = ?
               OR EXISTS (SELECT 1 FROM concept_mentions m
                          WHERE m.source_item_id = cl.source_item_id
                            AND m.concept_id = ?))
        ORDER BY (cl.concept_id = ?) DESC,
                 (cl.evidence_type IN ('benchmark','measurement')) DESC,
                 s.published_at DESC
        LIMIT 30
        """,
        (
            (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
            concept_id, concept_id, concept_id,
        ),
    ).fetchall()
    if claims:
        parts.append("## WHAT ARTICLES ACTUALLY CLAIMED (measured evidence first)")
        parts.append("")
        for row in claims:
            parts.append(
                f"- [{row['stance']}/{row['hedging']}/{row['evidence_type']}] "
                f"{row['proposition']} "
                f"— {row['author'] or 'unknown'}, {str(row['published_at'])[:10]}"
            )
        parts.append("")

    return "\n".join(parts)


def _gap_days(first: Any, last: Any) -> float | None:
    """Days between two observations, or None if either date is unusable."""
    a, b = _parse(str(first or "")), _parse(str(last or ""))
    if a is None or b is None:
        return None
    return abs((b - a).total_seconds()) / 86400


def _parse(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None
