"""The trend engine: what is actually moving.

Everything here is deterministic arithmetic over observed data. No LLM is
consulted about what is trending, because a model asked "what's hot?" answers
from its training distribution, not from this week.

Two rules the rest of the system depends on:

1. A component that cannot be computed is None, never 0. "No baseline yet" and
   "not accelerating" are completely different states, and collapsing them
   would rank a brand-new concept as dead. Missing components are excluded and
   the remaining weights renormalise.

2. Ratios are log-squashed. Without it a single 40x spike on a tiny base
   dominates every ranking, which is how trend systems end up recommending
   noise.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import Settings
from ..db import Database
from ..models import (
    INSUFFICIENT_HISTORY,
    NOT_AVAILABLE,
    ConceptMetrics,
    SignalPropagation,
    SourceRole,
    TrendComponents,
)
from .independence import assign_clusters, compute_independence
from .registry import ConceptRegistry

PLATFORMS = ("substack", "hacker_news", "arxiv", "company_blog", "x")
ROLE_BY_PLATFORM = {
    "substack": SourceRole.INDEPENDENT_ANALYSIS,
    "hacker_news": SourceRole.COMMUNITY_REACTION,
    "arxiv": SourceRole.PRIMARY_EVIDENCE,
    "company_blog": SourceRole.PRIMARY_EVIDENCE,
    "x": SourceRole.EARLY_SIGNAL,
}


def squash(ratio: float | None, cap: float = 10.0) -> float | None:
    """Compress a ratio into [0,1] logarithmically."""
    if ratio is None:
        return None
    if ratio <= 0:
        return 0.0
    return min(1.0, math.log1p(ratio) / math.log1p(cap))


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


class TrendEngine:
    def __init__(self, db: Database, settings: Settings, registry: ConceptRegistry | None = None):
        self.db = db
        self.settings = settings
        self.registry = registry or ConceptRegistry(db)
        self.scoring = settings.section("scoring", {})
        self.weights = settings.weights or {}

    # -- public API --------------------------------------------------------

    def compute_all(
        self, *, run_id: str | None = None, now: datetime | None = None
    ) -> list[ConceptMetrics]:
        """Score every tracked concept, best first."""
        now = now or datetime.now(timezone.utc)
        min_mentions = int(self.settings.pipeline.get("min_mentions_to_track", 2))
        metrics: list[ConceptMetrics] = []

        for concept_id in self.registry.concept_ids:
            m = self.compute_concept(concept_id, now=now)
            if m.mentions_30d < min_mentions:
                continue
            metrics.append(m)
            if run_id:
                self.db.save_metrics(m, run_id)

        metrics.sort(key=lambda m: m.insider_momentum, reverse=True)
        return metrics

    def compute_concept(self, concept_id: int, *, now: datetime | None = None) -> ConceptMetrics:
        now = now or datetime.now(timezone.utc)
        rows = self.db.conn.execute(
            "SELECT m.source_item_id, m.timestamp, m.author_id, s.source_type,"
            " s.publication_id, s.source_role, p.curated_tier, a.curated_tier AS author_tier"
            " FROM concept_mentions m"
            " JOIN source_items s ON s.id = m.source_item_id"
            " LEFT JOIN publications p ON p.id = s.publication_id"
            " LEFT JOIN authors a ON a.id = s.author_id"
            " WHERE m.concept_id = ? AND m.timestamp IS NOT NULL",
            (concept_id,),
        ).fetchall()

        m = ConceptMetrics(
            concept_id=concept_id,
            concept_label=self.registry.label(concept_id),
            status=self.registry.status(concept_id),
        )
        if not rows:
            m.notes.append("no dated mentions")
            return m

        # One item may mention a concept several times; windows count items.
        by_item: dict[str, dict[str, Any]] = {}
        for row in rows:
            ts = _parse(row["timestamp"])
            if ts is None:
                continue
            existing = by_item.get(row["source_item_id"])
            if existing is None:
                by_item[row["source_item_id"]] = {
                    "ts": ts,
                    "author_id": row["author_id"],
                    "publication_id": row["publication_id"],
                    "source_type": row["source_type"],
                    "source_role": row["source_role"],
                    "tier": row["curated_tier"] or row["author_tier"],
                }
        if not by_item:
            m.notes.append("no parseable timestamps")
            return m

        def window(days: float) -> list[dict[str, Any]]:
            cutoff = now - timedelta(days=days)
            return [v for v in by_item.values() if v["ts"] >= cutoff]

        w24, w7, w30 = window(1), window(7), window(30)
        prev7 = [
            v for v in by_item.values()
            if now - timedelta(days=14) <= v["ts"] < now - timedelta(days=7)
        ]

        m.mentions_24h, m.mentions_7d, m.mentions_30d = len(w24), len(w7), len(w30)
        m.unique_authors_24h = len({v["author_id"] for v in w24 if v["author_id"]})
        m.unique_authors_7d = len({v["author_id"] for v in w7 if v["author_id"]})
        m.unique_authors_prev_7d = len({v["author_id"] for v in prev7 if v["author_id"]})
        m.unique_publications_7d = len({v["publication_id"] for v in w7 if v["publication_id"]})

        # -- independence, over the 30-day window --------------------------
        recent_ids = [k for k, v in by_item.items() if v["ts"] >= now - timedelta(days=30)]
        indep = compute_independence(self.db, recent_ids)
        m.independent_source_count = indep.independent_source_count
        m.independence_ratio = indep.independence_ratio
        if indep.is_single_origin_cascade:
            m.notes.append(
                f"single-origin cascade: {indep.distinct_authors} authors collapse to "
                f"{indep.independent_source_count} independent sources "
                f"({'; '.join(indep.join_reasons[:3])})"
            )
        assign_clusters(self.db, concept_id, indep)

        # -- baseline and velocity -----------------------------------------
        baseline_days = int(self.scoring.get("baseline_window_days", 28))
        exclude_recent = int(self.scoring.get("baseline_exclude_recent_days", 3))
        min_baseline = int(self.scoring.get("min_baseline_days", 14))

        daily: defaultdict[str, int] = defaultdict(int)
        for v in by_item.values():
            daily[v["ts"].date().isoformat()] += 1

        baseline_start = now - timedelta(days=baseline_days + exclude_recent)
        baseline_end = now - timedelta(days=exclude_recent)
        baseline_values = [
            daily.get((baseline_start + timedelta(days=i)).date().isoformat(), 0)
            for i in range((baseline_end - baseline_start).days)
        ]
        observed_span = (now - min(v["ts"] for v in by_item.values())).days
        m.baseline_days_available = min(observed_span, len(baseline_values))

        if m.baseline_days_available >= min_baseline and baseline_values:
            m.baseline_daily = float(statistics.median(baseline_values))
            m.velocity_ratio = m.mentions_24h / max(m.baseline_daily, 0.5)
        else:
            m.notes.append(
                f"{INSUFFICIENT_HISTORY}: {m.baseline_days_available}d of history, "
                f"{min_baseline}d needed for a velocity baseline"
            )

        # 7-day rate vs the 7 days before it.
        if m.mentions_7d or len(prev7):
            m.acceleration = (m.mentions_7d - len(prev7)) / max(len(prev7), 1)

        # -- engagement ----------------------------------------------------
        m.engagement_sum, m.engagement_median, m.engagement_acceleration = self._engagement(
            list(by_item), now
        )

        # -- cross-platform ------------------------------------------------
        platforms = {v["source_type"] for v in w7}
        roles = {v["source_role"] for v in w7}
        m.cross_platform_count = len(platforms)
        m.cross_role_count = len(roles)

        tiered = [v for v in w30 if v["tier"] and int(v["tier"]) <= 2]
        m.respected_source_share = len(tiered) / len(w30) if w30 else 0.0

        m.paper_count = sum(1 for v in w30 if v["source_type"] == "arxiv")

        # Google Trends is alpha and application-gated. Absent access this stays
        # NOT_AVAILABLE rather than being silently swapped for another metric --
        # insider attention and public interest are different signals.
        m.google_search_change = NOT_AVAILABLE

        # -- components ----------------------------------------------------
        cap = float(self.scoring.get("velocity_cap", 10.0))
        comp = TrendComponents(
            normalized_velocity=squash(m.velocity_ratio, cap),
            unique_author_growth=squash(
                m.unique_authors_7d / max(m.unique_authors_prev_7d, 1), cap
            ) if (m.unique_authors_7d or m.unique_authors_prev_7d) else None,
            cross_platform_presence=self._cross_platform_score(platforms, roles),
            respected_source_presence=m.respected_source_share,
            engagement_acceleration=squash(m.engagement_acceleration, cap),
            novelty=self._novelty(concept_id, now),
        )
        m.components = comp

        available = comp.available()
        if available:
            total_weight = sum(self.weights.get(k, 0.0) for k in available) or 1.0
            m.insider_momentum_raw = sum(
                self.weights.get(k, 0.0) * v for k, v in available.items()
            ) / total_weight
        m.data_completeness = len(available) / max(len(comp.model_dump()), 1)
        if len(available) < len(comp.model_dump()):
            missing = [k for k, v in comp.model_dump().items() if v is None]
            m.missing_sources = missing
            m.notes.append(f"components unavailable (weights renormalised): {', '.join(missing)}")

        exponent = float(self.scoring.get("independence_exponent", 0.5))
        m.insider_momentum = m.insider_momentum_raw * (m.independence_ratio ** exponent)
        return m

    # -- helpers -----------------------------------------------------------

    def _cross_platform_score(self, platforms: set[str], roles: set[str]) -> float:
        """Reward propagation across communities, and across kinds of evidence.

        Roles are weighted more than raw platform count: an early signal plus
        primary evidence plus community reaction is a propagation chain, while
        three early signals is an echo.
        """
        platform_part = len(platforms & set(PLATFORMS)) / len(PLATFORMS)
        role_part = len(roles) / 4.0
        return min(1.0, 0.6 * platform_part + 0.4 * role_part)

    def _novelty(self, concept_id: int, now: datetime) -> float | None:
        """How new this concept is to OUR corpus.

        Uses our own first observation. External search APIs cap results per
        response, so their apparent 'first seen' is an artifact of pagination.
        """
        row = self.db.conn.execute(
            "SELECT first_observed_at FROM concepts WHERE id=?", (concept_id,)
        ).fetchone()
        if not row or not row["first_observed_at"]:
            return None
        first = _parse(row["first_observed_at"])
        if first is None:
            return None
        horizon = float(self.scoring.get("novelty_horizon_days", 180))
        age_days = max(0.0, (now - first).total_seconds() / 86400)
        return max(0.0, min(1.0, 1 - age_days / horizon))

    def _engagement(
        self, item_ids: list[str], now: datetime
    ) -> tuple[float, float, float | None]:
        if not item_ids:
            return 0.0, 0.0, None
        marks = ",".join("?" * len(item_ids))
        rows = self.db.conn.execute(
            f"SELECT e.source_item_id, e.observed_at,"
            f" COALESCE(e.score,0)+COALESCE(e.likes,0)+COALESCE(e.reposts,0)"
            f" +COALESCE(e.comment_count,0)+COALESCE(e.replies,0) AS total,"
            f" s.published_at FROM engagement_snapshots e"
            f" JOIN source_items s ON s.id = e.source_item_id"
            f" WHERE e.source_item_id IN ({marks})",
            item_ids,
        ).fetchall()
        if not rows:
            return 0.0, 0.0, None

        latest: dict[str, tuple[float, datetime | None]] = {}
        for row in rows:
            published = _parse(row["published_at"])
            total = float(row["total"] or 0)
            key = row["source_item_id"]
            if key not in latest or total > latest[key][0]:
                latest[key] = (total, published)

        totals = [t for t, _ in latest.values()]
        eng_sum = float(sum(totals))
        eng_median = float(statistics.median(totals)) if totals else 0.0

        recent = [t for t, p in latest.values() if p and p >= now - timedelta(days=7)]
        prior = [
            t for t, p in latest.values()
            if p and now - timedelta(days=35) <= p < now - timedelta(days=7)
        ]
        accel = None
        if recent and prior:
            prior_median = statistics.median(prior)
            accel = statistics.median(recent) / max(prior_median, 0.5)
        return eng_sum, eng_median, accel

    # -- propagation -------------------------------------------------------

    def propagation(self, concept_id: int, *, days: int = 60) -> SignalPropagation:
        """Reconstruct how a concept moved between platforms.

        The pattern worth finding: paper -> X -> newsletters -> HN, with
        YouTube still empty. That shape is an editorial opportunity.
        """
        cutoff = _iso(datetime.now(timezone.utc) - timedelta(days=days))
        rows = self.db.conn.execute(
            "SELECT m.source_item_id, m.timestamp, s.source_type, s.author_id, s.title, s.url"
            " FROM concept_mentions m JOIN source_items s ON s.id = m.source_item_id"
            " WHERE m.concept_id=? AND m.timestamp >= ? ORDER BY m.timestamp ASC",
            (concept_id, cutoff),
        ).fetchall()
        prop = SignalPropagation(concept_id=concept_id)
        if not rows:
            return prop

        seen: set[str] = set()
        for row in rows:
            platform = row["source_type"]
            if platform in seen:
                continue
            seen.add(platform)
            prop.platform_sequence.append(
                {
                    "platform": platform,
                    "first_seen": row["timestamp"],
                    "item_id": row["source_item_id"],
                    "title": (row["title"] or "")[:120],
                    "url": row["url"],
                }
            )

        first = rows[0]
        prop.origin_source_item_id = first["source_item_id"]
        prop.origin_date = _parse(first["timestamp"])
        prop.independent_author_count = len({r["author_id"] for r in rows if r["author_id"]})

        if len(prop.platform_sequence) > 1:
            start = _parse(prop.platform_sequence[0]["first_seen"])
            second = _parse(prop.platform_sequence[1]["first_seen"])
            if start and second:
                prop.time_to_cross_platform_hours = (second - start).total_seconds() / 3600
        return prop


def _parse(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
