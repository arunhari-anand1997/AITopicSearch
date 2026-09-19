"""Google Trends via SerpAPI -- the PUBLIC_INTEREST signal.

WHY THIS EXISTS
---------------
The official Google Trends API is alpha and application-gated, so the system
has been reporting public interest as NOT_AVAILABLE. SerpAPI is a commercial
data provider that supplies the same Trends data through its own infrastructure,
which closes that gap without us scraping anything.

PROVENANCE HONESTY
------------------
This is NOT the official Google Trends API, and the difference is recorded
rather than glossed. Every value is labelled `serpapi_google_trends` so a brief
never implies we have first-party Google data. The numbers are Google's
relative-interest index (0-100, scaled within a request), which means values
are comparable ACROSS TIME for one query but not across separate queries.

WHY IT IS KEPT SEPARATE FROM MOMENTUM
-------------------------------------
Public interest and insider attention are different signals, and conflating
them would defeat the whole coverage-gap thesis. A concept that insiders are
converging on while public search interest stays flat is exactly the editorial
opportunity we are hunting. So this feeds `google_search_change` and the
public-confusion dimension -- never INSIDER_MOMENTUM.

BUDGET
------
SerpAPI bills per search against a monthly quota. The connector checks the
remaining balance, refuses to exceed a configured ceiling, and caches results
for days, because weekly Trends data does not change hour to hour.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..models import ConnectorStatus
from .base import Connector

API = "https://serpapi.com/search.json"
ACCOUNT = "https://serpapi.com/account"

#: How the value is labelled everywhere downstream.
SOURCE_LABEL = "serpapi_google_trends"


@dataclass
class TrendsResult:
    """Public-interest reading for one query."""

    query: str
    status: ConnectorStatus
    #: Relative interest (0-100) over the trailing window, oldest first.
    timeline: list[tuple[str, int]] | None = None
    change_7d_vs_90d: float | None = None
    direction: str | None = None
    source: str = SOURCE_LABEL
    error: str | None = None
    searches_used: int = 0

    def summary(self) -> str:
        """One line for the brief. Never invents a number it does not have."""
        if self.status != ConnectorStatus.OK or not self.timeline:
            return f"NOT_AVAILABLE ({self.error or self.status})"
        latest = self.timeline[-1][1]
        if self.change_7d_vs_90d is None:
            return f"{latest}/100 latest ({self.source})"
        return (
            f"{latest}/100 latest, {self.direction} "
            f"({self.change_7d_vs_90d:+.0%} vs 90d average, {self.source})"
        )


class SerpApiTrendsConnector(Connector):
    name = "google_trends"
    min_interval = 1.0

    def __init__(self, settings, client=None):
        super().__init__(settings, client)
        self.config = settings.section("google_trends", {}) or {}
        self.searches_used = 0
        self._cache_dir = Path(settings.cache_dir) / "serpapi"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False)) and bool(self.settings.serpapi_key)

    @property
    def max_searches_per_run(self) -> int:
        return int(self.config.get("max_searches_per_run", 12))

    @property
    def cache_days(self) -> int:
        # Trends is weekly data; re-fetching daily just burns quota.
        return int(self.config.get("cache_days", 3))

    # -- quota -------------------------------------------------------------

    async def remaining_searches(self) -> int | None:
        """Ask SerpAPI how much quota is left. This call is itself free."""
        if not self.settings.serpapi_key:
            return None
        try:
            resp = await self.fetch(
                ACCOUNT, params={"api_key": self.settings.serpapi_key}
            )
            resp.raise_for_status()
            return int(resp.json().get("total_searches_left", 0))
        except Exception:  # noqa: BLE001 - quota check must never break a run
            return None

    # -- cache -------------------------------------------------------------

    def _cache_path(self, query: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in query)[:60]
        return self._cache_dir / f"{safe}.json"

    def _cached(self, query: str) -> TrendsResult | None:
        path = self._cache_path(query)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        fetched = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now(timezone.utc) - fetched > timedelta(days=self.cache_days):
            return None
        return TrendsResult(
            query=query,
            status=ConnectorStatus.OK,
            timeline=[tuple(t) for t in payload["timeline"]],
            change_7d_vs_90d=payload.get("change"),
            direction=payload.get("direction"),
        )

    def _store(self, result: TrendsResult) -> None:
        if result.status != ConnectorStatus.OK or not result.timeline:
            return
        self._cache_path(result.query).write_text(
            json.dumps(
                {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "timeline": result.timeline,
                    "change": result.change_7d_vs_90d,
                    "direction": result.direction,
                },
                default=str,
            )
        )

    # -- fetch -------------------------------------------------------------

    async def interest(self, query: str) -> TrendsResult:
        """Public search interest for one query."""
        if not self.settings.serpapi_key:
            return TrendsResult(
                query, ConnectorStatus.NOT_CONFIGURED,
                error="SERPAPI_KEY not set",
            )
        if not self.config.get("enabled", False):
            return TrendsResult(
                query, ConnectorStatus.NOT_CONFIGURED,
                error="google_trends.enabled is false in config/editorial.yaml",
            )

        cached = self._cached(query)
        if cached is not None:
            return cached

        if self.searches_used >= self.max_searches_per_run:
            return TrendsResult(
                query, ConnectorStatus.BUDGET_EXHAUSTED,
                error=f"per-run ceiling of {self.max_searches_per_run} SerpAPI searches reached",
            )

        try:
            self.searches_used += 1
            resp = await self.fetch(
                API,
                params={
                    "engine": "google_trends",
                    "q": query,
                    "data_type": "TIMESERIES",
                    "api_key": self.settings.serpapi_key,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            return TrendsResult(
                query, ConnectorStatus.SOURCE_UNAVAILABLE,
                error=f"{type(exc).__name__}: {exc}", searches_used=1,
            )

        if payload.get("error"):
            return TrendsResult(
                query, ConnectorStatus.SOURCE_UNAVAILABLE,
                error=str(payload["error"]), searches_used=1,
            )

        points = payload.get("interest_over_time", {}).get("timeline_data", []) or []
        timeline: list[tuple[str, int]] = []
        for point in points:
            values = point.get("values") or []
            if not values:
                continue
            raw = values[0].get("extracted_value")
            if raw is None:
                continue
            timeline.append((str(point.get("date", "")), int(raw)))

        if not timeline:
            return TrendsResult(
                query, ConnectorStatus.SOURCE_UNAVAILABLE,
                error="no timeline data returned", searches_used=1,
            )

        result = TrendsResult(
            query=query, status=ConnectorStatus.OK, timeline=timeline, searches_used=1
        )
        result.change_7d_vs_90d, result.direction = _trajectory(timeline)
        self._store(result)
        return result

    async def collect(self) -> list[Any]:
        """Not part of routine collection.

        Trends is queried per gated concept after the momentum gate, so that a
        scarce paid quota is spent only on concepts already worth attention.
        """
        return [
            self.result(
                "google_trends", ConnectorStatus.NOT_CONFIGURED,
                error="queried per-concept after the gate, not during collection",
            )
        ]


def _trajectory(timeline: list[tuple[str, int]]) -> tuple[float | None, str | None]:
    """Recent interest against the longer-run average.

    Deliberately coarse. This is a public-attention sanity check, not a
    precision instrument, and over-reading a 0-100 relative index is how people
    talk themselves into trends that are not there.
    """
    if len(timeline) < 8:
        return None, None
    values = [v for _, v in timeline]
    recent = values[-4:]
    baseline = values[:-4] or values
    recent_avg = sum(recent) / len(recent)
    baseline_avg = sum(baseline) / len(baseline)
    if baseline_avg <= 0:
        return None, None
    change = (recent_avg - baseline_avg) / baseline_avg
    if change > 0.25:
        direction = "rising"
    elif change < -0.25:
        direction = "falling"
    else:
        direction = "flat"
    return change, direction
