"""Collection orchestration.

Runs every enabled connector, persists what comes back, and records per-source
availability so that a failure is visible in the database rather than looking
like an empty result set.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx

from ..config import USER_AGENT, Settings
from ..db import Database
from ..models import CollectionResult, ConnectorStatus
from .arxiv import ArxivConnector
from .hacker_news import HackerNewsConnector, HNSearchConnector
from .primary_blogs import PrimaryBlogConnector
from .substack import SubstackConnector
from .x_api import XConnector


@dataclass
class CollectionSummary:
    results: list[CollectionResult] = field(default_factory=list)
    new_items: int = 0
    seen_items: int = 0

    @property
    def failures(self) -> list[CollectionResult]:
        return [r for r in self.results if not r.ok]

    @property
    def available_sources(self) -> set[str]:
        return {r.connector for r in self.results if r.ok and r.items}

    def completeness(self, expected: set[str]) -> float:
        """Fraction of the expected sources that actually delivered.

        Downstream scores carry this so a thin run is never mistaken for a
        quiet week.
        """
        if not expected:
            return 1.0
        return len(self.available_sources & expected) / len(expected)

    def table(self) -> list[tuple[str, str, str, int]]:
        return [
            (r.connector, r.target, str(r.status), len(r.items)) for r in self.results
        ]


CORE_CONNECTORS = (
    SubstackConnector,
    HackerNewsConnector,
    ArxivConnector,
    PrimaryBlogConnector,
)


async def collect_all(
    settings: Settings,
    db: Database,
    run_id: str,
    *,
    include_backfill: bool = False,
    only: set[str] | None = None,
) -> CollectionSummary:
    """Run the connectors and persist everything they return."""
    summary = CollectionSummary()

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(45.0),
        follow_redirects=True,
    ) as client:
        connectors = [cls(settings, client) for cls in CORE_CONNECTORS]
        if include_backfill:
            connectors.append(HNSearchConnector(settings, client))
        # X is opt-in and costs real money per returned post; the connector
        # itself refuses to run without an explicit budget ceiling.
        if (settings.x_config or {}).get("enabled", False):
            connectors.append(XConnector(settings, client))
        if only:
            connectors = [c for c in connectors if c.name in only]

        async def run(conn) -> list[CollectionResult]:
            async with conn:
                try:
                    return await conn.collect()
                except Exception as exc:  # noqa: BLE001 - never let one source kill the run
                    return [
                        CollectionResult(
                            connector=conn.name, target="*",
                            status=ConnectorStatus.SOURCE_UNAVAILABLE,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    ]

        batches = await asyncio.gather(*(run(c) for c in connectors))

    for batch in batches:
        for result in batch:
            new = 0
            for item in result.items:
                tier = item.metadata.get("curated_tier")
                if db.insert_source_item(item, run_id=run_id, curated_tier=tier):
                    new += 1
            summary.new_items += new
            summary.seen_items += len(result.items)
            db.record_fetch(result, run_id, new)
            summary.results.append(result)

    return summary
