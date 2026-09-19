"""YouTube coverage connector.

YouTube measures EXISTING COVERAGE. It never measures insider attention, and
nothing downstream treats it as such.

QUOTA IS THE DESIGN CONSTRAINT. `search.list` is capped at 100 calls per day as
a budget SEPARATE from the 10,000-unit pool, while `videos.list` costs 1 unit
from that pool. So search is scarce and enrichment is nearly free, which is why
coverage checks run only on concepts that already passed the momentum gate, and
why results are cached for days rather than re-fetched.

Coverage QUALITY is judged semantically by the Coverage Scout, never here.
Counting results would tell us that a hundred tutorials exist, which says
nothing about whether an intelligent non-specialist has ever had this explained.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from ..models import CollectionResult, ConnectorStatus, SourceItem, SourceRole, SourceType
from .base import Connector, parse_date

API = "https://www.googleapis.com/youtube/v3"
ISO_DURATION_UNITS = (("H", 3600), ("M", 60), ("S", 1))


def parse_duration(iso: str | None) -> int | None:
    """Parse ISO-8601 durations like PT12M34S into seconds."""
    if not iso or not iso.startswith("PT"):
        return None
    total, number = 0, ""
    for char in iso[2:]:
        if char.isdigit():
            number += char
            continue
        for unit, factor in ISO_DURATION_UNITS:
            if char == unit and number:
                total += int(number) * factor
        number = ""
    return total or None


class YouTubeConnector(Connector):
    name = "youtube"
    min_interval = 0.2

    def __init__(self, settings, client=None):
        super().__init__(settings, client)
        self.config = settings.youtube_queries or {}
        self.runtime = settings.section("youtube", {}) or {}
        self.searches_used = 0

    @property
    def search_budget(self) -> int:
        return int(self.runtime.get("max_search_calls_per_day", 60))

    async def collect(self) -> list[CollectionResult]:
        """Standing queries only; per-concept coverage runs after the gate."""
        queries = self.config.get("standing_queries") or []
        if not self.settings.youtube_api_key:
            return [
                self.result(
                    "youtube", ConnectorStatus.NOT_CONFIGURED,
                    error="YOUTUBE_API_KEY not set; coverage analysis is unavailable "
                          "and coverage-gap scores will be marked unknown rather than guessed",
                )
            ]
        results = []
        for query in queries:
            results.append(await self.coverage_for(query))
        return results

    async def coverage_for(
        self, query: str, *, orders: list[str] | None = None, cache_days: int | None = None
    ) -> CollectionResult:
        """Search several orderings, then enrich with real statistics."""
        if not self.settings.youtube_api_key:
            return self.result(
                f"query:{query}", ConnectorStatus.NOT_CONFIGURED,
                error="YOUTUBE_API_KEY not set",
            )

        cache_days = cache_days if cache_days is not None else int(
            self.runtime.get("cache_days", 7)
        )
        orders = orders or list(self.runtime.get("search_orders", ["relevance", "viewCount", "date"]))
        per_search = int(self.runtime.get("results_per_search", 15))

        video_ids: dict[str, str] = {}
        for order in orders:
            if self.searches_used >= self.search_budget:
                return self.result(
                    f"query:{query}", ConnectorStatus.BUDGET_EXHAUSTED,
                    error=f"daily search.list budget of {self.search_budget} calls spent",
                )
            try:
                self.searches_used += 1
                resp = await self.fetch(
                    f"{API}/search",
                    params={
                        "key": self.settings.youtube_api_key,
                        "q": query,
                        "part": "snippet",
                        "type": "video",
                        "order": order,
                        "maxResults": per_search,
                        "relevanceLanguage": "en",
                    },
                )
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                return self.failure(f"query:{query}", exc)

            for entry in resp.json().get("items", []):
                vid = entry.get("id", {}).get("videoId")
                if vid:
                    video_ids.setdefault(vid, order)

        if not video_ids:
            return self.result(f"query:{query}", ConnectorStatus.OK, [])

        # videos.list costs 1 unit and takes 50 IDs per call: enrichment is
        # effectively free, so always get real statistics rather than guessing
        # from search snippets.
        items: list[SourceItem] = []
        ids = list(video_ids)
        for start in range(0, len(ids), 50):
            batch = ids[start : start + 50]
            try:
                resp = await self.fetch(
                    f"{API}/videos",
                    params={
                        "key": self.settings.youtube_api_key,
                        "id": ",".join(batch),
                        "part": "snippet,statistics,contentDetails",
                    },
                )
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                return self.failure(f"query:{query}", exc)

            for video in resp.json().get("items", []):
                items.append(self._to_item(video, query, video_ids.get(video["id"], "")))

        return self.result(f"query:{query}", ConnectorStatus.OK, items)

    def _to_item(self, video: dict[str, Any], query: str, order: str) -> SourceItem:
        snippet = video.get("snippet", {})
        stats = video.get("statistics", {})
        details = video.get("contentDetails", {})
        return SourceItem(
            source_type=SourceType.YOUTUBE,
            source_name=snippet.get("channelTitle") or "YouTube",
            external_id=video["id"],
            title=snippet.get("title"),
            text=(snippet.get("description") or "")[:2000] or None,
            url=f"https://www.youtube.com/watch?v={video['id']}",
            author_handle=snippet.get("channelTitle"),
            author_name=snippet.get("channelTitle"),
            publication=snippet.get("channelTitle"),
            published_at=parse_date(snippet.get("publishedAt")),
            source_role=SourceRole.EXISTING_COVERAGE,
            primary_or_secondary="SECONDARY",
            metadata={
                "topic_query": query,
                "search_order": order,
                "channel_id": snippet.get("channelId"),
                "views": int(stats.get("viewCount", 0) or 0),
                "likes": int(stats.get("likeCount", 0) or 0),
                "comment_count": int(stats.get("commentCount", 0) or 0),
                "duration_seconds": parse_duration(details.get("duration")),
            },
        )


def render_coverage_block(items: list[SourceItem], limit: int = 25) -> str:
    """Render YouTube results for the Coverage Scout to classify."""
    if not items:
        return ""
    ranked = sorted(items, key=lambda i: i.metadata.get("views", 0), reverse=True)[:limit]
    lines = []
    for item in ranked:
        meta = item.metadata
        duration = meta.get("duration_seconds")
        lines.append(
            f"- \"{item.title}\" | channel: {item.source_name} | "
            f"{meta.get('views', 0):,} views | "
            f"{str(item.published_at)[:10] if item.published_at else 'undated'} | "
            f"{f'{duration // 60}min' if duration else 'unknown length'}"
        )
        if item.text:
            lines.append(f"  description: {item.text[:220]}")
    return "\n".join(lines)
