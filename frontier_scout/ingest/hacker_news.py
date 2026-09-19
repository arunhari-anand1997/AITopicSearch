"""Hacker News: live front page plus historical search.

Two connectors because they answer different questions.

  Firebase -> what the developer community is reacting to right now.
  Algolia  -> what it has said historically, which is what makes day-one
              baselines possible instead of waiting a month for one.

HN is ONE insider-community signal. It is emphatically not representative of
the general public, and nothing downstream treats it as public interest.

MEASUREMENT TRAP (verified 2026-09-19): Algolia returns at most 1000 hits in a
single response (`nbPages=1` even when `nbHits=4983`), so taking min(created_at)
over one page yields the 1000th-most-recent match, not the first. A naive
first-seen implementation reports a years-old term as brand new. We therefore
never derive novelty from Algolia; first observation is authoritative only from
our own database. See `_search_window` for how counts are paginated correctly.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from ..models import (
    CollectionResult,
    ConnectorStatus,
    Engagement,
    SourceItem,
    SourceRole,
    SourceType,
)
from .base import Connector, extract_arxiv_ids, parse_date
from .base import canonicalize as _canon

FIREBASE = "https://hacker-news.firebaseio.com/v0"
ALGOLIA = "https://hn.algolia.com/api/v1"


class HackerNewsConnector(Connector):
    """Live front-page feeds via the official Firebase API.

    Official docs state there is currently no rate limit and no key is needed;
    we still space requests politely rather than hammering it.
    """

    name = "hacker_news"
    min_interval = 0.05

    async def collect(self) -> list[CollectionResult]:
        cfg = self.settings.hn_config.get("firebase", {})
        feeds = cfg.get("feeds", ["topstories", "beststories", "newstories"])
        per_feed = int(cfg.get("items_per_feed", 80))
        min_score = int(cfg.get("min_score", 5))
        results = []
        seen: set[int] = set()
        for feed in feeds:
            results.append(await self._collect_feed(feed, per_feed, min_score, seen))
        return results

    async def _collect_feed(
        self, feed: str, per_feed: int, min_score: int, seen: set[int]
    ) -> CollectionResult:
        try:
            resp = await self.fetch(f"{FIREBASE}/{feed}.json")
            resp.raise_for_status()
            ids = [i for i in resp.json()[:per_feed] if i not in seen]
        except Exception as exc:  # noqa: BLE001
            return self.failure(feed, exc)

        sem = asyncio.Semaphore(12)

        async def item(item_id: int) -> SourceItem | None:
            async with sem:
                try:
                    r = await self.fetch(f"{FIREBASE}/item/{item_id}.json")
                    r.raise_for_status()
                    data = r.json()
                except Exception:  # noqa: BLE001 - one dead item must not kill the feed
                    return None
            if not data or data.get("dead") or data.get("deleted"):
                return None
            if (data.get("score") or 0) < min_score:
                return None
            return self._to_item(data, feed)

        gathered = await asyncio.gather(*(item(i) for i in ids))
        items = [i for i in gathered if i]
        seen.update(ids)
        return self.result(feed, ConnectorStatus.OK, items)

    def _to_item(self, data: dict[str, Any], feed: str) -> SourceItem:
        url = data.get("url")
        text = data.get("text") or ""
        return SourceItem(
            source_type=SourceType.HACKER_NEWS,
            source_name=f"hn:{feed}",
            external_id=str(data["id"]),
            title=data.get("title"),
            text=text or None,
            url=url or f"https://news.ycombinator.com/item?id={data['id']}",
            canonical_url=_canon(url) if url else None,
            author_handle=data.get("by"),
            author_name=data.get("by"),
            publication="Hacker News",
            published_at=parse_date(data.get("time")),
            source_role=SourceRole.COMMUNITY_REACTION,
            primary_or_secondary="SECONDARY",
            engagement=Engagement(
                score=data.get("score"),
                comment_count=data.get("descendants"),
            ),
            outbound_links=[u for u in [_canon(url)] if u],
            arxiv_refs=extract_arxiv_ids(f"{url or ''} {text}"),
            metadata={"hn_type": data.get("type"), "feed": feed},
        )


class HNSearchConnector(Connector):
    """Algolia HN search, used for historical backfill and concept counting.

    This is what gives the trend engine a real baseline on day one.
    """

    name = "hn_search"
    min_interval = 0.35

    async def collect(self) -> list[CollectionResult]:
        """Backfill the seed concepts' aliases across the configured window."""
        cfg = self.settings.hn_config.get("algolia", {})
        days = int(cfg.get("backfill_days", self.settings.backfill_days))
        phrases: list[str] = []
        for concept in self.settings.seed_concepts:
            phrases.extend(concept.get("aliases", [concept["canonical_label"]])[:2])
        seen_ids: set[str] = set()
        results = []
        for phrase in dict.fromkeys(phrases):
            results.append(await self.search_phrase(phrase, days, seen_ids, cfg))
        return results

    async def search_phrase(
        self,
        phrase: str,
        days: int,
        seen_ids: set[str] | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> CollectionResult:
        cfg = cfg or {}
        seen_ids = seen_ids if seen_ids is not None else set()
        hits_per_page = int(cfg.get("hits_per_page", 100))
        max_pages = int(cfg.get("max_pages", 5))
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        tags = "(story,comment)" if cfg.get("include_comments", True) else "story"

        items: list[SourceItem] = []
        try:
            # Paginate properly. A single page would cap unique-author counts at
            # hits_per_page and produce plausible-looking but meaningless numbers.
            for page in range(max_pages):
                url = (
                    f"{ALGOLIA}/search_by_date?query=%22{quote(phrase)}%22&tags={tags}"
                    f"&numericFilters=created_at_i>{since}"
                    f"&hitsPerPage={hits_per_page}&page={page}"
                )
                resp = await self.fetch(url)
                resp.raise_for_status()
                payload = resp.json()
                for hit in payload.get("hits", []):
                    obj_id = str(hit.get("objectID"))
                    if obj_id in seen_ids:
                        continue
                    seen_ids.add(obj_id)
                    item = self._to_item(hit, phrase)
                    if item:
                        items.append(item)
                if page + 1 >= payload.get("nbPages", 0):
                    break
        except Exception as exc:  # noqa: BLE001
            return self.failure(f"search:{phrase}", exc)

        return self.result(f"search:{phrase}", ConnectorStatus.OK, items)

    def _to_item(self, hit: dict[str, Any], phrase: str) -> SourceItem | None:
        obj_id = hit.get("objectID")
        if not obj_id:
            return None
        title = hit.get("title") or hit.get("story_title")
        comment = hit.get("comment_text")
        story_text = hit.get("story_text")
        body = comment or story_text or ""
        if not title and not body:
            return None
        url = hit.get("url") or hit.get("story_url")
        return SourceItem(
            source_type=SourceType.HACKER_NEWS,
            source_name="hn:search",
            external_id=str(obj_id),
            title=title,
            text=body or None,
            url=url or f"https://news.ycombinator.com/item?id={obj_id}",
            canonical_url=_canon(url) if url else None,
            author_handle=hit.get("author"),
            author_name=hit.get("author"),
            publication="Hacker News",
            published_at=parse_date(hit.get("created_at")),
            source_role=SourceRole.COMMUNITY_REACTION,
            primary_or_secondary="SECONDARY",
            engagement=Engagement(
                score=hit.get("points"),
                comment_count=hit.get("num_comments"),
            ),
            outbound_links=[u for u in [_canon(url)] if u],
            arxiv_refs=extract_arxiv_ids(f"{url or ''} {body}"),
            metadata={"matched_phrase": phrase, "is_comment": bool(comment)},
        )
