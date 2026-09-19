"""Substack (and Substack-shaped) RSS connector.

Public RSS only. Verified 2026-09-19: these feeds carry FULL article text in
<content:encoded>, plus author, date and the outbound link graph, from a single
unauthenticated request. That makes this the richest free source in the system
and the place where insider argument actually happens.

Paywalls are never bypassed. If a feed gives only an excerpt, we store the
excerpt and mark the item -- the mention still counts (we know it exists and
who wrote it) but it contributes no body text to the scouts.
"""

from __future__ import annotations

import asyncio
import html
import re
from typing import Any

import feedparser

from ..models import (
    CollectionResult,
    ConnectorStatus,
    Engagement,
    SourceItem,
    SourceRole,
    SourceType,
    TextCompleteness,
)
from .base import Connector, canonicalize, extract_arxiv_ids, parse_date

TAG_RE = re.compile(r"<[^>]+>")
HREF_RE = re.compile(r'href=["\'](https?://[^"\']+)["\']', re.I)

# Markers Substack emits when the body is truncated for non-subscribers.
PAYWALL_MARKERS = (
    "this post is for paid subscribers",
    "this post is for paying subscribers",
    "subscribe to read",
    "keep reading with a 7-day free trial",
    "this episode is for paid subscribers",
)


def strip_html(raw: str) -> str:
    text = TAG_RE.sub(" ", html.unescape(raw or ""))
    return re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]{2,}", " ", text)).strip()


class SubstackConnector(Connector):
    name = "substack"
    min_interval = 1.2

    async def collect(self) -> list[CollectionResult]:
        feeds = self.settings.substack_feeds
        if not feeds:
            return [self.result("config", ConnectorStatus.NOT_CONFIGURED,
                                error="no feeds in config/substack_feeds.yaml")]
        sem = asyncio.Semaphore(2)

        async def one(feed: dict[str, Any]) -> CollectionResult:
            async with sem:
                return await self.collect_feed(feed)

        return list(await asyncio.gather(*(one(f) for f in feeds)))

    async def collect_feed(self, feed: dict[str, Any]) -> CollectionResult:
        url = feed.get("url", "")
        name = feed.get("name") or url
        try:
            resp = await self.fetch(url)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - mapped to an honest status
            return self.failure(name, exc)

        parsed = feedparser.parse(resp.text)
        if parsed.bozo and not parsed.entries:
            return self.result(
                name, ConnectorStatus.SOURCE_UNAVAILABLE,
                error=f"unparseable feed: {getattr(parsed, 'bozo_exception', '')}",
                http_status=resp.status_code,
            )

        items = [self._to_item(e, feed, name) for e in parsed.entries]
        items = [i for i in items if i is not None]
        return self.result(
            name,
            ConnectorStatus.OK if items else ConnectorStatus.PARTIAL,
            items,
            http_status=resp.status_code,
        )

    def _to_item(self, entry: Any, feed: dict[str, Any], pub_name: str) -> SourceItem | None:
        link = entry.get("link")
        guid = entry.get("id") or link
        if not guid:
            return None

        # Prefer the full body; fall back through the progressively poorer fields.
        body_html = ""
        for key in ("content", "summary_detail", "summary", "description"):
            value = entry.get(key)
            if isinstance(value, list) and value:
                body_html = value[0].get("value", "")
            elif isinstance(value, dict):
                body_html = value.get("value", "")
            elif isinstance(value, str):
                body_html = value
            if body_html:
                break

        text = strip_html(body_html)
        lowered = text.lower()
        paywalled = any(marker in lowered for marker in PAYWALL_MARKERS)

        if paywalled or len(text) < 400:
            completeness = TextCompleteness.EXCERPT if text else TextCompleteness.TITLE_ONLY
        else:
            completeness = TextCompleteness.FULL

        links = [canonicalize(u) for u in HREF_RE.findall(body_html or "")]
        links = [u for u in links if u]

        author = entry.get("author") or (entry.get("authors") or [{}])[0].get("name")

        return SourceItem(
            source_type=SourceType.SUBSTACK,
            source_name=pub_name,
            external_id=str(guid),
            title=entry.get("title"),
            text=text or None,
            url=link,
            canonical_url=canonicalize(link),
            author_handle=author,
            author_name=author,
            publication=pub_name,
            published_at=parse_date(entry.get("published") or entry.get("updated")),
            source_role=SourceRole.INDEPENDENT_ANALYSIS,
            primary_or_secondary="SECONDARY",
            paywalled=paywalled,
            text_completeness=completeness,
            outbound_links=links[:80],
            arxiv_refs=extract_arxiv_ids(body_html),
            engagement=Engagement(),
            metadata={
                "domain": feed.get("domain"),
                "curated_tier": feed.get("curated_tier"),
                "feed_url": feed.get("url"),
            },
        )
