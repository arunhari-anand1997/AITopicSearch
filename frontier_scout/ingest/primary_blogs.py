"""Primary-source lab and company blogs.

Two modes:

  rss     -- parse the feed.
  sitemap -- no RSS exists. Diff sitemap.xml by <lastmod>, fetch only pages we
             have not seen, respect robots.txt, and stay polite.

The sitemap mode exists because Anthropic and Meta AI publish no feed (verified
2026-09-19: /rss.xml, /news/rss.xml, /engineering/rss.xml and /feed.xml all
404), while anthropic.com/robots.txt is `Allow: /` and advertises a sitemap
carrying 535 URLs each with a <lastmod>. That is a perfectly good change feed.

EDITORIAL RULE, enforced by the data rather than by prose: items here are
PRIMARY_EVIDENCE for what the organisation CLAIMS. They are never evidence the
claim is true. Benchmark and capability claims are stashed in
metadata['claimed_capabilities'] and the scout prompts are required to render
them as "X says Y".
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser

from ..models import (
    CollectionResult,
    ConnectorStatus,
    SourceItem,
    SourceRole,
    SourceType,
    TextCompleteness,
)
from .base import Connector, canonicalize, extract_arxiv_ids, parse_date
from .substack import HREF_RE, strip_html

LOC_RE = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.S | re.I)
LASTMOD_RE = re.compile(r"<lastmod>\s*(.*?)\s*</lastmod>", re.S | re.I)
URL_BLOCK_RE = re.compile(r"<url>(.*?)</url>", re.S | re.I)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)

# Sentences where an organisation asserts a capability or a benchmark number.
# Collected so a scout can quote the claim rather than restate it as fact.
CLAIM_RE = re.compile(
    r"[^.!?\n]*\b(?:state-of-the-art|SOTA|outperform\w*|benchmark\w*|achieves?|"
    r"surpass\w*|record|best-in-class|first (?:model|system)|\d+(?:\.\d+)?%)\b[^.!?\n]*[.!?]",
    re.I,
)


class PrimaryBlogConnector(Connector):
    name = "primary_blog"
    min_interval = 1.0

    async def collect(self) -> list[CollectionResult]:
        feeds = self.settings.primary_feeds
        if not feeds:
            return [self.result("config", ConnectorStatus.NOT_CONFIGURED,
                                error="no feeds in config/primary_feeds.yaml")]
        results = []
        for feed in feeds:
            mode = feed.get("mode", "rss")
            if mode == "sitemap":
                results.append(await self.collect_sitemap(feed))
            else:
                results.append(await self.collect_rss(feed))
        return results

    # -- RSS ---------------------------------------------------------------

    async def collect_rss(self, feed: dict[str, Any]) -> CollectionResult:
        url = feed.get("url", "")
        name = feed.get("name") or url
        try:
            resp = await self.fetch(url)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return self.failure(name, exc)

        parsed = feedparser.parse(resp.text)
        if parsed.bozo and not parsed.entries:
            return self.result(
                name, ConnectorStatus.SOURCE_UNAVAILABLE,
                error="unparseable feed", http_status=resp.status_code,
            )
        items = [i for i in (self._from_entry(e, feed, name) for e in parsed.entries) if i]
        return self.result(name, ConnectorStatus.OK, items, http_status=resp.status_code)

    def _from_entry(self, entry: Any, feed: dict[str, Any], name: str) -> SourceItem | None:
        link = entry.get("link")
        guid = entry.get("id") or link
        if not guid:
            return None
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
        return self._build(
            external_id=str(guid),
            title=entry.get("title"),
            text=text,
            url=link,
            published_at=parse_date(entry.get("published") or entry.get("updated")),
            author=entry.get("author"),
            body_html=body_html,
            feed=feed,
            name=name,
        )

    # -- sitemap fallback --------------------------------------------------

    async def collect_sitemap(self, feed: dict[str, Any]) -> CollectionResult:
        url = feed.get("url", "")
        name = feed.get("name") or url
        include = feed.get("include_paths") or []
        horizon = datetime.now(timezone.utc) - timedelta(days=self.settings.backfill_days)

        try:
            if not await self.robots.allowed(self.client, url):
                return self.result(name, ConnectorStatus.ACCESS_DENIED,
                                   error="robots.txt disallows the sitemap")
            resp = await self.fetch(url)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return self.failure(name, exc)

        targets: list[tuple[str, datetime | None]] = []
        for block in URL_BLOCK_RE.findall(resp.text):
            loc = LOC_RE.search(block)
            if not loc:
                continue
            page = loc.group(1).strip()
            if include and not any(part in page for part in include):
                continue
            lastmod_match = LASTMOD_RE.search(block)
            lastmod = parse_date(lastmod_match.group(1)) if lastmod_match else None
            if lastmod and lastmod < horizon:
                continue
            targets.append((page, lastmod))

        # Newest first, and bounded: a sitemap diff should not become a crawl.
        targets.sort(key=lambda t: t[1] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        limit = int(feed.get("max_pages", 25))
        targets = targets[:limit]

        sem = asyncio.Semaphore(2)

        async def page(entry: tuple[str, datetime | None]) -> SourceItem | None:
            page_url, lastmod = entry
            async with sem:
                try:
                    if not await self.robots.allowed(self.client, page_url):
                        return None
                    r = await self.fetch(page_url)
                    r.raise_for_status()
                except Exception:  # noqa: BLE001 - one dead page must not kill the source
                    return None
            html_text = r.text
            title_match = TITLE_RE.search(html_text)
            title = strip_html(title_match.group(1)) if title_match else page_url
            return self._build(
                external_id=page_url,
                title=title,
                text=strip_html(html_text)[:20000],
                url=page_url,
                published_at=lastmod,
                author=None,
                body_html=html_text,
                feed=feed,
                name=name,
            )

        gathered = await asyncio.gather(*(page(t) for t in targets))
        items = [i for i in gathered if i]
        status = ConnectorStatus.OK if items else ConnectorStatus.PARTIAL
        return self.result(name, status, items, http_status=resp.status_code)

    # -- shared ------------------------------------------------------------

    def _build(
        self,
        *,
        external_id: str,
        title: str | None,
        text: str,
        url: str | None,
        published_at: datetime | None,
        author: str | None,
        body_html: str,
        feed: dict[str, Any],
        name: str,
    ) -> SourceItem:
        links = [canonicalize(u) for u in HREF_RE.findall(body_html or "")]
        claims = [c.strip() for c in CLAIM_RE.findall(text or "")][:12]
        return SourceItem(
            source_type=SourceType.COMPANY_BLOG,
            source_name=name,
            external_id=external_id,
            title=title,
            text=text or None,
            url=url,
            canonical_url=canonicalize(url),
            author_handle=author or feed.get("org"),
            author_name=author,
            publication=name,
            published_at=published_at,
            source_role=SourceRole.PRIMARY_EVIDENCE,
            primary_or_secondary="PRIMARY",
            text_completeness=TextCompleteness.FULL if len(text) > 400 else TextCompleteness.EXCERPT,
            outbound_links=[u for u in links if u][:80],
            arxiv_refs=extract_arxiv_ids(f"{body_html} {text}"),
            metadata={
                "org": feed.get("org"),
                "curated_tier": feed.get("curated_tier"),
                "mode": feed.get("mode", "rss"),
                # Stored as CLAIMS, never as established facts. Scout prompts
                # must render these as "<org> says X".
                "claimed_capabilities": claims,
            },
        )
