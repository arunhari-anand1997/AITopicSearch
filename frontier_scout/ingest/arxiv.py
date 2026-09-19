"""arXiv connector.

No credentials. arXiv's official terms of use require "no more than one request
every three seconds, and limit requests to a single connection at a time",
binding all machines under our control -- so the throttle here is a single
global limiter, not a per-task one.

A paper existing is a weak signal. Papers earn their score from INBOUND
references: arXiv IDs pulled out of newsletters, HN submissions and lab blogs.
The trend engine resolves those against paper_refs, so the papers that surface
are the ones people are arguing about, not the ones with tidy abstracts.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, urlencode

import feedparser

from ..models import (
    CollectionResult,
    ConnectorStatus,
    SourceItem,
    SourceRole,
    SourceType,
)
from .base import Connector, canonicalize, parse_date

API = "https://export.arxiv.org/api/query"  # https; plain http 301s
ABS_ID_RE = re.compile(r"arxiv\.org/abs/([^v\s]+)(?:v(\d+))?")


class ArxivConnector(Connector):
    name = "arxiv"
    #: arXiv terms: one request per three seconds, single connection.
    min_interval = 3.0

    async def collect(self) -> list[CollectionResult]:
        cfg = self.settings.arxiv_queries
        results: list[CollectionResult] = []

        for category in cfg.get("categories", []):
            results.append(
                await self._query(
                    f"cat:{category}",
                    label=f"cat:{category}",
                    max_results=int(cfg.get("max_results_per_category", 60)),
                )
            )
        for phrase in cfg.get("phrase_queries", []):
            results.append(
                await self._query(
                    f'all:"{phrase}"',
                    label=f"phrase:{phrase}",
                    max_results=int(cfg.get("max_results_per_phrase", 30)),
                )
            )
        return results

    async def _query(self, search_query: str, label: str, max_results: int) -> CollectionResult:
        params = urlencode(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            quote_via=quote,
        )
        try:
            resp = await self.fetch(f"{API}?{params}")
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return self.failure(label, exc)

        parsed = feedparser.parse(resp.text)
        items = [i for i in (self._to_item(e) for e in parsed.entries) if i]
        return self.result(label, ConnectorStatus.OK, items, http_status=resp.status_code)

    def _to_item(self, entry: Any) -> SourceItem | None:
        link = entry.get("id")
        if not link:
            return None
        match = ABS_ID_RE.search(link)
        arxiv_id = match.group(1) if match else link.rsplit("/", 1)[-1]
        version = match.group(2) if match and match.group(2) else None

        authors = [a.get("name") for a in entry.get("authors", []) if a.get("name")]
        categories = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
        abstract = (entry.get("summary") or "").strip()

        return SourceItem(
            source_type=SourceType.ARXIV,
            source_name="arXiv",
            external_id=arxiv_id,
            title=(entry.get("title") or "").replace("\n", " ").strip(),
            text=abstract or None,
            url=link,
            canonical_url=canonicalize(f"https://arxiv.org/abs/{arxiv_id}"),
            author_handle=authors[0] if authors else None,
            author_name=", ".join(authors[:6]) if authors else None,
            publication="arXiv",
            published_at=parse_date(entry.get("published")),
            source_role=SourceRole.PRIMARY_EVIDENCE,
            primary_or_secondary="PRIMARY",
            arxiv_refs=[arxiv_id],
            metadata={
                "categories": categories,
                "authors": authors,
                "version": version,
                "updated": entry.get("updated"),
                "doi": entry.get("arxiv_doi"),
                "comment": entry.get("arxiv_comment"),
            },
        )
