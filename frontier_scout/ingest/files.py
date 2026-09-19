"""File ingestion: JSON, JSONL, CSV and plain text.

For manually gathered material -- a batch of posts pasted into a file, an
export from somewhere without an API, notes from a conference. Ingested items
are marked source_type=manual so they are never confused with something a
connector observed directly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

from ..models import (
    CollectionResult,
    ConnectorStatus,
    Engagement,
    SourceItem,
    SourceRole,
    SourceType,
)
from .base import canonicalize, extract_arxiv_ids, parse_date

ROLE_BY_TYPE = {
    SourceType.X: SourceRole.EARLY_SIGNAL,
    SourceType.SUBSTACK: SourceRole.INDEPENDENT_ANALYSIS,
    SourceType.ARXIV: SourceRole.PRIMARY_EVIDENCE,
    SourceType.HACKER_NEWS: SourceRole.COMMUNITY_REACTION,
    SourceType.REDDIT: SourceRole.COMMUNITY_REACTION,
    SourceType.YOUTUBE: SourceRole.EXISTING_COVERAGE,
    SourceType.COMPANY_BLOG: SourceRole.PRIMARY_EVIDENCE,
    SourceType.RESEARCH_BLOG: SourceRole.PRIMARY_EVIDENCE,
    SourceType.PODCAST: SourceRole.INDEPENDENT_ANALYSIS,
    SourceType.MANUAL: SourceRole.EARLY_SIGNAL,
}


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
    elif suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("items") or data.get("data") or [data]
        yield from data
    elif suffix == ".csv":
        with path.open(newline="") as fh:
            yield from csv.DictReader(fh)
    else:
        # Plain text: blank-line-separated blocks, each one item.
        for idx, block in enumerate(path.read_text().split("\n\n")):
            block = block.strip()
            if block:
                yield {"text": block, "external_id": f"{path.stem}:{idx}"}


def row_to_item(row: dict[str, Any], default_name: str) -> SourceItem | None:
    text = row.get("text") or row.get("content") or row.get("body")
    title = row.get("title")
    if not text and not title:
        return None
    try:
        source_type = SourceType(str(row.get("source_type", "manual")).lower())
    except ValueError:
        source_type = SourceType.MANUAL

    url = row.get("url") or row.get("link")
    external_id = str(
        row.get("external_id") or row.get("id") or url or f"{default_name}:{hash(text or title)}"
    )
    role_raw = row.get("source_role")
    try:
        role = SourceRole(role_raw) if role_raw else ROLE_BY_TYPE[source_type]
    except ValueError:
        role = ROLE_BY_TYPE[source_type]

    engagement = None
    if any(k in row for k in ("likes", "reposts", "replies", "views", "score", "comment_count")):
        def num(key: str) -> int | None:
            value = row.get(key)
            return int(value) if value not in (None, "") else None

        engagement = Engagement(
            likes=num("likes"), reposts=num("reposts"), replies=num("replies"),
            views=num("views"), score=num("score"), comment_count=num("comment_count"),
        )

    return SourceItem(
        source_type=source_type,
        source_name=str(row.get("source_name") or default_name),
        external_id=external_id,
        title=title,
        text=text,
        url=url,
        canonical_url=canonicalize(url),
        author_handle=row.get("author") or row.get("author_handle"),
        author_name=row.get("author_name") or row.get("author"),
        publication=row.get("publication"),
        published_at=parse_date(row.get("published_at") or row.get("date")),
        source_role=role,
        primary_or_secondary=(
            "PRIMARY" if role == SourceRole.PRIMARY_EVIDENCE else "SECONDARY"
        ),
        engagement=engagement,
        arxiv_refs=extract_arxiv_ids(f"{url or ''} {text or ''}"),
        metadata={"ingested_from": default_name},
    )


def ingest_file(path: str | Path) -> CollectionResult:
    path = Path(path)
    if not path.exists():
        return CollectionResult(
            connector="file", target=str(path),
            status=ConnectorStatus.SOURCE_UNAVAILABLE, error="file not found",
        )
    try:
        items = [i for i in (row_to_item(r, path.stem) for r in _rows(path)) if i]
    except Exception as exc:  # noqa: BLE001
        return CollectionResult(
            connector="file", target=str(path),
            status=ConnectorStatus.SOURCE_UNAVAILABLE,
            error=f"{type(exc).__name__}: {exc}",
        )
    return CollectionResult(
        connector="file", target=str(path), status=ConnectorStatus.OK, items=items
    )
