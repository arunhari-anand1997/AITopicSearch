"""X signal without the X API.

THE IDEA
--------
We never need X's firehose. What we need is "which X posts did knowledgeable
people think were worth pointing at" -- and that information is already in our
corpus, because newsletters, Hacker News and lab blogs cite X posts constantly.
Every one of those citations was an editorial decision by someone we already
trust enough to track.

Measured on the live corpus with zero X API calls: 629 X/Twitter links across
326 distinct accounts.

WHY THIS IS ARGUABLY THE BETTER SIGNAL
--------------------------------------
Raw X volume tells you how many people posted. A citation tells you that a
writer with a reputation chose to point at something. For finding ideas moving
among knowledgeable insiders -- which is the whole editorial premise -- the
second is closer to what we actually want, and it is immune to engagement
farming, bot amplification and follower-count theatre.

WHAT IT CANNOT DO
-----------------
It is a LAGGING signal by construction: a post only appears once somebody has
written about it. It cannot see a conversation that has not yet been cited
anywhere, it cannot measure reply threads, and it cannot give engagement
counts. Those need the paid API. This is recorded honestly rather than sold as
equivalent.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..db import Database

STATUS_RE = re.compile(
    r"(?:twitter|x)\.com/([A-Za-z0-9_]{1,15})/status/(\d+)", re.I
)
HANDLE_RE = re.compile(r"(?:twitter|x)\.com/([A-Za-z0-9_]{1,15})(?:[/?#]|$)", re.I)

#: Not people: X's own product surfaces.
NON_ACCOUNTS = {
    "i", "home", "search", "explore", "intent", "share", "hashtag",
    "settings", "compose", "messages", "notifications", "login", "signup",
}


@dataclass
class CitedPost:
    url: str
    handle: str | None
    citing_items: int = 0
    citing_authors: int = 0
    citing_publications: int = 0
    first_cited: datetime | None = None
    last_cited: datetime | None = None

    @property
    def is_independently_endorsed(self) -> bool:
        """Several unrelated writers chose to point at the same post."""
        return self.citing_authors >= 3


@dataclass
class XCitationSignal:
    concept_id: int | None = None
    top_posts: list[CitedPost] = field(default_factory=list)
    top_accounts: list[tuple[str, int, int]] = field(default_factory=list)
    total_citations: int = 0
    distinct_accounts: int = 0
    #: Stated on every rendering so the limitation is never lost.
    caveat: str = (
        "Derived from X links cited in newsletters, Hacker News and blogs. "
        "No X API was used. This is a LAGGING signal: it sees only posts "
        "someone already wrote about, and carries no engagement counts."
    )

    def render(self, limit: int = 6) -> str:
        if not self.top_posts and not self.top_accounts:
            return ""
        lines = ["## X ACTIVITY (derived from citations, no X API)", "", self.caveat, ""]
        if self.top_posts:
            lines.append("Most-cited X posts:")
            for post in self.top_posts[:limit]:
                mark = " [independently endorsed]" if post.is_independently_endorsed else ""
                lines.append(
                    f"- {post.url} — cited by {post.citing_authors} distinct "
                    f"authors across {post.citing_publications} publications{mark}"
                )
            lines.append("")
        if self.top_accounts:
            lines.append("Most-referenced X accounts:")
            for handle, links, authors in self.top_accounts[:limit]:
                lines.append(f"- @{handle} — {links} links from {authors} distinct authors")
            lines.append("")
        return "\n".join(lines)


def _extract_handle(url: str) -> str | None:
    match = STATUS_RE.search(url) or HANDLE_RE.search(url)
    if not match:
        return None
    handle = match.group(1)
    return None if handle.lower() in NON_ACCOUNTS else handle


def compute(
    db: Database,
    *,
    concept_id: int | None = None,
    days: int = 60,
    limit: int = 20,
) -> XCitationSignal:
    """Build the X citation signal, optionally scoped to one concept."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    if concept_id is None:
        sql = """
            SELECT l.canonical_url url, l.source_item_id item,
                   si.author_id author, si.publication_id pub, si.published_at pub_at
            FROM links l JOIN source_items si ON si.id = l.source_item_id
            WHERE (l.canonical_url LIKE '%x.com/%'
                   OR l.canonical_url LIKE '%twitter.com/%')
              AND si.published_at >= ?
        """
        params: tuple = (cutoff,)
    else:
        sql = """
            SELECT l.canonical_url url, l.source_item_id item,
                   si.author_id author, si.publication_id pub, si.published_at pub_at
            FROM links l
            JOIN source_items si ON si.id = l.source_item_id
            JOIN concept_mentions cm ON cm.source_item_id = si.id
            WHERE (l.canonical_url LIKE '%x.com/%'
                   OR l.canonical_url LIKE '%twitter.com/%')
              AND cm.concept_id = ?
              AND si.published_at >= ?
        """
        params = (concept_id, cutoff)

    rows = db.conn.execute(sql, params).fetchall()
    signal = XCitationSignal(concept_id=concept_id)
    if not rows:
        return signal

    posts: dict[str, dict] = defaultdict(
        lambda: {"items": set(), "authors": set(), "pubs": set(), "dates": []}
    )
    account_links: Counter[str] = Counter()
    account_authors: defaultdict[str, set] = defaultdict(set)

    for row in rows:
        url = row["url"] or ""
        handle = _extract_handle(url)
        if handle:
            account_links[handle] += 1
            if row["author"] is not None:
                account_authors[handle].add(row["author"])

        # Only /status/ URLs are individual posts worth ranking.
        if not STATUS_RE.search(url):
            continue
        entry = posts[url]
        entry["items"].add(row["item"])
        if row["author"] is not None:
            entry["authors"].add(row["author"])
        if row["pub"] is not None:
            entry["pubs"].add(row["pub"])
        if row["pub_at"]:
            entry["dates"].append(str(row["pub_at"]))

    cited: list[CitedPost] = []
    for url, entry in posts.items():
        dates = sorted(entry["dates"])
        cited.append(
            CitedPost(
                url=url,
                handle=_extract_handle(url),
                citing_items=len(entry["items"]),
                citing_authors=len(entry["authors"]),
                citing_publications=len(entry["pubs"]),
                first_cited=_parse(dates[0]) if dates else None,
                last_cited=_parse(dates[-1]) if dates else None,
            )
        )
    cited.sort(key=lambda p: (p.citing_authors, p.citing_items), reverse=True)

    signal.top_posts = cited[:limit]
    signal.top_accounts = [
        (handle, count, len(account_authors[handle]))
        for handle, count in account_links.most_common(limit)
    ]
    signal.total_citations = len(rows)
    signal.distinct_accounts = len(account_links)
    return signal


def _parse(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None
