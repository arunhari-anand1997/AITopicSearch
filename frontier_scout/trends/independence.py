"""Independence: telling convergence apart from an echo.

Counting authors is not enough. Five posts by one person is one source. Four
writers who all cite the same origin essay is *also* close to one source -- and
that is the case the whole system most needs to catch, because it looks exactly
like consensus from the outside.

So independence is a graph problem. Build a graph over the items mentioning a
concept, join any two that share an origin, and count connected components:

    same author            -> not independent
    explicit citation      -> derived
    shared outbound URL    -> common ancestor
    shared arXiv reference -> common ancestor
    near-duplicate text    -> copied

    independent_source_count = number of connected components
    independence_ratio       = components / distinct_authors

Five posts, one author          -> 1 component, ratio 0.2
Four writers citing one essay   -> 1 component, ratio 0.25   <- the trap
Four writers, no shared origin  -> 4 components, ratio 1.0   <- the buy signal

The ratio then discounts momentum, so a single-origin cascade is suppressed
arithmetically rather than argued down by a prompt.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..db import Database

SHINGLE_RE = re.compile(r"[a-z0-9]+")
# Domains that are shared context rather than evidence of derivation: everyone
# links GitHub and arXiv, so joining on them would collapse unrelated writers
# into one component and hide genuine convergence.
GENERIC_LINK_HOSTS = (
    "twitter.com", "x.com", "youtube.com", "youtu.be", "github.com",
    "en.wikipedia.org", "google.com", "substack.com", "linkedin.com",
    "news.ycombinator.com", "reddit.com",
)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self.parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:  # path compression
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def components(self, keys: list[str]) -> dict[str, list[str]]:
        groups: defaultdict[str, list[str]] = defaultdict(list)
        for key in keys:
            groups[self.find(key)].append(key)
        return dict(groups)


@dataclass
class IndependenceResult:
    independent_source_count: int = 0
    distinct_authors: int = 0
    independence_ratio: float = 1.0
    components: dict[str, list[str]] = field(default_factory=dict)
    #: Why items were joined, for the editor's benefit.
    join_reasons: list[str] = field(default_factory=list)

    @property
    def is_single_origin_cascade(self) -> bool:
        """True when many authors collapse into very few independent sources."""
        return self.distinct_authors >= 3 and self.independence_ratio <= 0.5


def _shingles(text: str, size: int = 8) -> set[str]:
    words = SHINGLE_RE.findall((text or "").lower())
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(0, len(words) - size + 1, 3)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_generic(url: str | None) -> bool:
    return not url or any(host in url for host in GENERIC_LINK_HOSTS)


def compute_independence(
    db: Database,
    item_ids: list[str],
    *,
    duplicate_threshold: float = 0.35,
) -> IndependenceResult:
    """Cluster `item_ids` into independent sources."""
    if not item_ids:
        return IndependenceResult()

    marks = ",".join("?" * len(item_ids))
    rows = db.conn.execute(
        f"SELECT id, author_id, publication_id, canonical_url, title, text"
        f" FROM source_items WHERE id IN ({marks})",
        item_ids,
    ).fetchall()
    if not rows:
        return IndependenceResult()

    uf = UnionFind()
    reasons: list[str] = []
    for row in rows:
        uf.add(row["id"])

    # 1. Same author -> one source.
    by_author: defaultdict[int, list[str]] = defaultdict(list)
    for row in rows:
        if row["author_id"] is not None:
            by_author[row["author_id"]].append(row["id"])
    for author_id, ids in by_author.items():
        for other in ids[1:]:
            uf.union(ids[0], other)
        if len(ids) > 1:
            reasons.append(f"same author ({len(ids)} items)")

    # 2. One item's canonical URL being cited by another -> derived.
    url_owner = {r["canonical_url"]: r["id"] for r in rows if r["canonical_url"]}
    link_rows = db.conn.execute(
        f"SELECT source_item_id, canonical_url FROM links WHERE source_item_id IN ({marks})",
        item_ids,
    ).fetchall()
    for link in link_rows:
        owner = url_owner.get(link["canonical_url"])
        if owner and owner != link["source_item_id"]:
            uf.union(owner, link["source_item_id"])
            reasons.append("explicit citation")

    # 3. Two items citing the same non-generic URL -> common ancestor.
    shared: defaultdict[str, set[str]] = defaultdict(set)
    for link in link_rows:
        url = link["canonical_url"]
        if not _is_generic(url):
            shared[url].add(link["source_item_id"])
    for url, ids in shared.items():
        ids_list = sorted(ids)
        if len(ids_list) > 1:
            for other in ids_list[1:]:
                uf.union(ids_list[0], other)
            reasons.append(f"shared source: {url[:60]}")

    # 4. Two items referencing the same paper -> common ancestor.
    paper_rows = db.conn.execute(
        f"SELECT source_item_id, arxiv_id FROM paper_refs WHERE source_item_id IN ({marks})",
        item_ids,
    ).fetchall()
    by_paper: defaultdict[str, set[str]] = defaultdict(set)
    for ref in paper_rows:
        if ref["arxiv_id"]:
            by_paper[ref["arxiv_id"]].add(ref["source_item_id"])
    for arxiv_id, ids in by_paper.items():
        ids_list = sorted(ids)
        if len(ids_list) > 1:
            for other in ids_list[1:]:
                uf.union(ids_list[0], other)
            reasons.append(f"shared paper: arXiv:{arxiv_id}")

    # 5. Near-duplicate phrasing -> copied. Bounded: this is O(n^2).
    if len(rows) <= 220:
        shingle_map = {
            r["id"]: _shingles(f"{r['title'] or ''} {(r['text'] or '')[:2500]}") for r in rows
        }
        ids_list = [r["id"] for r in rows]
        for i, a in enumerate(ids_list):
            for b in ids_list[i + 1 :]:
                if uf.find(a) == uf.find(b):
                    continue
                if _jaccard(shingle_map[a], shingle_map[b]) >= duplicate_threshold:
                    uf.union(a, b)
                    reasons.append("near-duplicate text")

    components = uf.components([r["id"] for r in rows])
    distinct_authors = len({r["author_id"] for r in rows if r["author_id"] is not None})
    count = len(components)
    ratio = count / distinct_authors if distinct_authors else 1.0

    return IndependenceResult(
        independent_source_count=count,
        distinct_authors=distinct_authors,
        independence_ratio=min(1.0, ratio),
        components=components,
        join_reasons=sorted(set(reasons))[:12],
    )


def assign_clusters(db: Database, concept_id: int, result: IndependenceResult) -> None:
    """Write component membership back onto the mentions, for inspection."""
    mapping: dict[str, int] = {}
    for idx, (_root, members) in enumerate(sorted(result.components.items())):
        for member in members:
            mapping[member] = idx
    if not mapping:
        return
    with db.tx() as c:
        for item_id, cluster_idx in mapping.items():
            c.execute(
                "UPDATE concept_mentions SET independent_cluster_id=?"
                " WHERE concept_id=? AND source_item_id=?",
                (cluster_idx, concept_id, item_id),
            )
