"""The concept registry: stable, lexical concept identity.

This is the least glamorous module in the system and the one most responsible
for whether any of the numbers mean anything.

If concepts were extracted afresh by a model each day, the same idea would be
labelled "context engineering" on Monday and "prompt scaffolding" on Tuesday,
and the 30-day series would silently fragment into disconnected pieces. Every
velocity number computed on top would be fiction.

So concept identity is lexical and deterministic: a canonical label plus an
alias set, matched against normalised text. The LLM's role is to PROPOSE
additions to the registry -- never to do the counting.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from ..db import Database
from ..models import SourceItem

WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
# Smart quotes and dashes, normalised before tokenising: otherwise "don't"
# tokenises as "don" + "t" and the bigram "don t" looks like a rising phrase.
UNICODE_PUNCT = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2011": "-", "\u00a0": " ",
})
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Phrases too generic to carry meaning on their own. A bare "harness" matches
# "harness the power of"; requiring specificity keeps the counts honest.
STOPWORD_PHRASES = {
    "ai", "model", "models", "data", "system", "systems", "tool", "tools",
    "agent", "agents", "compute", "training", "context", "harness", "loop",
    "the", "this", "that", "new", "open", "work", "code", "team",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "by", "for", "with", "from", "as", "it", "its",
    "we", "you", "they", "he", "she", "i", "not", "no", "so", "such", "can",
    "will", "would", "could", "should", "may", "might", "must", "have", "has",
    "had", "do", "does", "did", "our", "their", "your", "his", "her", "about",
    "into", "over", "more", "most", "some", "any", "all", "one", "two", "up",
    "out", "just", "like", "also", "very", "much", "how", "what", "when",
    "which", "who", "why", "there", "here", "now", "get", "got", "make",
}


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation to a token stream."""
    return " ".join(WORD_RE.findall(text.translate(UNICODE_PUNCT).lower()))


# Words so common in this corpus that a phrase built only from them carries no
# information: "ai models", "large language", "new model". A nominated phrase
# must contain at least one token from outside this set.
GENERIC_TOKENS = {
    "ai", "llm", "llms", "model", "models", "language", "large", "small",
    "data", "system", "systems", "tool", "tools", "agent", "agents", "use",
    "using", "used", "new", "open", "source", "code", "coding", "work",
    "working", "team", "teams", "company", "companies", "people", "time",
    "week", "year", "day", "good", "better", "best", "big", "real", "thing",
    "things", "lot", "way", "ways", "post", "read", "write", "writing",
    "think", "thinking", "know", "want", "need", "going", "still", "even",
    "many", "first", "last", "next", "long", "short", "high", "low",
    "down", "up", "inside", "show", "shows", "view", "results", "results",
}


def is_specific_enough(phrase: str) -> bool:
    """Reject phrases too generic or too malformed to be a concept.

    This filter exists because the nominator is otherwise extremely good at
    discovering that people say "large language" and "ai models" a lot.
    """
    words = phrase.split()
    if not words:
        return False
    # Contraction and possessive fragments: "don t", "claude s".
    if any(len(w) <= 1 for w in words):
        return False
    # Version numbers and bare digits: "5 1", "2026 release".
    if any(w.isdigit() for w in words):
        return False
    # At least one token has to carry information.
    if all(w in GENERIC_TOKENS or w in STOPWORDS for w in words):
        return False
    if len(words) == 1:
        return (
            phrase not in STOPWORD_PHRASES
            and phrase not in STOPWORDS
            and phrase not in GENERIC_TOKENS
            and len(phrase) > 4
        )
    return not all(w in STOPWORDS for w in words)


class ConceptRegistry:
    """Loads, matches and grows the set of tracked concepts."""

    def __init__(self, db: Database):
        self.db = db
        self._alias_to_concept: dict[str, int] = {}
        self._labels: dict[int, str] = {}
        self._status: dict[int, str] = {}
        self._max_alias_words = 1
        self.reload()

    # -- loading -----------------------------------------------------------

    def reload(self) -> None:
        self._alias_to_concept.clear()
        self._labels.clear()
        self._status.clear()
        rows = self.db.conn.execute(
            "SELECT a.normalized, a.concept_id, c.canonical_label, c.status"
            " FROM concept_aliases a JOIN concepts c ON c.id = a.concept_id"
            " WHERE c.status IN ('tracked','nominated') AND c.merged_into IS NULL"
        ).fetchall()
        for row in rows:
            self._alias_to_concept[row["normalized"]] = row["concept_id"]
            self._labels[row["concept_id"]] = row["canonical_label"]
            self._status[row["concept_id"]] = row["status"]
        self._max_alias_words = max(
            (len(a.split()) for a in self._alias_to_concept), default=1
        )

    def seed(self, concepts: list[dict[str, Any]]) -> int:
        """Load the editor's seed concepts from config. Idempotent."""
        added = 0
        for spec in concepts:
            label = spec["canonical_label"]
            aliases = spec.get("aliases") or [label]
            cid = self.ensure_concept(
                label, kind=spec.get("kind", "term"), status="tracked", aliases=aliases
            )
            added += 1 if cid else 0
        self.reload()
        return added

    def ensure_concept(
        self,
        label: str,
        *,
        kind: str = "term",
        status: str = "tracked",
        aliases: Iterable[str] | None = None,
        added_by: str = "config",
    ) -> int:
        with self.db.tx() as c:
            c.execute(
                "INSERT INTO concepts(canonical_label,kind,status,created_at) VALUES(?,?,?,?)"
                " ON CONFLICT(canonical_label) DO UPDATE SET kind=excluded.kind",
                (label, kind, status, datetime.now(timezone.utc).isoformat()),
            )
            row = c.execute(
                "SELECT id FROM concepts WHERE canonical_label=?", (label,)
            ).fetchone()
            cid = int(row["id"])
            for alias in set(aliases or [label]) | {label}:
                c.execute(
                    "INSERT OR IGNORE INTO concept_aliases(concept_id,alias,normalized,added_by)"
                    " VALUES(?,?,?,?)",
                    (cid, alias, normalize(alias), added_by),
                )
        return cid

    def label(self, concept_id: int) -> str:
        return self._labels.get(concept_id, f"concept:{concept_id}")

    @property
    def concept_ids(self) -> list[int]:
        return sorted(self._labels)

    def status(self, concept_id: int) -> str:
        return self._status.get(concept_id, "tracked")

    @property
    def tracked_ids(self) -> list[int]:
        """Editor-approved concepts only.

        Auto-nominated phrases are measured but must not reach the expensive
        stage on their own: a nomination is a suggestion for the editor, not a
        decision the system gets to make for itself.
        """
        return sorted(cid for cid, st in self._status.items() if st == "tracked")

    # -- matching ----------------------------------------------------------

    def find_mentions(self, text: str) -> list[tuple[int, int, str]]:
        """Return (concept_id, char_offset, context) for each alias hit.

        Context is kept for every mention because a literal phrase match does
        not guarantee the same sense: "loop engineering" appears on Hacker News
        back to 2012 meaning control-loop engineering. Without the snippet, a
        scout cannot tell an old idea from a new one.
        """
        if not text:
            return []
        lowered = text.lower()
        tokens = list(WORD_RE.finditer(lowered))
        if not tokens:
            return []

        hits: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int]] = set()
        for start in range(len(tokens)):
            for span in range(self._max_alias_words, 0, -1):
                if start + span > len(tokens):
                    continue
                phrase = " ".join(t.group(0) for t in tokens[start : start + span])
                cid = self._alias_to_concept.get(phrase)
                if cid is None:
                    continue
                offset = tokens[start].start()
                if (cid, offset) in seen:
                    continue
                seen.add((cid, offset))
                lo = max(0, offset - 160)
                hi = min(len(text), tokens[start + span - 1].end() + 160)
                hits.append((cid, offset, text[lo:hi].replace("\n", " ").strip()))
                break  # longest alias wins at this position
        return hits

    def record_mentions(self, item: SourceItem, author_id: int | None = None) -> int:
        """Persist every concept mention found in an item."""
        hits = self.find_mentions(item.body())
        if not hits:
            return 0
        stance = None
        with self.db.tx() as c:
            for cid, offset, context in hits:
                c.execute(
                    "INSERT OR IGNORE INTO concept_mentions(concept_id,source_item_id,author_id,"
                    "timestamp,context,char_offset,sentiment_or_stance,confidence,extractor)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        cid,
                        item.id,
                        author_id,
                        item.published_at.isoformat() if item.published_at else None,
                        context,
                        offset,
                        stance,
                        1.0,
                        "lexical",
                    ),
                )
        return len(hits)

    def backfill_mentions(self, limit: int | None = None) -> int:
        """Match every stored item that has not been matched yet.

        Re-run after adding concepts: new aliases should apply to history, not
        only to what arrives next.
        """
        sql = (
            "SELECT s.id, s.title, s.text, s.published_at, s.author_id FROM source_items s"
            " WHERE NOT EXISTS (SELECT 1 FROM concept_mentions m WHERE m.source_item_id = s.id)"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.db.conn.execute(sql).fetchall()
        total = 0
        with self.db.tx() as c:
            for row in rows:
                body = "\n".join(p for p in (row["title"], row["text"]) if p)
                for cid, offset, context in self.find_mentions(body):
                    c.execute(
                        "INSERT OR IGNORE INTO concept_mentions(concept_id,source_item_id,"
                        "author_id,timestamp,context,char_offset,sentiment_or_stance,confidence,"
                        "extractor) VALUES(?,?,?,?,?,?,?,?,?)",
                        (cid, row["id"], row["author_id"], row["published_at"], context,
                         offset, None, 1.0, "lexical"),
                    )
                    total += 1
        return total

    def update_first_observed(self) -> None:
        """Set each concept's first observation from OUR corpus.

        Deliberately not taken from an external search API. Algolia returns at
        most 1000 hits per response, so min(created_at) over one page is the
        1000th-most-recent match, not the first -- which would report a
        years-old term as brand new and score it as maximally novel.
        """
        with self.db.tx() as c:
            c.execute(
                "UPDATE concepts SET"
                " first_observed_at = (SELECT MIN(m.timestamp) FROM concept_mentions m"
                "   WHERE m.concept_id = concepts.id AND m.timestamp IS NOT NULL),"
                " first_source_item_id = (SELECT m.source_item_id FROM concept_mentions m"
                "   WHERE m.concept_id = concepts.id AND m.timestamp IS NOT NULL"
                "   ORDER BY m.timestamp LIMIT 1)"
                " WHERE EXISTS (SELECT 1 FROM concept_mentions m WHERE m.concept_id = concepts.id)"
            )

    # -- novel phrase nomination ------------------------------------------

    def nominate_phrases(
        self,
        *,
        recent_days: int = 14,
        min_authors: int = 5,
        min_count: int = 4,
        novelty_ratio: float = 3.0,
        max_nominations: int = 25,
    ) -> list[tuple[str, int, int]]:
        """Find phrases spreading across independent authors that we don't track.

        A phrase qualifies when it is frequent recently, used by several
        different authors, and rare in our older corpus. The author requirement
        is what separates a spreading term from one person's tic.

        Returns (phrase, recent_count, author_count) for the nominations made.
        """
        cutoff = _days_ago_iso(recent_days)
        recent_rows = self.db.conn.execute(
            "SELECT id, title, text, author_id, publication_id FROM source_items"
            " WHERE published_at >= ?",
            (cutoff,),
        ).fetchall()
        older_rows = self.db.conn.execute(
            "SELECT title, text FROM source_items WHERE published_at < ? LIMIT 6000", (cutoff,)
        ).fetchall()

        recent_counts: Counter[str] = Counter()
        phrase_authors: defaultdict[str, set] = defaultdict(set)
        phrase_publications: defaultdict[str, set] = defaultdict(set)
        for row in recent_rows:
            body = normalize("\n".join(p for p in (row["title"], row["text"]) if p))
            # Count documents, not occurrences: one essay using a phrase nine
            # times is one document's worth of evidence.
            for phrase in set(_ngrams(body, 2, 3)):
                recent_counts[phrase] += 1
                phrase_authors[phrase].add(row["author_id"])
                phrase_publications[phrase].add(row["publication_id"])

        older_counts: Counter[str] = Counter()
        for row in older_rows:
            body = normalize("\n".join(p for p in (row["title"], row["text"]) if p))
            for phrase in set(_ngrams(body, 2, 3)):
                older_counts[phrase] += 1

        older_scale = max(1, len(older_rows)) / max(1, len(recent_rows))
        # A phrase appearing in a large share of all recent documents is
        # background vocabulary, not an emerging term.
        ubiquity_ceiling = max(3.0, 0.12 * len(recent_rows))

        nominations: list[tuple[str, int, int]] = []
        for phrase, count in recent_counts.most_common(4000):
            if count < min_count or phrase in self._alias_to_concept:
                continue
            if not is_specific_enough(phrase):
                continue
            if count > ubiquity_ceiling:
                continue
            authors = len(phrase_authors[phrase] - {None})
            if authors < min_authors:
                continue
            # Spreading means several authors AND several venues; one
            # publication repeating itself is not a trend. Crawled site pages
            # from a single org are the main offender here.
            if len(phrase_publications[phrase] - {None}) < 3:
                continue
            expected_old = older_counts.get(phrase, 0) / older_scale
            if expected_old and count / max(expected_old, 0.5) < novelty_ratio:
                continue
            nominations.append((phrase, count, authors))
            if len(nominations) >= max_nominations * 3:
                break

        nominations = _drop_subsumed(nominations)[:max_nominations]

        for phrase, _count, _authors in nominations:
            self.ensure_concept(
                phrase, kind="term", status="nominated", aliases=[phrase], added_by="auto"
            )
        if nominations:
            self.reload()
        return nominations


def _drop_subsumed(
    nominations: list[tuple[str, int, int]], ratio: float = 1.25
) -> list[tuple[str, int, int]]:
    """Collapse overlapping n-grams onto their longest useful form.

    Nominating "navier-stokes millennium", "millennium prize" and
    "navier-stokes millennium prize" separately would create three concepts for
    one idea and split its time series three ways -- precisely the fragmentation
    the registry exists to prevent.

    A shorter phrase is dropped when it is contained in a longer nominated one
    and appears in barely more documents, i.e. it almost never occurs except as
    part of the longer phrase.
    """
    ordered = sorted(nominations, key=lambda n: (-len(n[0].split()), -n[1]))
    kept: list[tuple[str, int, int]] = []
    for phrase, count, authors in ordered:
        subsumed = any(
            f" {phrase} " in f" {longer} " and count <= longer_count * ratio
            for longer, longer_count, _ in kept
        )
        if not subsumed:
            kept.append((phrase, count, authors))
    return sorted(kept, key=lambda n: -n[1])


def _ngrams(text: str, lo: int, hi: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    for n in range(lo, hi + 1):
        for i in range(len(words) - n + 1):
            window = words[i : i + n]
            if window[0] in STOPWORDS or window[-1] in STOPWORDS:
                continue
            out.append(" ".join(window))
    return out


def _days_ago_iso(days: int) -> str:
    from datetime import timedelta

    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
