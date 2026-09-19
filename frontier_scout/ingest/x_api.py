"""X connector. OFF BY DEFAULT, and deliberately hard to switch on carelessly.

WHY THIS FILE IS SO DEFENSIVE
-----------------------------
X has no free tier. Reads are billed PER RETURNED RESOURCE at roughly $0.005
per post, so a single search returning 100 posts costs about $0.50, and the
nine standing queries in config/x_queries.yaml run daily would cost roughly
$135/month before any account timelines. A careless config here spends real
money quietly, so the connector refuses to run without explicit budget ceilings
and estimates the cost of every call before making it.

Three facts drive the design:

1. Billing is per returned resource, not per request. `max_results` IS the cost
   dial, so it is checked against the remaining budget before each call.

2. Resources are deduplicated within a 24-hour UTC window -- re-reading the
   same post on the same UTC day is free. So collection runs ONCE PER UTC DAY
   with an aggressive local cache, never on a poll.

3. Recent search covers 7 days; full-archive is restricted and may require
   Enterprise. X therefore cannot supply its own 28-day baseline, so X velocity
   reports INSUFFICIENT_HISTORY until our own database has grown one. We never
   compute a ratio against a baseline we do not have.

Endpoint availability differs by account, so the connector PROBES what it
actually has and records the answer rather than assuming.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..models import (
    CollectionResult,
    ConnectorStatus,
    Engagement,
    SourceItem,
    SourceRole,
    SourceType,
)
from .base import Connector, canonicalize, extract_arxiv_ids, parse_date

API = "https://api.x.com/2"

#: Published read pricing, used to estimate spend BEFORE a call is made.
PRICE_PER_POST = 0.005
PRICE_PER_USER = 0.010

TWEET_FIELDS = "created_at,public_metrics,author_id,referenced_tweets,entities,lang"
USER_FIELDS = "username,name,public_metrics,verified"


class XBudgetError(RuntimeError):
    """Raised when a call would exceed the configured spend ceiling."""


class XConnector(Connector):
    name = "x"
    min_interval = 1.0

    def __init__(self, settings, client=None):
        super().__init__(settings, client)
        self.config = settings.x_config or {}
        self.spent_usd = 0.0
        self.capabilities: dict[str, bool | str] = {}

    # -- guards ------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def daily_budget(self) -> float:
        return float(self.config.get("daily_budget_usd", 0.0))

    def _check_configured(self) -> CollectionResult | None:
        if not self.enabled:
            return self.result(
                "x", ConnectorStatus.NOT_CONFIGURED,
                error="X is disabled in config/editorial.yaml (x.enabled: false). "
                      "X has no free tier; enabling it spends real money.",
            )
        if not self.settings.x_bearer_token:
            return self.result(
                "x", ConnectorStatus.NOT_CONFIGURED,
                error="X_BEARER_TOKEN is not set",
            )
        if self.daily_budget <= 0:
            return self.result(
                "x", ConnectorStatus.NOT_CONFIGURED,
                error="x.daily_budget_usd must be set above 0 before X can run. "
                      "Refusing to make paid calls without an explicit ceiling.",
            )
        return None

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.x_bearer_token}"}

    def _reserve(self, estimated_posts: int) -> None:
        """Refuse a call that would break the ceiling, before making it."""
        estimate = estimated_posts * PRICE_PER_POST
        if self.spent_usd + estimate > self.daily_budget:
            raise XBudgetError(
                f"call would cost about ${estimate:.2f}; ${self.spent_usd:.2f} of "
                f"${self.daily_budget:.2f} already spent today"
            )

    # -- capability probe --------------------------------------------------

    async def probe(self) -> dict[str, bool | str]:
        """Find out what this account can actually do.

        Access differs by account and changes over time; full-archive search in
        particular is documented as restricted and may require Enterprise. A
        1-result probe is the cheapest honest way to find out, and the answer is
        recorded rather than assumed.
        """
        if self._check_configured():
            return {"configured": False}

        capabilities: dict[str, bool | str] = {"configured": True}
        declared = self.config.get("endpoints", {}) or {}

        for endpoint, path in (
            ("recent_search", "/tweets/search/recent"),
            ("full_archive_search", "/tweets/search/all"),
        ):
            if not declared.get(endpoint, endpoint == "recent_search"):
                capabilities[endpoint] = "disabled_in_config"
                continue
            try:
                self._reserve(1)
                resp = await self.fetch(
                    f"{API}{path}",
                    params={"query": "ai -is:retweet", "max_results": 10},
                    headers=self._headers(),
                )
                self.spent_usd += 10 * PRICE_PER_POST
                capabilities[endpoint] = resp.status_code == 200
                if resp.status_code in (401, 403):
                    capabilities[endpoint] = f"denied_{resp.status_code}"
            except XBudgetError as exc:
                capabilities[endpoint] = f"budget: {exc}"
            except Exception as exc:  # noqa: BLE001
                capabilities[endpoint] = f"error: {type(exc).__name__}"

        self.capabilities = capabilities
        return capabilities

    # -- collection --------------------------------------------------------

    async def collect(self) -> list[CollectionResult]:
        guard = self._check_configured()
        if guard:
            return [guard]

        results: list[CollectionResult] = []
        max_results = int(self.config.get("max_results_per_query", 50))
        for query in self.settings.x_queries:
            results.append(await self.search(query, max_results=max_results))
        return results

    async def search(self, query: str, *, max_results: int = 50) -> CollectionResult:
        guard = self._check_configured()
        if guard:
            return guard

        try:
            self._reserve(max_results)
        except XBudgetError as exc:
            return self.result(
                f"query:{query}", ConnectorStatus.BUDGET_EXHAUSTED, error=str(exc)
            )

        try:
            resp = await self.fetch(
                f"{API}/tweets/search/recent",
                params={
                    "query": query,
                    "max_results": min(max_results, 100),
                    "tweet.fields": TWEET_FIELDS,
                    "expansions": "author_id",
                    "user.fields": USER_FIELDS,
                },
                headers=self._headers(),
            )
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return self.failure(f"query:{query}", exc)

        payload = resp.json()
        posts = payload.get("data", []) or []
        users = {
            u["id"]: u for u in payload.get("includes", {}).get("users", []) or []
        }
        cost = len(posts) * PRICE_PER_POST
        self.spent_usd += cost

        items = [self._to_item(post, users, query) for post in posts]
        return self.result(
            f"query:{query}", ConnectorStatus.OK, items,
            cost_usd=cost, http_status=resp.status_code,
        )

    def _to_item(
        self, post: dict[str, Any], users: dict[str, Any], query: str
    ) -> SourceItem:
        metrics = post.get("public_metrics", {}) or {}
        author = users.get(post.get("author_id", ""), {})
        entities = post.get("entities", {}) or {}
        urls = [
            canonicalize(u.get("expanded_url") or u.get("url"))
            for u in entities.get("urls", []) or []
        ]
        # Quote/retweet ancestry: the independence graph uses this to collapse
        # a cascade back onto its origin.
        referenced = [
            {"type": r.get("type"), "id": r.get("id")}
            for r in post.get("referenced_tweets", []) or []
        ]
        handle = author.get("username")
        curated = self._curated_category(handle)

        return SourceItem(
            source_type=SourceType.X,
            source_name="X",
            external_id=post["id"],
            text=post.get("text"),
            url=f"https://x.com/{handle or 'i'}/status/{post['id']}",
            author_handle=handle,
            author_name=author.get("name"),
            publication="X",
            published_at=parse_date(post.get("created_at")),
            source_role=SourceRole.EARLY_SIGNAL,
            primary_or_secondary="SECONDARY",
            lang=post.get("lang"),
            engagement=Engagement(
                likes=metrics.get("like_count"),
                reposts=metrics.get("retweet_count"),
                replies=metrics.get("reply_count"),
                quotes=metrics.get("quote_count"),
                views=metrics.get("impression_count"),
            ),
            outbound_links=[u for u in urls if u],
            arxiv_refs=extract_arxiv_ids(" ".join(u or "" for u in urls) + " " + (post.get("text") or "")),
            metadata={
                "matched_query": query,
                "referenced_tweets": referenced,
                # Follower count is ONE observable field. It is never treated as
                # authority: authority comes from the curated lists below.
                "author_followers": (author.get("public_metrics", {}) or {}).get(
                    "followers_count"
                ),
                "curated_category": curated,
                # Tier 1 only for editor-curated accounts; never inferred.
                "curated_tier": 1 if curated else None,
            },
        )

    def _curated_category(self, handle: str | None) -> str | None:
        if not handle:
            return None
        lowered = handle.lower()
        for category, handles in (self.settings.x_accounts or {}).items():
            if any(lowered == str(h).lower().lstrip("@") for h in handles or []):
                return category
        return None
