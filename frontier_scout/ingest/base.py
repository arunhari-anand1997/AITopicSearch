"""Connector base.

Every connector returns a CollectionResult carrying an explicit status. A
connector that cannot reach its source says SOURCE_UNAVAILABLE and the pipeline
continues with reduced data_completeness. Nothing in this system is permitted
to invent or substitute data to cover a failure.
"""

from __future__ import annotations

import asyncio
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx

from ..config import USER_AGENT, Settings
from ..models import CollectionResult, ConnectorStatus, SourceItem

ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "ref_src", "s", "fbclid", "gclid", "mc_cid", "mc_eid", "triedRedirect",
}


def canonicalize(url: str | None) -> str | None:
    """Strip tracking noise so two links to the same thing compare equal.

    This matters more than it looks: the independence graph joins items by
    shared canonical URL, so sloppy normalisation would let one origin essay
    masquerade as several independent sources.
    """
    if not url:
        return None
    try:
        p = urlparse(url.strip())
    except ValueError:
        return url
    if not p.scheme:
        return url
    query = "&".join(
        part for part in p.query.split("&")
        if part and part.split("=")[0].lower() not in TRACKING_PARAMS
    )
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), netloc, path, "", query, ""))


def extract_arxiv_ids(text: str | None) -> list[str]:
    if not text:
        return []
    return sorted({m.group(1) for m in ARXIV_ID_RE.finditer(text)})


def parse_date(value: Any) -> datetime | None:
    """Parse the many date shapes RSS and JSON APIs produce."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, time.struct_time):
        return datetime.fromtimestamp(time.mktime(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(text, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class RateLimiter:
    """Global minimum spacing between requests.

    Global rather than per-worker on purpose: arXiv's terms bind all machines
    under our control, so a semaphore-per-task would violate them while looking
    correct.
    """

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delta = time.monotonic() - self._last
            if delta < self.min_interval:
                await asyncio.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class RobotsCache:
    """robots.txt per host, fetched once and honoured.

    A host we cannot read robots.txt for is treated as allowed, which matches
    the convention, but a host that explicitly disallows a path is never
    fetched.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._lock = asyncio.Lock()

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        async with self._lock:
            if host not in self._parsers:
                parser: RobotFileParser | None = RobotFileParser()
                try:
                    resp = await client.get(urljoin(host, "/robots.txt"), timeout=15)
                    if resp.status_code == 200:
                        parser.parse(resp.text.splitlines())
                    else:
                        parser = None
                except httpx.HTTPError:
                    parser = None
                self._parsers[host] = parser
        parser = self._parsers[host]
        return True if parser is None else parser.can_fetch(USER_AGENT, url)


class Connector(ABC):
    """Base class for all source connectors."""

    name: str = "connector"
    #: Minimum seconds between requests for this connector.
    min_interval: float = 0.0

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._client = client
        self._owns_client = client is None
        self.limiter = RateLimiter(self.min_interval) if self.min_interval else None
        self.robots = RobotsCache()

    async def __aenter__(self) -> Connector:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=httpx.Timeout(30.0),
                follow_redirects=True,
            )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(f"{self.name}: use the connector as an async context manager")
        return self._client

    async def fetch(self, url: str, *, retries: int = 3, **kwargs: Any) -> httpx.Response:
        """GET with polite spacing and backoff on transient throttling.

        Hosts that are fine with one request at a time will return 403 or 429
        when several arrive at once. Retrying with backoff distinguishes a host
        that is throttling us from one that is genuinely refusing access, so we
        do not record ACCESS_DENIED for what is really our own impatience.
        """
        last: httpx.Response | None = None
        for attempt in range(retries + 1):
            if self.limiter:
                await self.limiter.wait()
            try:
                resp = await self.client.get(url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= retries:
                    raise
                await asyncio.sleep(2 ** attempt)
                continue
            if resp.status_code in (403, 429, 500, 502, 503, 504) and attempt < retries:
                last = resp
                await asyncio.sleep(2 ** attempt)
                continue
            return resp
        return last if last is not None else resp

    def result(
        self,
        target: str,
        status: ConnectorStatus,
        items: list[SourceItem] | None = None,
        **kw: Any,
    ) -> CollectionResult:
        return CollectionResult(
            connector=self.name,
            target=target,
            status=status,
            items=items or [],
            finished_at=datetime.now(timezone.utc),
            **kw,
        )

    def failure(self, target: str, exc: Exception) -> CollectionResult:
        """Map a transport failure onto an honest status.

        A bot challenge is called out by name. It is an access control, so we
        do not attempt to work around it -- but "blocked by a Cloudflare
        challenge, try the publication's own domain" is far more actionable for
        an editor than a bare 403.
        """
        status = ConnectorStatus.SOURCE_UNAVAILABLE
        http_status = None
        detail = f"{type(exc).__name__}: {exc}"

        if isinstance(exc, httpx.HTTPStatusError):
            http_status = exc.response.status_code
            if http_status in (401, 403):
                status = ConnectorStatus.ACCESS_DENIED
                if exc.response.headers.get("cf-mitigated") == "challenge":
                    detail = (
                        "blocked by a Cloudflare bot challenge (cf-mitigated: challenge). "
                        "This is an access control and is not bypassed. Many publications "
                        "serve the same feed from their own domain -- try that instead."
                    )
            elif http_status == 429:
                status = ConnectorStatus.RATE_LIMITED
        return self.result(target, status, error=detail, http_status=http_status)

    @abstractmethod
    async def collect(self) -> list[CollectionResult]:
        """Gather everything this connector is configured to gather."""
