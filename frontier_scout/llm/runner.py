"""The call runner: caching, budget, concurrency and provenance enforcement.

Every LLM call in the system goes through `AgentRunner.run`. That is
deliberate: it is the single place where a fabricated citation can be caught,
a budget can be enforced, and a cost can be recorded.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..db import Database
from ..models import Claim, LLMCall, validate_citations
from .base import LLMProvider, LLMRequest, LLMResponse
from .cache import CallCache

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class BudgetExhausted(RuntimeError):
    """The run hit its configured spend ceiling."""


@dataclass
class RunStats:
    calls: int = 0
    cache_hits: int = 0
    spend_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    citation_repairs: int = 0
    dropped_claims: int = 0
    failures: list[str] = field(default_factory=list)


class AgentRunner:
    """Runs agent calls with caching, budget control and citation validation."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        db: Database | None = None,
        run_id: str | None = None,
        *,
        use_cache: bool = True,
    ):
        self.provider = provider
        self.settings = settings
        self.db = db
        self.run_id = run_id
        self.cache = CallCache(settings.cache_dir, enabled=use_cache)
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_calls)
        self.budget_usd = settings.daily_llm_budget_usd
        self.stats = RunStats()
        self._lock = asyncio.Lock()

    # -- budget ------------------------------------------------------------

    async def _reserve(self, estimate: float = 0.0) -> None:
        async with self._lock:
            if self.stats.spend_usd + estimate > self.budget_usd:
                raise BudgetExhausted(
                    f"run has spent ${self.stats.spend_usd:.2f} of a "
                    f"${self.budget_usd:.2f} ceiling; refusing further calls"
                )

    async def _record(self, response: LLMResponse, request: LLMRequest) -> None:
        async with self._lock:
            self.stats.calls += 1
            self.stats.spend_usd += response.cost_usd
            self.stats.input_tokens += response.input_tokens
            self.stats.output_tokens += response.output_tokens
            if response.cache_hit:
                self.stats.cache_hits += 1
        if self.db:
            self.db.save_llm_call(
                LLMCall(
                    run_id=self.run_id,
                    agent=request.agent,
                    model=response.model,
                    prompt_version=request.prompt_version,
                    input_hash=request.cache_key()[:16],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    latency_ms=response.latency_ms,
                    cache_hit=response.cache_hit,
                    retry_count=response.retry_count,
                )
            )

    # -- the one call path -------------------------------------------------

    async def run(
        self,
        request: LLMRequest,
        *,
        allowed_source_ids: set[str] | None = None,
        max_attempts: int = 3,
    ) -> T | None:
        """Execute a request and return the validated model, or None on failure.

        `allowed_source_ids` turns on citation enforcement: any claim citing an
        ID that was not in this call's input is a fabricated citation. The model
        gets exactly one chance to repair it, then the offending claims are
        dropped and the drop is logged. This is the mechanism behind the
        provenance guarantee -- reading outputs for plausibility does not scale.
        """
        key = request.cache_key()
        cached = self.cache.get(key)
        if cached is not None:
            try:
                parsed = request.schema.model_validate(cached["data"])
                await self._record(
                    LLMResponse(
                        data=cached["data"],
                        model=cached.get("model", ""),
                        cache_hit=True,
                    ),
                    request,
                )
                return parsed  # type: ignore[return-value]
            except ValidationError:
                pass  # stale cache entry from an older schema; re-run

        feedback = ""
        async with self.semaphore:
            for attempt in range(max_attempts):
                await self._reserve()
                task = request.task if not feedback else f"{request.task}\n\n{feedback}"
                attempt_request = LLMRequest(
                    **{**request.__dict__, "task": task}
                )
                try:
                    response = await self.provider.complete(attempt_request)
                except Exception as exc:  # noqa: BLE001 - surfaced in stats
                    self.stats.retries += 1
                    if attempt == max_attempts - 1:
                        self.stats.failures.append(f"{request.agent}: {exc}")
                        log.warning("%s failed after %d attempts: %s",
                                    request.agent, max_attempts, exc)
                        return None
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue

                response.retry_count = attempt
                await self._record(response, request)

                try:
                    parsed = request.schema.model_validate(response.data)
                except ValidationError as exc:
                    self.stats.retries += 1
                    feedback = (
                        "Your previous response did not match the required schema. "
                        f"Fix these problems and return valid JSON only:\n{exc}"
                    )
                    if attempt == max_attempts - 1:
                        self.stats.failures.append(f"{request.agent}: schema {exc}")
                        return None
                    continue

                if allowed_source_ids is not None:
                    bad = _fabricated_citations(parsed, allowed_source_ids)
                    if bad:
                        if attempt < max_attempts - 1:
                            self.stats.citation_repairs += 1
                            feedback = (
                                "You cited source IDs that were not provided to you: "
                                f"{', '.join(bad[:8])}. Every source_id must appear "
                                "verbatim in the source material above. Re-answer "
                                "using only real source IDs, or mark the claim as "
                                "INFERENCE with an empty source_ids list."
                            )
                            continue
                        # Out of attempts: drop rather than publish a fake citation.
                        dropped = _strip_fabricated(parsed, allowed_source_ids)
                        self.stats.dropped_claims += dropped
                        log.warning(
                            "%s: dropped %d claim(s) with fabricated citations %s",
                            request.agent, dropped, bad[:5],
                        )

                self.cache.put(key, {"data": response.data, "model": response.model})
                return parsed  # type: ignore[return-value]
        return None

    async def gather(self, coros: list[Any]) -> list[Any]:
        """Run agent coroutines concurrently, tolerating individual failures."""
        results = await asyncio.gather(*coros, return_exceptions=True)
        out = []
        for result in results:
            if isinstance(result, Exception):
                self.stats.failures.append(str(result))
                log.warning("agent task failed: %s", result)
                continue
            out.append(result)
        return out

    def cost_report(self) -> dict[str, Any]:
        return {
            "calls": self.stats.calls,
            "cache_hits": self.stats.cache_hits,
            "input_tokens": self.stats.input_tokens,
            "output_tokens": self.stats.output_tokens,
            "estimated_cost_usd": round(self.stats.spend_usd, 4),
            "budget_usd": self.budget_usd,
            "retries": self.stats.retries,
            "citation_repairs": self.stats.citation_repairs,
            "dropped_claims": self.stats.dropped_claims,
            "failures": self.stats.failures[:10],
        }


# -- citation helpers ------------------------------------------------------


def _iter_claims(obj: Any, depth: int = 0) -> list[Claim]:
    """Collect every Claim anywhere in a nested model."""
    if depth > 6:
        return []
    found: list[Claim] = []
    if isinstance(obj, Claim):
        return [obj]
    if isinstance(obj, BaseModel):
        for value in obj.__dict__.values():
            found.extend(_iter_claims(value, depth + 1))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found.extend(_iter_claims(value, depth + 1))
    elif isinstance(obj, dict):
        for value in obj.values():
            found.extend(_iter_claims(value, depth + 1))
    return found


def _fabricated_citations(parsed: BaseModel, allowed: set[str]) -> list[str]:
    return validate_citations(_iter_claims(parsed), allowed)


def _strip_fabricated(parsed: BaseModel, allowed: set[str]) -> int:
    """Remove unknown source IDs in place; demote emptied facts to inference."""
    dropped = 0
    for claim in _iter_claims(parsed):
        unknown = [s for s in claim.source_ids if s not in allowed]
        if not unknown:
            continue
        dropped += 1
        claim.source_ids = [s for s in claim.source_ids if s in allowed]
        if not claim.source_ids:
            # It can no longer be presented as a sourced fact.
            claim.kind = "INFERENCE"  # type: ignore[assignment]
            claim.confidence = min(claim.confidence, 0.3)
    return dropped
