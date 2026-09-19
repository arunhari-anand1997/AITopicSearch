"""Provider abstraction and the LLM call contract.

Providers return structured, schema-validated output. The rest of the system
never sees raw model text, which keeps parsing failures in one place instead of
spread across thirteen agents.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResponse:
    """One completed call: the parsed payload plus what it cost."""

    data: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cache_hit: bool = False
    retry_count: int = 0
    stop_reason: str | None = None
    raw_text: str = ""


@dataclass
class LLMRequest:
    """A single structured request.

    `shared_context` exists for prompt caching. The eight scouts all reason over
    the same source dossier, so that block goes first and is marked cacheable;
    the agent-specific instruction follows it. Caching is a prefix match, so
    shared-then-specific is the only ordering that earns a cache hit.
    """

    agent: str
    instructions: str
    task: str
    schema: type[BaseModel]
    shared_context: str = ""
    prompt_version: str = ""
    effort: str = "high"
    max_tokens: int = 16000
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def agent_role(self) -> str:
        """Pipeline role used to pick model and effort.

        Defaults to the agent name; scouts share the "scout" role so they all
        resolve to the same (cheaper) model.
        """
        return self.metadata.get("role", self.agent)

    def cache_key(self) -> str:
        """Content hash over everything that could change the answer."""
        payload = json.dumps(
            {
                "agent": self.agent,
                "prompt_version": self.prompt_version,
                "instructions": self.instructions,
                "task": self.task,
                "shared_context": self.shared_context,
                "schema": self.schema.model_json_schema(),
                "effort": self.effort,
                "temperature": self.temperature,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class LLMProvider(ABC):
    """Swap-in point for a different model vendor."""

    name: str = "provider"

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one structured request."""

    @abstractmethod
    def model_for(self, role: str) -> str:
        """Resolve a pipeline role (scout, generator, evaluator...) to a model."""


class LLMUnavailable(RuntimeError):
    """Raised when no credential is configured.

    Surfaced as NOT_CONFIGURED so a run without an API key degrades to the
    quantitative layer instead of dying.
    """
