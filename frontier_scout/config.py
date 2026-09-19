"""Configuration.

Monitored sources live in config/*.yaml so they can be changed without touching
Python. Scoring weights live there too, because the weights are editorial
opinion and will be refit against real video performance later.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

USER_AGENT = (
    "FrontierScout/0.1 (editorial research prototype; "
    "+https://github.com/arunhari-anand1997/AITopicSearch)"
)


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class Settings:
    """Runtime settings, assembled from config/editorial.yaml plus the env."""

    db_path: Path = DATA_DIR / "frontier_scout.db"
    runs_dir: Path = DATA_DIR / "runs"
    exports_dir: Path = DATA_DIR / "exports"
    cache_dir: Path = DATA_DIR / "cache"

    # credentials
    anthropic_api_key: str | None = field(default=None, repr=False)
    deepseek_api_key: str | None = field(default=None, repr=False)
    youtube_api_key: str | None = field(default=None, repr=False)
    x_bearer_token: str | None = field(default=None, repr=False)
    google_trends_api_key: str | None = field(default=None, repr=False)

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Settings:
        raw = _load_yaml("editorial.yaml")
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            youtube_api_key=os.getenv("YOUTUBE_API_KEY") or None,
            x_bearer_token=os.getenv("X_BEARER_TOKEN") or None,
            google_trends_api_key=os.getenv("GOOGLE_TRENDS_API_KEY") or None,
            raw=raw,
        )

    # -- sections ----------------------------------------------------------

    def section(self, name: str, default: Any = None) -> Any:
        return self.raw.get(name, default if default is not None else {})

    @property
    def weights(self) -> dict[str, float]:
        return self.section("momentum_weights", {})

    @property
    def opportunity_weights(self) -> dict[str, float]:
        return self.section("opportunity_weights", {})

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.section("pipeline", {})

    @property
    def models(self) -> dict[str, str]:
        return self.section("models", {})

    @property
    def llm_provider(self) -> str:
        """Which provider the agent layer uses: 'anthropic', 'deepseek' or 'auto'."""
        return str(self.section("llm", {}).get("provider", "auto"))

    @property
    def x_config(self) -> dict[str, Any]:
        return self.section("x", {})

    @property
    def gate_width(self) -> int:
        return int(self.pipeline.get("gate_width", 12))

    @property
    def max_concurrent_calls(self) -> int:
        return int(self.pipeline.get("max_concurrent_calls", 6))

    @property
    def backfill_days(self) -> int:
        return int(self.pipeline.get("backfill_days", 90))

    @property
    def daily_llm_budget_usd(self) -> float:
        return float(self.pipeline.get("daily_llm_budget_usd", 15.0))

    # -- source lists ------------------------------------------------------

    @cached_property
    def substack_feeds(self) -> list[dict[str, Any]]:
        return _load_yaml("substack_feeds.yaml").get("feeds", [])

    @cached_property
    def primary_feeds(self) -> list[dict[str, Any]]:
        return _load_yaml("primary_feeds.yaml").get("feeds", [])

    @cached_property
    def arxiv_queries(self) -> dict[str, Any]:
        return _load_yaml("arxiv_queries.yaml")

    @cached_property
    def youtube_queries(self) -> dict[str, Any]:
        return _load_yaml("youtube_queries.yaml")

    @cached_property
    def x_queries(self) -> list[str]:
        return _load_yaml("x_queries.yaml").get("queries", [])

    @cached_property
    def x_accounts(self) -> dict[str, list[str]]:
        return _load_yaml("x_accounts.yaml").get("categories", {})

    @cached_property
    def seed_concepts(self) -> list[dict[str, Any]]:
        return _load_yaml("concepts.yaml").get("concepts", [])

    @cached_property
    def hn_config(self) -> dict[str, Any]:
        return _load_yaml("hacker_news.yaml")

    def config_hash(self) -> str:
        payload = json.dumps(
            {
                "editorial": self.raw,
                "substack": self.substack_feeds,
                "primary": self.primary_feeds,
                "arxiv": self.arxiv_queries,
                "concepts": self.seed_concepts,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text()


def prompt_version(name: str) -> str:
    """Hash of the prompt text, so the cache invalidates when a prompt changes."""
    return hashlib.sha256(load_prompt(name).encode()).hexdigest()[:8]
