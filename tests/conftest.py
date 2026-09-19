"""Shared fixtures.

The fake provider lets the entire agent layer be tested deterministically with
no network and no API key -- including the failure paths (malformed output,
fabricated citations) that are hard to trigger against a real model but are
exactly the paths that must not regress.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from frontier_scout.config import Settings
from frontier_scout.db import Database
from frontier_scout.llm.base import LLMProvider, LLMRequest, LLMResponse
from frontier_scout.models import Engagement, SourceItem, SourceRole, SourceType


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings.load()
    s.db_path = tmp_path / "test.db"
    s.cache_dir = tmp_path / "cache"
    s.runs_dir = tmp_path / "runs"
    return s


class FakeProvider(LLMProvider):
    """Scripted provider.

    `responses` maps agent name -> payload (or a list of payloads consumed in
    order, to script a repair sequence). A payload may be an Exception, which
    is raised instead.
    """

    name = "fake"

    def __init__(self, responses: dict[str, Any] | None = None):
        self.responses = responses or {}
        self.calls: list[LLMRequest] = []

    def model_for(self, role: str) -> str:
        return "fake-model"

    def effort_for(self, role: str) -> str:
        return "medium"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        payload = self.responses.get(request.agent, {})
        if isinstance(payload, list):
            payload = payload.pop(0) if payload else {}
        if isinstance(payload, Exception):
            raise payload
        return LLMResponse(
            data=payload,
            model="fake-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            raw_text=json.dumps(payload),
        )


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


def make_item(
    *,
    external_id: str,
    text: str,
    source_type: SourceType = SourceType.SUBSTACK,
    author: str = "writer_a",
    days_ago: int = 1,
    publication: str = "Test Pub",
    links: list[str] | None = None,
    arxiv: list[str] | None = None,
    score: int = 10,
) -> SourceItem:
    return SourceItem(
        source_type=source_type,
        source_name=publication,
        external_id=external_id,
        title=text[:60],
        text=text,
        url=f"https://example.com/{external_id}",
        author_handle=author,
        author_name=author,
        publication=publication,
        published_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        source_role=SourceRole.INDEPENDENT_ANALYSIS,
        engagement=Engagement(score=score),
        outbound_links=links or [],
        arxiv_refs=arxiv or [],
    )


@pytest.fixture
def make_source_item():
    return make_item
