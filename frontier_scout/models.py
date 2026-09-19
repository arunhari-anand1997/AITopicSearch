"""Pydantic schemas for Frontier Scout.

The central invariant of this system is provenance: an editorial claim is
either traceable to a source item we actually ingested, or it is explicitly
labelled an inference. `Claim` enforces that structurally rather than asking a
prompt nicely, and `validate_citations` catches models that cite source IDs
they were never shown.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


# --------------------------------------------------------------------------
# Source taxonomy
# --------------------------------------------------------------------------


class SourceType(StrEnum):
    X = "x"
    SUBSTACK = "substack"
    ARXIV = "arxiv"
    HACKER_NEWS = "hacker_news"
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    COMPANY_BLOG = "company_blog"
    RESEARCH_BLOG = "research_blog"
    PODCAST = "podcast"
    MANUAL = "manual"


class SourceRole(StrEnum):
    """What kind of evidence a source can supply.

    These are deliberately not merged. Three EARLY_SIGNAL sources agreeing is an
    echo chamber; an EARLY_SIGNAL plus PRIMARY_EVIDENCE plus COMMUNITY_REACTION
    is a propagation chain. The trend engine scores those differently.
    """

    EARLY_SIGNAL = "EARLY_SIGNAL"
    PRIMARY_EVIDENCE = "PRIMARY_EVIDENCE"
    INDEPENDENT_ANALYSIS = "INDEPENDENT_ANALYSIS"
    COMMUNITY_REACTION = "COMMUNITY_REACTION"
    PUBLIC_INTEREST = "PUBLIC_INTEREST"
    EXISTING_COVERAGE = "EXISTING_COVERAGE"


class TextCompleteness(StrEnum):
    FULL = "full"
    EXCERPT = "excerpt"
    TITLE_ONLY = "title_only"


class ConnectorStatus(StrEnum):
    """Explicit failure vocabulary.

    A connector that cannot reach its source says so. It never returns
    fabricated or substituted data, and the run continues with reduced
    `data_completeness` rather than silently pretending the source was empty.
    """

    OK = "OK"
    PARTIAL = "PARTIAL"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    ACCESS_DENIED = "ACCESS_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


NOT_AVAILABLE = "NOT_AVAILABLE"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


# --------------------------------------------------------------------------
# Ingested data
# --------------------------------------------------------------------------


class Engagement(BaseModel):
    """One observation of an item's engagement, at a point in time.

    Stored as a time series rather than columns on the item because
    acceleration needs at least two observations to exist.
    """

    observed_at: datetime = Field(default_factory=utcnow)
    likes: int | None = None
    reposts: int | None = None
    replies: int | None = None
    quotes: int | None = None
    views: int | None = None
    score: int | None = None
    comment_count: int | None = None

    def total(self) -> int:
        return sum(
            v or 0
            for v in (self.likes, self.reposts, self.replies, self.quotes, self.score, self.comment_count)
        )


class SourceItem(BaseModel):
    """One ingested artifact: a post, essay, paper, story or video."""

    id: str = ""
    source_type: SourceType
    source_name: str
    external_id: str
    title: str | None = None
    text: str | None = None
    url: str | None = None
    canonical_url: str | None = None
    author_handle: str | None = None
    author_name: str | None = None
    publication: str | None = None
    published_at: datetime | None = None
    collected_at: datetime = Field(default_factory=utcnow)
    source_role: SourceRole
    primary_or_secondary: Literal["PRIMARY", "SECONDARY"] = "SECONDARY"
    paywalled: bool = False
    text_completeness: TextCompleteness = TextCompleteness.FULL
    lang: str | None = None
    engagement: Engagement | None = None
    outbound_links: list[str] = Field(default_factory=list)
    arxiv_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _assign_id(self) -> SourceItem:
        if not self.id:
            digest = hashlib.sha256(f"{self.source_type}|{self.external_id}".encode()).hexdigest()
            self.id = f"{self.source_type}_{digest[:16]}"
        return self

    def body(self) -> str:
        """Title plus text, which is what concept matching runs over."""
        return "\n".join(p for p in (self.title, self.text) if p)


class CollectionResult(BaseModel):
    """What a connector returns. Always includes an honest status."""

    connector: str
    target: str
    status: ConnectorStatus
    items: list[SourceItem] = Field(default_factory=list)
    http_status: int | None = None
    error: str | None = None
    cost_usd: float = 0.0
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    @property
    def ok(self) -> bool:
        return self.status in (ConnectorStatus.OK, ConnectorStatus.PARTIAL)


# --------------------------------------------------------------------------
# Concepts and trends
# --------------------------------------------------------------------------


class ConceptMention(BaseModel):
    concept_id: int
    source_item_id: str
    author_id: int | None = None
    timestamp: datetime | None = None
    context: str | None = None
    char_offset: int | None = None
    sentiment_or_stance: str | None = None
    confidence: float = 1.0
    extractor: str = "lexical"


class TrendComponents(BaseModel):
    """The six momentum components, stored individually.

    Kept separate so weights can be refit against real video performance later.
    That is impossible if only the composite survives. A component that could
    not be computed is None, not zero: 'no baseline yet' must never be
    indistinguishable from 'not accelerating'.
    """

    normalized_velocity: float | None = None
    unique_author_growth: float | None = None
    cross_platform_presence: float | None = None
    respected_source_presence: float | None = None
    engagement_acceleration: float | None = None
    novelty: float | None = None

    def available(self) -> dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ConceptMetrics(BaseModel):
    concept_id: int
    concept_label: str = ""
    computed_at: datetime = Field(default_factory=utcnow)

    mentions_24h: int = 0
    mentions_7d: int = 0
    mentions_30d: int = 0
    unique_authors_24h: int = 0
    unique_authors_7d: int = 0
    unique_authors_prev_7d: int = 0
    unique_publications_7d: int = 0

    independent_source_count: int = 0
    independence_ratio: float = 1.0

    baseline_daily: float | None = None
    baseline_days_available: int = 0
    velocity_ratio: float | None = None
    acceleration: float | None = None

    engagement_sum: float = 0.0
    engagement_median: float = 0.0
    engagement_acceleration: float | None = None

    cross_platform_count: int = 0
    cross_role_count: int = 0
    respected_source_share: float = 0.0
    paper_count: int = 0
    youtube_recent_count: int | None = None
    google_search_change: str | None = NOT_AVAILABLE

    components: TrendComponents = Field(default_factory=TrendComponents)
    insider_momentum_raw: float = 0.0
    insider_momentum: float = 0.0
    data_completeness: float = 1.0
    missing_sources: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SignalPropagation(BaseModel):
    concept_id: int
    origin_source_item_id: str | None = None
    origin_date: datetime | None = None
    platform_sequence: list[dict[str, Any]] = Field(default_factory=list)
    independent_author_count: int = 0
    time_to_cross_platform_hours: float | None = None
    coverage_gap: float | None = None


# --------------------------------------------------------------------------
# Provenance contract for everything an LLM says
# --------------------------------------------------------------------------


class ClaimKind(StrEnum):
    SOURCE_FACT = "SOURCE_FACT"
    INFERENCE = "INFERENCE"


class Claim(BaseModel):
    """A single assertion from an agent.

    A SOURCE_FACT must cite at least one source. An INFERENCE is the model's own
    reasoning and is stored under a different kind so that no rendering path can
    present the two as the same thing.
    """

    text: str
    kind: ClaimKind
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return min(1.0, max(0.0, v))

    @model_validator(mode="after")
    def _require_sources(self) -> Claim:
        if self.kind == ClaimKind.SOURCE_FACT and not self.source_ids:
            raise ValueError("a SOURCE_FACT claim must cite at least one source_id")
        return self


class CitationError(ValueError):
    """Raised when an agent cites source IDs it was never given."""


def validate_citations(claims: list[Claim], allowed_ids: set[str]) -> list[str]:
    """Return the fabricated IDs found across `claims`.

    Catching hallucinated citations structurally is the only way the provenance
    guarantee actually holds; reading the output for plausibility does not scale.
    """
    bad: list[str] = []
    for claim in claims:
        for sid in claim.source_ids:
            if sid not in allowed_ids:
                bad.append(sid)
    return sorted(set(bad))


# --------------------------------------------------------------------------
# Agent outputs
# --------------------------------------------------------------------------


class Signal(BaseModel):
    """A scout's structured observation about a concept group."""

    id: str = ""
    agent: str
    cluster_id: str | None = None
    short_label: str
    claim: str
    concept: str | None = None
    what_is_new: str | None = None
    why_now: str | None = None
    actors: list[str] = Field(default_factory=list)
    evidence_summary: str | None = None
    disagreement_summary: str | None = None
    technical_mechanism: str | None = None
    possible_human_consequence: str | None = None
    possible_economic_consequence: str | None = None
    possible_societal_consequence: str | None = None
    possible_visuals: list[str] = Field(default_factory=list)
    raw_quotes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5

    @model_validator(mode="after")
    def _assign_id(self) -> Signal:
        if not self.id:
            seed = f"{self.agent}|{self.short_label}|{self.claim}"
            self.id = "sig_" + hashlib.sha256(seed.encode()).hexdigest()[:12]
        if not self.source_ids:
            self.source_ids = sorted({s for c in self.claims for s in c.source_ids})
        return self


class ViewPoint(BaseModel):
    """One side of a real disagreement, with real proponents."""

    label: str
    position: str
    proponents: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    what_would_weaken_it: str | None = None


class Cluster(BaseModel):
    """Several surface terms resolved into one underlying change."""

    id: str = ""
    cluster_name: str
    underlying_change: str
    raw_terms: list[str] = Field(default_factory=list)
    concept_ids: list[int] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    evidence: list[Claim] = Field(default_factory=list)
    disagreements: list[ViewPoint] = Field(default_factory=list)
    potential_public_relevance: str | None = None
    breadth_warning: bool = False
    momentum: float = 0.0

    @model_validator(mode="after")
    def _assign_id(self) -> Cluster:
        if not self.id:
            self.id = "clu_" + hashlib.sha256(self.cluster_name.encode()).hexdigest()[:12]
        return self


class Translation(BaseModel):
    insider_term: str
    mechanism: str
    consequence: str
    public_questions: list[str] = Field(default_factory=list)


class CandidateVideo(BaseModel):
    """A proposed video. Must carry both a mechanism and a stake."""

    id: str = ""
    cluster_id: str | None = None
    working_title: str
    alternative_titles: list[str] = Field(default_factory=list)
    central_question: str
    one_sentence_premise: str
    real_current_trigger: str
    why_now: str
    target_viewer: str = "intelligent non-specialist"
    viewer_problem_solved: str | None = None
    technical_mechanism: str
    old_mental_model: str
    new_mental_model: str
    technical_concepts_to_teach: list[str] = Field(default_factory=list)
    stakes: str | None = None
    human_consequence: str
    economic_consequence: str | None = None
    societal_consequence: str | None = None
    human_angle: str | None = None
    credible_disagreement: str
    story_engine: str | None = None
    people_orgs: list[str] = Field(default_factory=list)
    evidence_available: str | None = None
    coverage_gap: str | None = None
    visual_opportunities: list[str] = Field(default_factory=list)
    strongest_counterargument: str | None = None
    what_we_need_to_investigate: list[str] = Field(default_factory=list)
    possible_ending_insight: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    revision_of: str | None = None

    @model_validator(mode="after")
    def _assign_id(self) -> CandidateVideo:
        if not self.id:
            self.id = "cand_" + hashlib.sha256(
                f"{self.cluster_id}|{self.working_title}|{self.revision_of}".encode()
            ).hexdigest()[:12]
        return self


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

SCORE_DIMENSIONS: tuple[str, ...] = (
    "buzz_timeliness",
    "public_curiosity",
    "stakes",
    "novelty",
    "technical_depth",
    "accessibility",
    "human_relevance",
    "serious_disagreement",
    "story_motion",
    "visual_potential",
    "evidence_availability",
    "coverage_gap",
    "retellability",
    "sponsor_audience_quality",
    "expansion_potential",
)


class Evaluation(BaseModel):
    candidate_id: str = ""
    evaluator_model: str = ""
    prompt_version: str = ""
    scores: dict[str, int] = Field(default_factory=dict)
    cares_before_knowing_term: bool = False
    interests_non_ai_person: bool = False
    strongest_failure_reason: str = ""
    viewer_understanding_delta: str = ""
    old_model: str | None = None
    new_model: str | None = None

    @field_validator("scores")
    @classmethod
    def _validate_scores(cls, v: dict[str, int]) -> dict[str, int]:
        out = {}
        for key, val in v.items():
            if key not in SCORE_DIMENSIONS:
                continue
            out[key] = int(min(10, max(1, val)))
        return out

    def mean_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


PAIRWISE_QUESTIONS: dict[str, str] = {
    "publish_next_week": "If you could publish only one next week for this audience, which?",
    "share_with_friend": "Which is more likely to be shared with a smart friend?",
    "mission_fit": "Which better makes the technological frontier legible?",
}


class PairwiseResult(BaseModel):
    candidate_a: str
    candidate_b: str
    question_key: str
    winner: str | None = None
    confidence: float = 0.5
    explanation: str = ""
    judge_model: str = ""
    order_swapped: bool = False


RED_TEAM_ATTACKS: tuple[str, ...] = (
    "merely_a_buzzword",
    "fake_stakes",
    "trend_already_fading",
    "single_origin_cascade",
    "already_well_covered",
    "title_broader_than_evidence",
    "only_interesting_to_developers",
    "investment_mistaken_for_proof",
    "engagement_mistaken_for_importance",
    "old_concept_new_name",
)


class RedTeamFinding(BaseModel):
    candidate_id: str = ""
    attack_key: str
    verdict: Literal["SURVIVES", "WOUNDED", "FATAL"] = "SURVIVES"
    argument: str = ""
    severity: int = 1
    abandon_evidence: str | None = None

    @field_validator("severity")
    @classmethod
    def _clamp_severity(cls, v: int) -> int:
        return int(min(10, max(1, v)))


class HumanReview(BaseModel):
    candidate_id: str
    reviewer: str = "editor"
    decision: Literal["APPROVE", "REJECT", "WATCH", "RESEARCH_MORE"]
    favorite: bool = False
    personal_excitation_score: int | None = None
    would_spend_two_weeks: bool | None = None
    notes: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class LLMCall(BaseModel):
    run_id: str | None = None
    agent: str = ""
    model: str = ""
    prompt_version: str = ""
    input_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cache_hit: bool = False
    retry_count: int = 0
