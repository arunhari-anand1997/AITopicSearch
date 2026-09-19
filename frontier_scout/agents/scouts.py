"""The eight scouts.

Each answers one narrow question over the same evidence, and none of them sees
another's output. That independence is the point: eight narrow views that can
disagree are more informative than one broad view that cannot.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..models import Claim, Signal, ViewPoint
from .base import Agent
from .dossier import ConceptGroup

# --------------------------------------------------------------------------
# Trend Scout
# --------------------------------------------------------------------------


class TrendObservation(BaseModel):
    label: str
    movement_kind: Literal[
        "NEW_TERM", "NEW_IDEA", "OLD_IDEA_NEW_LABEL", "CAPABILITY_CHANGE", "DECAYING"
    ]
    what_is_moving: str
    underlying_movement: str = Field(
        description="The change beneath the vocabulary, not the vocabulary itself"
    )
    older_name: str | None = Field(
        default=None, description="If OLD_IDEA_NEW_LABEL, what it used to be called"
    )
    sense_shift_note: str | None = Field(
        default=None,
        description="If the phrase is old but its meaning has changed, explain",
    )
    evidence: list[Claim] = Field(default_factory=list)
    confidence: float = 0.5


class TrendScoutOutput(BaseModel):
    observations: list[TrendObservation] = Field(default_factory=list)
    decaying_terms: list[str] = Field(
        default_factory=list,
        description="Terms that look inactive. A dying term is a real finding.",
    )
    overall_read: str = ""


class TrendScout(Agent[TrendScoutOutput]):
    prompt_name = "trend_scout"
    role = "scout"
    schema = TrendScoutOutput

    def task(self, group: ConceptGroup, **_: object) -> str:
        return (
            f"Concepts in this group: {', '.join(group.labels)}.\n\n"
            "What is actually moving here, and what kind of movement is it? "
            "Classify each observation, and name any term that looks like it is "
            "decaying rather than rising."
        )


# --------------------------------------------------------------------------
# Frontier Scout
# --------------------------------------------------------------------------


class TechnicalChange(BaseModel):
    what_changed: str
    change_kind: Literal[
        "new_capability", "cost_change", "architecture", "workflow",
        "benchmark", "constraint", "research_result", "none_detected",
    ]
    is_real_change_or_just_language: Literal[
        "REAL_CHANGE", "MOSTLY_LANGUAGE", "INSUFFICIENT_EVIDENCE"
    ]
    reasoning: str
    evidence: list[Claim] = Field(default_factory=list)
    what_evidence_would_settle_it: str | None = None
    confidence: float = 0.5


class FrontierScoutOutput(BaseModel):
    changes: list[TechnicalChange] = Field(default_factory=list)
    summary: str = ""


class FrontierScout(Agent[FrontierScoutOutput]):
    prompt_name = "frontier_scout"
    role = "scout"
    schema = FrontierScoutOutput

    def task(self, group: ConceptGroup, **_: object) -> str:
        return (
            f"Concepts: {', '.join(group.labels)}.\n\n"
            "What actually changed technically? For each candidate change, judge "
            "whether it is a real capability change or mostly new vocabulary for "
            "an existing practice."
        )


# --------------------------------------------------------------------------
# Disagreement Scout
# --------------------------------------------------------------------------


class Disagreement(BaseModel):
    question_at_issue: str
    views: list[ViewPoint] = Field(default_factory=list)
    is_empirical_or_values: Literal["EMPIRICAL", "VALUES", "VOCABULARY"]
    why_it_matters_publicly: str
    confidence: float = 0.5


class DisagreementScoutOutput(BaseModel):
    disagreements: list[Disagreement] = Field(default_factory=list)
    consensus_note: str = Field(
        default="",
        description="If the sources broadly agree, say so. Agreement is a finding.",
    )


class DisagreementScout(Agent[DisagreementScoutOutput]):
    prompt_name = "disagreement_scout"
    role = "scout"
    schema = DisagreementScoutOutput

    def task(self, group: ConceptGroup, **_: object) -> str:
        return (
            f"Concepts: {', '.join(group.labels)}.\n\n"
            "Where do credible people in these sources genuinely disagree? Only "
            "report a disagreement if you can cite specific people taking "
            "specific differing positions. If they agree, say they agree."
        )


# --------------------------------------------------------------------------
# Human Consequence Scout
# --------------------------------------------------------------------------


class Consequence(BaseModel):
    domain: str
    who_is_affected: str
    concrete_change: str = Field(description="Specific. Never 'society will change'.")
    conditional_on: str = Field(description="What must be true for this to follow")
    time_horizon: Literal["already_happening", "1-2_years", "3-5_years", "speculative"]
    is_second_order: bool = False
    evidence: list[Claim] = Field(default_factory=list)
    confidence: float = 0.5


class HumanConsequenceScoutOutput(BaseModel):
    consequences: list[Consequence] = Field(default_factory=list)
    strongest_single_consequence: str = ""


class HumanConsequenceScout(Agent[HumanConsequenceScoutOutput]):
    prompt_name = "human_consequence_scout"
    role = "scout"
    schema = HumanConsequenceScoutOutput

    def task(self, group: ConceptGroup, **_: object) -> str:
        return (
            f"Concepts: {', '.join(group.labels)}.\n\n"
            "If this is real, what concretely changes for someone outside this "
            "field? Be specific enough that a viewer could recognise it happening."
        )


# --------------------------------------------------------------------------
# Mechanism Scout
# --------------------------------------------------------------------------


class Mechanism(BaseModel):
    name: str
    plain_explanation: str
    why_it_matters_here: str
    analogy_candidates: list[str] = Field(default_factory=list)
    where_the_analogy_breaks: str | None = None
    common_misconception: str | None = None
    difficulty: Literal["easy", "moderate", "hard"] = "moderate"


class MechanismScoutOutput(BaseModel):
    primary_mechanism: Mechanism | None = None
    supporting_mechanisms: list[Mechanism] = Field(default_factory=list)
    the_real_concept_note: str = Field(
        default="",
        description="'People argue about X but need to understand Y' -- if it applies",
    )


class MechanismScout(Agent[MechanismScoutOutput]):
    prompt_name = "mechanism_scout"
    role = "scout"
    schema = MechanismScoutOutput

    def task(self, group: ConceptGroup, **_: object) -> str:
        return (
            f"Concepts: {', '.join(group.labels)}.\n\n"
            "What is the ONE technical idea an intelligent non-specialist must "
            "understand for this story to land? Explain it without jargon."
        )


# --------------------------------------------------------------------------
# Capital Scout
# --------------------------------------------------------------------------


class CapitalSignal(BaseModel):
    who_is_spending: str
    what_they_are_betting_on: str = Field(
        description="Phrase as a bet, never as an established fact"
    )
    signal_kind: Literal[
        "capex", "acquisition", "hiring", "infrastructure", "enterprise_adoption",
        "government", "startup_formation", "strategic_pivot",
    ]
    scale_if_known: str | None = None
    is_obscure_bottleneck: bool = Field(
        default=False,
        description="True if money is concentrating on something the public has not heard of",
    )
    evidence: list[Claim] = Field(default_factory=list)
    confidence: float = 0.5


class CapitalScoutOutput(BaseModel):
    signals: list[CapitalSignal] = Field(default_factory=list)
    note: str = ""


class CapitalScout(Agent[CapitalScoutOutput]):
    prompt_name = "capital_scout"
    role = "scout"
    schema = CapitalScoutOutput

    def task(self, group: ConceptGroup, **_: object) -> str:
        return (
            f"Concepts: {', '.join(group.labels)}.\n\n"
            "What are serious organisations spending money or strategic effort "
            "on here? Remember: capital is evidence of belief, not of truth."
        )


# --------------------------------------------------------------------------
# Story Scout
# --------------------------------------------------------------------------


class StoryPossibility(BaseModel):
    story: str
    story_kind: Literal[
        "person", "company_conflict", "race", "failure", "experiment",
        "bet_against_consensus", "demonstration", "weird_implementation",
        "historical_rhyme",
    ]
    who_is_involved: list[str] = Field(default_factory=list)
    what_is_in_motion: str = Field(description="What changes from start to end")
    visual_potential: str | None = None
    evidence: list[Claim] = Field(default_factory=list)


class StoryScoutOutput(BaseModel):
    possibilities: list[StoryPossibility] = Field(default_factory=list)
    is_currently_only_an_argument: bool = Field(
        default=False,
        description="True if the sources hold only abstract discussion, no human specifics",
    )
    research_needed: list[str] = Field(default_factory=list)


class StoryScout(Agent[StoryScoutOutput]):
    prompt_name = "story_scout"
    role = "scout"
    schema = StoryScoutOutput

    def task(self, group: ConceptGroup, **_: object) -> str:
        return (
            f"Concepts: {', '.join(group.labels)}.\n\n"
            "What makes this a story rather than an explainer? Find people, "
            "conflicts, demonstrations and reversals. If there is only argument "
            "and no human specifics, say so."
        )


# --------------------------------------------------------------------------
# Coverage Scout
# --------------------------------------------------------------------------


class CoverageAssessment(BaseModel):
    content_kind: str
    title: str
    channel: str | None = None
    views: int | None = None
    audience: Literal["developers", "general", "practitioners", "unclear"] = "unclear"
    counts_toward_saturation: bool = False


class CoverageScoutOutput(BaseModel):
    coverage_data_available: bool = False
    assessments: list[CoverageAssessment] = Field(default_factory=list)
    general_audience_saturation: float | None = Field(
        default=None,
        description="0 = nobody has explained this to non-specialists, 1 = thoroughly covered. "
                    "null when no coverage data was supplied -- never guess.",
    )
    gap_description: str = ""
    demand_signal_note: str = Field(
        default="",
        description="High tutorial volume can validate interest while leaving the gap open",
    )


class CoverageScout(Agent[CoverageScoutOutput]):
    prompt_name = "coverage_scout"
    role = "scout"
    schema = CoverageScoutOutput

    def task(self, group: ConceptGroup, *, youtube_block: str = "", **_: object) -> str:
        if not youtube_block:
            return (
                f"Concepts: {', '.join(group.labels)}.\n\n"
                "NO YOUTUBE DATA WAS SUPPLIED for this run (the YouTube connector "
                "is not configured). Set coverage_data_available to false and "
                "general_audience_saturation to null. Do not guess what exists on "
                "YouTube. You may still describe, from the source material alone, "
                "what a general-audience explanation would need to cover."
            )
        return (
            f"Concepts: {', '.join(group.labels)}.\n\n"
            f"EXISTING YOUTUBE COVERAGE:\n{youtube_block}\n\n"
            "Classify each video and assess whether a strong general-audience "
            "explanation already exists. Remember that tutorial volume is demand, "
            "not saturation."
        )


ALL_SCOUTS: tuple[type[Agent], ...] = (
    TrendScout,
    FrontierScout,
    DisagreementScout,
    HumanConsequenceScout,
    MechanismScout,
    CapitalScout,
    StoryScout,
    CoverageScout,
)


def to_signals(agent_name: str, output: BaseModel, cluster_id: str | None) -> list[Signal]:
    """Flatten a scout's structured output into Signal rows.

    Signals are the common currency the clusterer and generator read, and the
    join table `signal_sources` is what keeps every one of them attached to the
    evidence that produced it.
    """
    signals: list[Signal] = []

    def add(label: str, claim_text: str, **kw: object) -> None:
        signals.append(
            Signal(
                agent=agent_name,
                cluster_id=cluster_id,
                short_label=label[:120],
                claim=claim_text,
                **kw,  # type: ignore[arg-type]
            )
        )

    if isinstance(output, TrendScoutOutput):
        for obs in output.observations:
            add(obs.label, obs.underlying_movement, what_is_new=obs.what_is_moving,
                claims=obs.evidence, confidence=obs.confidence,
                tags=[obs.movement_kind])
    elif isinstance(output, FrontierScoutOutput):
        for change in output.changes:
            add(change.change_kind, change.what_changed,
                technical_mechanism=change.reasoning, claims=change.evidence,
                confidence=change.confidence,
                tags=[change.is_real_change_or_just_language])
    elif isinstance(output, DisagreementScoutOutput):
        for dis in output.disagreements:
            add(dis.question_at_issue, dis.why_it_matters_publicly,
                disagreement_summary=" | ".join(
                    f"{v.label}: {v.position}" for v in dis.views
                ),
                source_ids=sorted({s for v in dis.views for s in v.source_ids}),
                confidence=dis.confidence, tags=[dis.is_empirical_or_values])
    elif isinstance(output, HumanConsequenceScoutOutput):
        for cons in output.consequences:
            add(cons.domain, cons.concrete_change,
                possible_human_consequence=cons.concrete_change,
                why_now=cons.conditional_on, claims=cons.evidence,
                confidence=cons.confidence, tags=[cons.time_horizon])
    elif isinstance(output, MechanismScoutOutput):
        for mech in filter(None, [output.primary_mechanism, *output.supporting_mechanisms]):
            add(mech.name, mech.plain_explanation,
                technical_mechanism=mech.why_it_matters_here,
                tags=[mech.difficulty])
    elif isinstance(output, CapitalScoutOutput):
        for sig in output.signals:
            add(sig.who_is_spending, sig.what_they_are_betting_on,
                possible_economic_consequence=sig.scale_if_known,
                claims=sig.evidence, confidence=sig.confidence,
                tags=[sig.signal_kind] + (["obscure_bottleneck"] if sig.is_obscure_bottleneck else []))
    elif isinstance(output, StoryScoutOutput):
        for poss in output.possibilities:
            add(poss.story_kind, poss.story, actors=poss.who_is_involved,
                possible_visuals=[poss.visual_potential] if poss.visual_potential else [],
                claims=poss.evidence, tags=[poss.story_kind])
    elif isinstance(output, CoverageScoutOutput):
        add("coverage", output.gap_description or "no coverage assessment",
            evidence_summary=output.demand_signal_note,
            tags=["coverage_available" if output.coverage_data_available else "coverage_unavailable"])

    return signals
