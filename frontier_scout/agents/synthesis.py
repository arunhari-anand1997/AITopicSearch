"""Clusterer, translator and candidate generator.

These run after the scouts and differ from them in one important way: they see
the scouts' output as well as the source material. That is intentional -- the
scouts are kept independent so they can disagree; synthesis is where the
disagreement gets resolved, visibly.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ..config import load_prompt, prompt_version
from ..llm.base import LLMRequest
from ..models import CandidateVideo, Claim, Cluster, Signal, Translation, ViewPoint
from .base import PROVENANCE_RULES, Agent
from .dossier import ConceptGroup, build_dossier, source_ids_for


def _scout_digest(signals: list[Signal], limit: int = 60) -> str:
    """Render scout findings compactly, grouped by agent."""
    by_agent: dict[str, list[Signal]] = {}
    for sig in signals:
        by_agent.setdefault(sig.agent, []).append(sig)

    lines = ["# SCOUT FINDINGS", ""]
    lines.append(
        "These are eight independent agents' readings of the same evidence. "
        "They did not see each other's work, so where they disagree, that "
        "disagreement is information -- do not average it away."
    )
    lines.append("")
    for agent, items in by_agent.items():
        lines.append(f"## {agent}")
        for sig in items[:limit]:
            tags = f" [{', '.join(sig.tags)}]" if sig.tags else ""
            cites = f" (sources: {', '.join(sig.source_ids[:4])})" if sig.source_ids else ""
            lines.append(f"- **{sig.short_label}**{tags}: {sig.claim}{cites}")
            if sig.disagreement_summary:
                lines.append(f"  - views: {sig.disagreement_summary}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Clusterer
# --------------------------------------------------------------------------


class ClusterOutput(BaseModel):
    cluster_name: str
    underlying_change: str = Field(
        description="One sentence about how the world is moving, minimal jargon"
    )
    raw_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(
        default_factory=list,
        description="Terms deliberately left out (e.g. decaying ones), with reason",
    )
    old_mental_model: str = ""
    evidence: list[Claim] = Field(default_factory=list)
    disagreements: list[ViewPoint] = Field(default_factory=list)
    potential_public_relevance: str = ""
    breadth_warning: bool = Field(
        default=False, description="True if this cluster is too broad to be useful"
    )


class Clusterer(Agent[ClusterOutput]):
    prompt_name = "clusterer"
    role = "clusterer"
    schema = ClusterOutput

    def task(self, group: ConceptGroup, *, signals: list[Signal] | None = None, **_: Any) -> str:
        digest = _scout_digest(signals or [])
        return (
            f"{digest}\n\n"
            f"Concept family: {', '.join(group.labels)}.\n\n"
            "Are these manifestations of one underlying change? If so, name that "
            "change in plain language. Exclude anything the scouts flagged as "
            "decaying, but say whether it belongs as the OLD MENTAL MODEL."
        )

    async def run(  # type: ignore[override]
        self, group: ConceptGroup, *, signals: list[Signal] | None = None, **kwargs: Any
    ) -> ClusterOutput | None:
        return await self._run_with(group, self.task(group, signals=signals))

    async def _run_with(self, group: ConceptGroup, task: str) -> Any:
        request = LLMRequest(
            agent=self.name,
            instructions=load_prompt(self.prompt_name) + PROVENANCE_RULES,
            task=task,
            schema=self.schema,
            shared_context=build_dossier(self.db, group),
            prompt_version=prompt_version(self.prompt_name),
            effort=self.runner.provider.effort_for(self.role),
            metadata={"role": self.role, "group": group.key},
        )
        return await self.runner.run(request, allowed_source_ids=source_ids_for(self.db, group))


# --------------------------------------------------------------------------
# Translator
# --------------------------------------------------------------------------


class TranslatorOutput(BaseModel):
    translations: list[Translation] = Field(default_factory=list)
    best_public_questions: list[str] = Field(
        default_factory=list, description="Strongest first, jargon-free"
    )


class Translator(Agent[TranslatorOutput]):
    prompt_name = "translator"
    role = "translator"
    schema = TranslatorOutput

    def task(self, group: ConceptGroup, *, cluster: ClusterOutput | None = None, **_: Any) -> str:
        change = cluster.underlying_change if cluster else ", ".join(group.labels)
        return (
            f"UNDERLYING CHANGE: {change}\n"
            f"INSIDER TERMS: {', '.join(group.labels)}\n\n"
            "Translate this into public-facing questions an intelligent "
            "non-specialist would want answered. 5-10 questions, no jargon, "
            "ordered strongest first."
        )

    async def run(  # type: ignore[override]
        self, group: ConceptGroup, *, cluster: ClusterOutput | None = None, **kwargs: Any
    ) -> TranslatorOutput | None:
        request = LLMRequest(
            agent=self.name,
            instructions=load_prompt(self.prompt_name) + PROVENANCE_RULES,
            task=self.task(group, cluster=cluster),
            schema=self.schema,
            shared_context=build_dossier(self.db, group),
            prompt_version=prompt_version(self.prompt_name),
            effort=self.runner.provider.effort_for(self.role),
            metadata={"role": self.role, "group": group.key},
        )
        return await self.runner.run(request)


# --------------------------------------------------------------------------
# Candidate generator
# --------------------------------------------------------------------------


class GeneratedCandidate(BaseModel):
    working_title: str
    alternative_titles: list[str] = Field(default_factory=list)
    central_question: str
    one_sentence_premise: str
    real_current_trigger: str
    why_now: str
    viewer_problem_solved: str
    technical_mechanism: str
    old_mental_model: str
    new_mental_model: str
    technical_concepts_to_teach: list[str] = Field(default_factory=list)
    stakes: str
    human_consequence: str
    economic_consequence: str | None = None
    societal_consequence: str | None = None
    human_angle: str | None = None
    credible_disagreement: str
    story_engine: str
    people_orgs: list[str] = Field(default_factory=list)
    evidence_available: str
    coverage_gap: str
    visual_opportunities: list[str] = Field(default_factory=list)
    strongest_counterargument: str
    what_we_need_to_investigate: list[str] = Field(default_factory=list)
    possible_ending_insight: str
    source_ids: list[str] = Field(default_factory=list)


class CandidateGeneratorOutput(BaseModel):
    candidates: list[GeneratedCandidate] = Field(default_factory=list)


class CandidateGenerator(Agent[CandidateGeneratorOutput]):
    prompt_name = "candidate_generator"
    role = "generator"
    schema = CandidateGeneratorOutput

    def task(  # type: ignore[override]
        self,
        group: ConceptGroup,
        *,
        cluster: ClusterOutput | None = None,
        translation: TranslatorOutput | None = None,
        signals: list[Signal] | None = None,
        count: int = 3,
        **_: Any,
    ) -> str:
        parts = [_scout_digest(signals or [])]
        if cluster:
            parts.append(
                f"# UNDERLYING CHANGE\n{cluster.underlying_change}\n\n"
                f"Old mental model: {cluster.old_mental_model}\n"
                f"Excluded as not currently moving: {', '.join(cluster.excluded_terms) or 'none'}"
            )
        if translation and translation.best_public_questions:
            parts.append(
                "# CANDIDATE PUBLIC QUESTIONS\n"
                + "\n".join(f"- {q}" for q in translation.best_public_questions)
            )
        parts.append(
            f"Generate {count} video candidates, strongest first. Each must pass "
            "the two-part test: something worth learning AND something at stake."
        )
        return "\n\n".join(parts)

    async def run(  # type: ignore[override]
        self,
        group: ConceptGroup,
        *,
        cluster: ClusterOutput | None = None,
        translation: TranslatorOutput | None = None,
        signals: list[Signal] | None = None,
        count: int = 3,
        **kwargs: Any,
    ) -> CandidateGeneratorOutput | None:
        request = LLMRequest(
            agent=self.name,
            instructions=load_prompt(self.prompt_name) + PROVENANCE_RULES,
            task=self.task(
                group, cluster=cluster, translation=translation,
                signals=signals, count=count,
            ),
            schema=self.schema,
            shared_context=build_dossier(self.db, group),
            prompt_version=prompt_version(self.prompt_name),
            effort=self.runner.provider.effort_for(self.role),
            max_tokens=24000,
            metadata={"role": self.role, "group": group.key},
        )
        return await self.runner.run(request)


def to_candidate_video(
    gen: GeneratedCandidate, cluster_id: str | None, fallback_sources: list[str]
) -> CandidateVideo:
    return CandidateVideo(
        cluster_id=cluster_id,
        working_title=gen.working_title,
        alternative_titles=gen.alternative_titles,
        central_question=gen.central_question,
        one_sentence_premise=gen.one_sentence_premise,
        real_current_trigger=gen.real_current_trigger,
        why_now=gen.why_now,
        viewer_problem_solved=gen.viewer_problem_solved,
        technical_mechanism=gen.technical_mechanism,
        old_mental_model=gen.old_mental_model,
        new_mental_model=gen.new_mental_model,
        technical_concepts_to_teach=gen.technical_concepts_to_teach,
        stakes=gen.stakes,
        human_consequence=gen.human_consequence,
        economic_consequence=gen.economic_consequence,
        societal_consequence=gen.societal_consequence,
        human_angle=gen.human_angle,
        credible_disagreement=gen.credible_disagreement,
        story_engine=gen.story_engine,
        people_orgs=gen.people_orgs,
        evidence_available=gen.evidence_available,
        coverage_gap=gen.coverage_gap,
        visual_opportunities=gen.visual_opportunities,
        strongest_counterargument=gen.strongest_counterargument,
        what_we_need_to_investigate=gen.what_we_need_to_investigate,
        possible_ending_insight=gen.possible_ending_insight,
        source_ids=gen.source_ids or fallback_sources,
    )


def to_cluster(out: ClusterOutput, group: ConceptGroup) -> Cluster:
    return Cluster(
        cluster_name=out.cluster_name,
        underlying_change=out.underlying_change,
        raw_terms=out.raw_terms or group.labels,
        concept_ids=group.concept_ids,
        source_ids=sorted({s for c in out.evidence for s in c.source_ids}),
        evidence=out.evidence,
        disagreements=out.disagreements,
        potential_public_relevance=out.potential_public_relevance,
        breadth_warning=out.breadth_warning,
        momentum=group.momentum,
    )
