"""Independent evaluation.

Generators never score their own work. The evaluator is a separate call with a
separate prompt and no knowledge of which generator produced a candidate.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import load_prompt, prompt_version
from ..db import Database
from ..llm.base import LLMRequest
from ..llm.runner import AgentRunner
from ..models import SCORE_DIMENSIONS, CandidateVideo, Evaluation


class EvaluatorOutput(BaseModel):
    buzz_timeliness: int
    public_curiosity: int
    stakes: int
    novelty: int
    technical_depth: int
    accessibility: int
    human_relevance: int
    serious_disagreement: int
    story_motion: int
    visual_potential: int
    evidence_availability: int
    coverage_gap: int
    retellability: int
    sponsor_audience_quality: int
    expansion_potential: int

    cares_before_knowing_term: bool
    interests_non_ai_person: bool
    strongest_failure_reason: str
    viewer_understanding_delta: str
    old_model: str
    new_model: str


def render_candidate(cand: CandidateVideo, *, include_counterargument: bool = True) -> str:
    """Render a candidate for judging.

    Deliberately omits which agent produced it and any prior scores, so the
    evaluator cannot anchor on someone else's opinion.
    """
    lines = [
        f"TITLE: {cand.working_title}",
        f"CENTRAL QUESTION: {cand.central_question}",
        f"PREMISE: {cand.one_sentence_premise}",
        f"WHY NOW: {cand.why_now}",
        f"CURRENT TRIGGER: {cand.real_current_trigger}",
        f"TECHNICAL MECHANISM: {cand.technical_mechanism}",
        f"OLD MENTAL MODEL: {cand.old_mental_model}",
        f"NEW MENTAL MODEL: {cand.new_mental_model}",
        f"HUMAN CONSEQUENCE: {cand.human_consequence}",
        f"STAKES: {cand.stakes or 'not stated'}",
        f"DISAGREEMENT: {cand.credible_disagreement}",
        f"STORY ENGINE: {cand.story_engine or 'not stated'}",
        f"COVERAGE GAP: {cand.coverage_gap or 'unknown'}",
        f"EVIDENCE AVAILABLE: {cand.evidence_available or 'unknown'}",
        f"VISUALS: {', '.join(cand.visual_opportunities) or 'none listed'}",
        f"SOURCE COUNT: {len(cand.source_ids)}",
    ]
    if include_counterargument and cand.strongest_counterargument:
        lines.append(f"STATED COUNTERARGUMENT: {cand.strongest_counterargument}")
    return "\n".join(lines)


class Evaluator:
    role = "evaluator"
    prompt_name = "evaluator"

    def __init__(self, db: Database, runner: AgentRunner):
        self.db = db
        self.runner = runner

    async def evaluate(
        self, cand: CandidateVideo, *, trend_context: str = ""
    ) -> Evaluation | None:
        task = render_candidate(cand)
        if trend_context:
            task = f"{task}\n\nMEASURED TREND DATA:\n{trend_context}"

        request = LLMRequest(
            agent="evaluator",
            instructions=load_prompt(self.prompt_name),
            task=task,
            schema=EvaluatorOutput,
            prompt_version=prompt_version(self.prompt_name),
            effort=self.runner.provider.effort_for(self.role),
            metadata={"role": self.role, "candidate": cand.id},
        )
        out: EvaluatorOutput | None = await self.runner.run(request)
        if out is None:
            return None

        payload = out.model_dump()
        return Evaluation(
            candidate_id=cand.id,
            evaluator_model=self.runner.provider.model_for(self.role),
            prompt_version=request.prompt_version,
            scores={k: payload[k] for k in SCORE_DIMENSIONS},
            cares_before_knowing_term=out.cares_before_knowing_term,
            interests_non_ai_person=out.interests_non_ai_person,
            strongest_failure_reason=out.strongest_failure_reason,
            viewer_understanding_delta=out.viewer_understanding_delta,
            old_model=out.old_model,
            new_model=out.new_model,
        )
