"""Red team, plus the single revision pass it earns.

The most valuable field here is `abandon_evidence`: what we would have to
observe to conclude the topic should be dropped. It converts a vague worry into
a checkable test, which is the difference between criticism and useful
criticism.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..config import load_prompt, prompt_version
from ..llm.base import LLMRequest
from ..llm.runner import AgentRunner
from ..models import RED_TEAM_ATTACKS, CandidateVideo, RedTeamFinding
from .evaluator import render_candidate


class Attack(BaseModel):
    attack_key: str
    verdict: str = Field(description="SURVIVES, WOUNDED or FATAL")
    argument: str
    severity: int
    abandon_evidence: str | None = None


class RedTeamOutput(BaseModel):
    attacks: list[Attack] = Field(default_factory=list)
    overall_verdict: str
    single_biggest_risk: str


class RevisionOutput(BaseModel):
    should_revise: bool
    revised_working_title: str | None = None
    revised_central_question: str | None = None
    revised_premise: str | None = None
    what_changed: str = ""
    why_unchanged: str = ""


class RedTeam:
    role = "red_team"
    prompt_name = "red_team"

    def __init__(self, runner: AgentRunner):
        self.runner = runner

    async def attack(
        self, cand: CandidateVideo, *, trend_context: str = ""
    ) -> tuple[list[RedTeamFinding], RedTeamOutput | None]:
        task = (
            f"{render_candidate(cand)}\n\n"
            + (f"MEASURED TREND DATA:\n{trend_context}\n\n" if trend_context else "")
            + "Attack this candidate on every angle: "
            + ", ".join(RED_TEAM_ATTACKS)
            + ".\nUse the measured data where it is relevant, especially for "
            "trend_already_fading and single_origin_cascade."
        )
        request = LLMRequest(
            agent="red_team",
            instructions=load_prompt(self.prompt_name),
            task=task,
            schema=RedTeamOutput,
            prompt_version=prompt_version(self.prompt_name),
            effort=self.runner.provider.effort_for(self.role),
            metadata={"role": self.role, "candidate": cand.id},
        )
        out: RedTeamOutput | None = await self.runner.run(request)
        if out is None:
            return [], None

        findings = [
            RedTeamFinding(
                candidate_id=cand.id,
                attack_key=a.attack_key,
                verdict=a.verdict if a.verdict in ("SURVIVES", "WOUNDED", "FATAL") else "WOUNDED",
                argument=a.argument,
                severity=a.severity,
                abandon_evidence=a.abandon_evidence,
            )
            for a in out.attacks
        ]
        return findings, out

    async def revise(
        self, cand: CandidateVideo, findings: list[RedTeamFinding]
    ) -> CandidateVideo | None:
        """One revision pass, in response to the attacks.

        Only the framing is revised -- title, question, premise. The evidence
        does not change because criticism arrived, and rewriting the mechanism
        or the disagreement in response to an attack would be exactly the kind
        of motivated reasoning this stage exists to catch.
        """
        serious = [f for f in findings if f.verdict in ("WOUNDED", "FATAL")]
        if not serious:
            return None

        attack_text = "\n".join(
            f"- [{f.attack_key}] {f.verdict} (severity {f.severity}): {f.argument}"
            for f in serious
        )
        task = (
            f"{render_candidate(cand)}\n\n"
            f"=== RED TEAM FINDINGS ===\n{attack_text}\n\n"
            "You may revise ONLY the title, central question and premise, to make "
            "the promise match the evidence. Do NOT invent new evidence, and do "
            "NOT soften the topic into vagueness to dodge criticism.\n\n"
            "CONSTRAINT THAT SURVIVES REVISION: the narrowed title must still be "
            "a jargon-free public question an intelligent non-specialist would "
            "want answered. Narrowing the SCOPE is required; narrowing into "
            "insider vocabulary is a different failure and is not acceptable. "
            "'Is test-time compute already hitting diminishing returns?' is NOT "
            "an improvement on a too-broad title -- it just fails a different "
            "test. Neither is a news headline about one company's incident. "
            "Aim for a question that is both honest about scope AND readable by "
            "someone who has never heard the insider term.\n\n"
            "If the findings are fatal and no honest reframing survives both "
            "constraints, set should_revise=false and explain why in "
            "why_unchanged."
        )
        request = LLMRequest(
            agent="red_team_revision",
            instructions=load_prompt(self.prompt_name),
            task=task,
            schema=RevisionOutput,
            prompt_version=prompt_version(self.prompt_name),
            effort=self.runner.provider.effort_for(self.role),
            metadata={"role": self.role, "candidate": cand.id},
        )
        out: RevisionOutput | None = await self.runner.run(request)
        if out is None or not out.should_revise:
            return None

        revised = cand.model_copy(
            update={
                "id": "",  # re-derived, so the revision is a distinct row
                "working_title": out.revised_working_title or cand.working_title,
                "central_question": out.revised_central_question or cand.central_question,
                "one_sentence_premise": out.revised_premise or cand.one_sentence_premise,
                "revision_of": cand.id,
            }
        )
        return CandidateVideo(**revised.model_dump(exclude={"id"}))


def severity_penalty(findings: list[RedTeamFinding]) -> float:
    """Convert red-team findings into a multiplier on opportunity.

    A FATAL finding is decisive; wounds accumulate. Kept explicit and small so
    the editor can see exactly what the red team cost a candidate.
    """
    if any(f.verdict == "FATAL" for f in findings):
        return 0.25
    wounded = [f for f in findings if f.verdict == "WOUNDED"]
    if not wounded:
        return 1.0
    total = sum(f.severity for f in wounded)
    return max(0.4, 1.0 - 0.04 * total)
