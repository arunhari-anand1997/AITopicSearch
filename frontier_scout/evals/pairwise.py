"""Pairwise tournament.

Numeric LLM scores compress toward the middle, so the numeric ranking is not
trusted on its own. Candidates compete head to head on three separate
questions, each judged and stored separately, and the two rankings are kept
apart rather than blended into one number.

Every pair is judged twice with the order swapped, because position bias is
real and cheap to cancel.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..config import load_prompt, prompt_version
from ..llm.base import LLMRequest
from ..llm.runner import AgentRunner
from ..models import PAIRWISE_QUESTIONS, CandidateVideo, PairwiseResult
from .evaluator import render_candidate


class PairwiseOutput(BaseModel):
    winner: str = Field(description="Exactly 'A' or 'B'")
    confidence: float = Field(description="0.5 = coin flip, 0.95 = obvious")
    explanation: str


@dataclass
class TournamentResult:
    results: list[PairwiseResult] = field(default_factory=list)
    wins: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    played: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def ranking(self) -> list[tuple[str, float, int]]:
        """Win rate, best first. Ties broken by number of comparisons."""
        rows = [
            (cid, self.wins[cid] / self.played[cid], self.played[cid])
            for cid in self.played
            if self.played[cid]
        ]
        return sorted(rows, key=lambda r: (-r[1], -r[2]))

    def record_for(self, candidate_id: str) -> str:
        return f"{self.wins[candidate_id]:.1f}/{self.played[candidate_id]}"


class PairwiseTournament:
    role = "pairwise"
    prompt_name = "pairwise"

    def __init__(self, runner: AgentRunner):
        self.runner = runner

    async def run(
        self,
        candidates: list[CandidateVideo],
        *,
        swap_check: bool = True,
        sink: Callable[[PairwiseResult], None] | None = None,
    ) -> TournamentResult:
        """Judge every pair on every question.

        `sink` persists each judgement the moment it lands. A tournament over
        ten candidates is hundreds of calls, and saving only at the end meant a
        death partway through discarded every comparison already completed.
        """
        result = TournamentResult()
        if len(candidates) < 2:
            return result

        by_id = {c.id: c for c in candidates}
        jobs: list[Any] = []
        for a, b in itertools.combinations(candidates, 2):
            for question_key in PAIRWISE_QUESTIONS:
                jobs.append(self._judge(a, b, question_key, swapped=False, sink=sink))
                if swap_check:
                    jobs.append(self._judge(b, a, question_key, swapped=True, sink=sink))

        for judgement in await self.runner.gather(jobs):
            if judgement is None:
                continue
            result.results.append(judgement)
            result.played[judgement.candidate_a] += 1
            result.played[judgement.candidate_b] += 1
            if judgement.winner and judgement.winner in by_id:
                result.wins[judgement.winner] += 1
        return result

    async def _judge(
        self,
        a: CandidateVideo,
        b: CandidateVideo,
        question_key: str,
        *,
        swapped: bool,
        sink: Callable[[PairwiseResult], None] | None = None,
    ) -> PairwiseResult | None:
        question = PAIRWISE_QUESTIONS[question_key]
        task = (
            f"QUESTION: {question}\n\n"
            f"=== CANDIDATE A ===\n{render_candidate(a, include_counterargument=False)}\n\n"
            f"=== CANDIDATE B ===\n{render_candidate(b, include_counterargument=False)}\n\n"
            "Answer with winner 'A' or 'B'."
        )
        request = LLMRequest(
            agent="pairwise",
            instructions=load_prompt(self.prompt_name),
            task=task,
            schema=PairwiseOutput,
            prompt_version=prompt_version(self.prompt_name),
            effort=self.runner.provider.effort_for(self.role),
            # Generous despite the tiny schema: a verbose model that overruns
            # the limit returns finish_reason=length with NO usable JSON, and
            # a lost judgement is far more expensive than unused headroom.
            # On the first live tournament 24 of 270 judgements died this way
            # at max_tokens=2000.
            max_tokens=8000,
            metadata={"role": self.role, "question": question_key},
        )
        out: PairwiseOutput | None = await self.runner.run(request)
        if out is None:
            return None

        winner_id = a.id if out.winner.strip().upper() == "A" else b.id
        judgement = PairwiseResult(
            candidate_a=a.id,
            candidate_b=b.id,
            question_key=question_key,
            winner=winner_id,
            confidence=out.confidence,
            explanation=out.explanation,
            judge_model=self.runner.provider.model_for(self.role),
            order_swapped=swapped,
        )
        if sink is not None:
            sink(judgement)
        return judgement
