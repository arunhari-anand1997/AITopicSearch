"""Editorial opportunity scoring.

Momentum ranks what is MOVING. That is not the same as what is worth making, so
opportunity combines momentum with the qualitative dimensions the scouts and
evaluator supply, then penalises existing general-audience coverage.

The combination is a weighted GEOMETRIC mean, which implements the intended
multiplicative semantics: a near-zero on any single dimension collapses the
score. That is the correct behaviour -- a huge trend with no consequence is not
a video, and a big consequence with no mechanism is punditry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..models import ConceptMetrics, Evaluation

#: Floor for any dimension, so one missing judgement cannot zero out a
#: candidate that is strong everywhere else.
FLOOR = 0.05


@dataclass
class OpportunityScore:
    value: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    coverage_adjustment: float = 1.0
    red_team_penalty: float = 1.0
    coverage_known: bool = False
    notes: list[str] = field(default_factory=list)

    def explain(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.components.items()))
        return (
            f"{self.value:.3f} = geo_mean({parts}) "
            f"x coverage {self.coverage_adjustment:.2f} "
            f"x red_team {self.red_team_penalty:.2f}"
        )


def weighted_geometric_mean(values: dict[str, float], weights: dict[str, float]) -> float:
    """Geometric mean, renormalising over whatever dimensions are present."""
    usable = {k: max(FLOOR, min(1.0, v)) for k, v in values.items() if v is not None}
    if not usable:
        return 0.0
    total_weight = sum(weights.get(k, 0.0) for k in usable)
    if total_weight <= 0:
        total_weight = float(len(usable))
        weights = {k: 1.0 for k in usable}
    exponent_sum = sum(
        weights.get(k, 0.0) * math.log(v) for k, v in usable.items()
    )
    return math.exp(exponent_sum / total_weight)


def score_candidate(
    *,
    metrics: ConceptMetrics | None,
    evaluation: Evaluation | None,
    saturation: float | None,
    weights: dict[str, float],
    coverage_penalty: float = 0.7,
    red_team_penalty: float = 1.0,
) -> OpportunityScore:
    """Combine measured momentum with judged dimensions."""
    out = OpportunityScore(red_team_penalty=red_team_penalty)
    components: dict[str, float] = {}

    if metrics is not None:
        components["insider_momentum"] = metrics.insider_momentum

    if evaluation is not None and evaluation.scores:
        def norm(*keys: str) -> float | None:
            present = [evaluation.scores[k] for k in keys if k in evaluation.scores]
            return (sum(present) / len(present)) / 10.0 if present else None

        for name, keys in (
            ("consequence", ("stakes", "human_relevance")),
            ("public_confusion", ("public_curiosity", "novelty")),
            ("story_potential", ("story_motion", "visual_potential", "retellability")),
            ("technical_substance", ("technical_depth", "evidence_availability")),
        ):
            value = norm(*keys)
            if value is not None:
                components[name] = value
    else:
        out.notes.append("no evaluation available; opportunity is momentum-only")

    out.components = components
    base = weighted_geometric_mean(components, weights)

    # Coverage: only a KNOWN saturation adjusts the score. Absent YouTube data
    # we neither reward nor punish, and we say so, rather than guessing what
    # exists and silently killing a good topic.
    if saturation is None:
        out.coverage_known = False
        out.notes.append(
            "coverage unknown (YouTube not configured): no coverage adjustment applied"
        )
    else:
        out.coverage_known = True
        out.coverage_adjustment = max(0.1, 1.0 - coverage_penalty * max(0.0, min(1.0, saturation)))

    out.value = base * out.coverage_adjustment * red_team_penalty
    return out


def binary_gate(evaluation: Evaluation | None) -> tuple[bool, str]:
    """The two binary questions act as a hard filter, not a score.

    A candidate that only becomes interesting once you know the jargon has
    failed the channel's core premise, however high it scores elsewhere.
    """
    if evaluation is None:
        return True, ""
    if not evaluation.cares_before_knowing_term:
        return False, "fails: viewer would not care before learning the insider term"
    if not evaluation.interests_non_ai_person:
        return False, "fails: title would not interest someone outside AI"
    return True, ""
