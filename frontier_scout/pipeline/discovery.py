"""The discovery pipeline.

Order matters and is the whole design: everything cheap and deterministic runs
first, a gate selects what deserves an LLM, and only then does the expensive
layer start. A run with no API key stops at the gate and still produces a
ranked momentum table.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..agents.dossier import ConceptGroup, build_groups
from ..agents.scouts import ALL_SCOUTS, CoverageScout, CoverageScoutOutput, to_signals
from ..agents.synthesis import (
    CandidateGenerator,
    Clusterer,
    ClusterOutput,
    Translator,
    to_candidate_video,
    to_cluster,
)
from ..config import Settings
from ..db import Database
from ..evals.evaluator import Evaluator
from ..evals.pairwise import PairwiseTournament, TournamentResult
from ..evals.red_team import RedTeam, severity_penalty
from ..llm.base import LLMProvider
from ..llm.runner import AgentRunner, BudgetExhausted
from ..models import ConnectorStatus
from ..models import (
    CandidateVideo,
    Cluster,
    ConceptMetrics,
    Evaluation,
    RedTeamFinding,
    Signal,
)
from ..trends.engine import TrendEngine
from ..trends.opportunity import OpportunityScore, binary_gate, score_candidate
from ..trends.registry import ConceptRegistry

log = logging.getLogger(__name__)


@dataclass
class Finalist:
    candidate: CandidateVideo
    evaluation: Evaluation | None = None
    opportunity: OpportunityScore | None = None
    red_team: list[RedTeamFinding] = field(default_factory=list)
    pairwise_record: str = ""
    pairwise_rate: float | None = None
    group_labels: list[str] = field(default_factory=list)
    gate_note: str = ""


@dataclass
class DiscoveryResult:
    run_id: str
    metrics: list[ConceptMetrics] = field(default_factory=list)
    groups: list[ConceptGroup] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    candidates: list[CandidateVideo] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    tournament: TournamentResult | None = None
    red_team: list[RedTeamFinding] = field(default_factory=list)
    finalists: list[Finalist] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    llm_available: bool = True


class DiscoveryPipeline:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        provider: LLMProvider | None = None,
        *,
        run_id: str | None = None,
        use_cache: bool = True,
    ):
        self.db = db
        self.settings = settings
        self.provider = provider
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        self.registry = ConceptRegistry(db)
        self.engine = TrendEngine(db, settings, self.registry)
        self.runner = (
            AgentRunner(provider, settings, db, self.run_id, use_cache=use_cache)
            if provider
            else None
        )

    # -- quantitative half (free) -----------------------------------------

    def compute_trends(self) -> tuple[list[ConceptMetrics], list[ConceptGroup]]:
        """Everything before the gate. No LLM, no cost."""
        self.registry.seed(self.settings.seed_concepts)
        self.registry.backfill_mentions()
        self.registry.update_first_observed()
        metrics = self.engine.compute_all(run_id=self.run_id)
        groups = build_groups(
            self.db,
            metrics,
            max_groups=self.settings.gate_width,
            min_independent_sources=int(
                self.settings.pipeline.get("min_independent_sources", 3)
            ),
        )
        for group in groups:
            group.propagation = [
                self.engine.propagation(cid) for cid in group.concept_ids[:3]
            ]
            for prop in group.propagation:
                self.db.save_propagation(prop)
        return metrics, groups

    # -- full run ----------------------------------------------------------

    async def run(
        self, *, top_n: int | None = None, coverage_blocks: dict[str, str] | None = None
    ) -> DiscoveryResult:
        result = DiscoveryResult(run_id=self.run_id)
        self.db.start_run(self.run_id, "discover", self.settings.config_hash())

        metrics, groups = self.compute_trends()
        result.metrics = metrics
        gate = top_n or self.settings.gate_width
        result.groups = groups[:gate]
        log.info("gate: %d concepts -> %d groups", len(metrics), len(result.groups))

        if self.runner is None:
            result.llm_available = False
            result.notes.append(
                "ANTHROPIC_API_KEY not configured: stopped after the quantitative "
                "layer. The momentum table above is complete and cost nothing."
            )
            self.db.finish_run(self.run_id, "partial", "no LLM provider")
            return result

        # Public interest first: the scouts should see the divergence.
        await self.fetch_public_interest(result.groups)

        if coverage_blocks is None:
            coverage_blocks = await self.fetch_coverage(result.groups)
            if not coverage_blocks:
                result.notes.append(
                    "YouTube coverage unavailable (no YOUTUBE_API_KEY): coverage-gap "
                    "scores are marked UNKNOWN rather than guessed, and no coverage "
                    "adjustment is applied to opportunity."
                )

        try:
            await self._agent_phase(result, coverage_blocks)
        except BudgetExhausted as exc:
            result.notes.append(f"BUDGET STOP: {exc}")
            log.warning("budget exhausted: %s", exc)

        result.cost = self.runner.cost_report()
        self.db.finish_run(self.run_id, "ok")
        return result

    async def _agent_phase(
        self, result: DiscoveryResult, coverage_blocks: dict[str, str]
    ) -> None:
        assert self.runner is not None

        # --- scouts, in parallel, blind to each other --------------------
        scout_jobs = []
        for group in result.groups:
            for scout_cls in ALL_SCOUTS:
                scout = scout_cls(self.db, self.runner)
                kwargs: dict[str, Any] = {}
                if scout_cls is CoverageScout:
                    kwargs["youtube_block"] = coverage_blocks.get(group.key, "")
                scout_jobs.append(self._run_scout(scout, group, kwargs))

        for group_key, agent_name, output in await self.runner.gather(scout_jobs):
            if output is None:
                continue
            signals = to_signals(agent_name, output, cluster_id=None)
            for sig in signals:
                sig.cluster_id = group_key
                self.db.save_signal(sig, self.run_id)
            result.signals.extend(signals)

        signals_by_group: dict[str, list[Signal]] = {}
        for sig in result.signals:
            signals_by_group.setdefault(sig.cluster_id or "", []).append(sig)

        coverage_by_group = self._coverage_saturation(result.groups, coverage_blocks)

        # --- cluster -> translate -> generate, per group -----------------
        cluster_jobs = [
            self._cluster_and_generate(group, signals_by_group.get(group.key, []))
            for group in result.groups
        ]
        for group, cluster, candidates in await self.runner.gather(cluster_jobs):
            if cluster is not None:
                result.clusters.append(cluster)
                self.db.save_cluster(cluster, self.run_id)
            for cand in candidates:
                self.db.save_candidate(cand, self.run_id)
            result.candidates.extend(candidates)

        if not result.candidates:
            result.notes.append("no candidates were generated")
            return

        # --- independent evaluation --------------------------------------
        evaluator = Evaluator(self.db, self.runner)
        group_by_candidate = self._group_index(result)
        eval_jobs = [
            evaluator.evaluate(
                cand, trend_context=self._trend_context(group_by_candidate.get(cand.id))
            )
            for cand in result.candidates
        ]
        evaluations = [e for e in await self.runner.gather(eval_jobs) if e]
        for ev in evaluations:
            self.db.save_evaluation(ev, self.run_id)
        result.evaluations = evaluations
        eval_by_id = {e.candidate_id: e for e in evaluations}

        # --- opportunity scoring + the binary gate ------------------------
        scored: list[tuple[CandidateVideo, OpportunityScore, str]] = []
        for cand in result.candidates:
            group = group_by_candidate.get(cand.id)
            ev = eval_by_id.get(cand.id)
            passed, note = binary_gate(ev)
            opportunity = score_candidate(
                metrics=max(group.metrics, key=lambda m: m.insider_momentum) if group else None,
                evaluation=ev,
                saturation=coverage_by_group.get(group.key) if group else None,
                weights=self.settings.opportunity_weights,
                coverage_penalty=float(
                    self.settings.section("scoring", {}).get("coverage_penalty", 0.7)
                ),
            )
            if not passed:
                # Kept, but demoted: the editor should see why it was dropped.
                opportunity.value *= 0.3
                opportunity.notes.append(note)
            scored.append((cand, opportunity, note))

        scored.sort(key=lambda row: row[1].value, reverse=True)

        # --- pairwise tournament on the top slice -------------------------
        top_n = int(self.settings.pipeline.get("pairwise_top_n", 10))
        contenders = [row[0] for row in scored[:top_n]]
        tournament = await PairwiseTournament(self.runner).run(contenders)
        for pr in tournament.results:
            self.db.save_pairwise(pr, self.run_id)
        result.tournament = tournament
        rates = {cid: rate for cid, rate, _ in tournament.ranking()}

        # --- red team on the finalists ------------------------------------
        red_n = int(self.settings.pipeline.get("red_team_top_n", 5))
        ranked_ids = [cid for cid, _, _ in tournament.ranking()] or [c.id for c in contenders]
        by_id = {c.id: c for c in result.candidates}
        red_targets = [by_id[cid] for cid in ranked_ids[:red_n] if cid in by_id]

        red_team = RedTeam(self.runner)
        attack_jobs = [
            red_team.attack(
                cand, trend_context=self._trend_context(group_by_candidate.get(cand.id))
            )
            for cand in red_targets
        ]
        findings_by_candidate: dict[str, list[RedTeamFinding]] = {}
        for findings, _summary in await self.runner.gather(attack_jobs):
            for f in findings:
                self.db.save_red_team(f, self.run_id)
                findings_by_candidate.setdefault(f.candidate_id, []).append(f)
            result.red_team.extend(findings)

        # --- one revision pass --------------------------------------------
        revision_jobs = [
            red_team.revise(cand, findings_by_candidate.get(cand.id, []))
            for cand in red_targets
            if findings_by_candidate.get(cand.id)
        ]
        for revised in await self.runner.gather(revision_jobs):
            if revised is not None:
                self.db.save_candidate(revised, self.run_id)
                result.candidates.append(revised)

        # --- assemble finalists -------------------------------------------
        opportunity_by_id = {c.id: o for c, o, _ in scored}
        for cand in red_targets:
            findings = findings_by_candidate.get(cand.id, [])
            opportunity = opportunity_by_id.get(cand.id)
            if opportunity:
                opportunity.red_team_penalty = severity_penalty(findings)
                opportunity.value *= opportunity.red_team_penalty
            group = group_by_candidate.get(cand.id)
            result.finalists.append(
                Finalist(
                    candidate=cand,
                    evaluation=eval_by_id.get(cand.id),
                    opportunity=opportunity,
                    red_team=findings,
                    pairwise_record=tournament.record_for(cand.id),
                    pairwise_rate=rates.get(cand.id),
                    group_labels=group.labels if group else [],
                )
            )
        result.finalists.sort(
            key=lambda f: f.opportunity.value if f.opportunity else 0.0, reverse=True
        )

    # -- helpers -----------------------------------------------------------

    async def _run_scout(self, scout: Any, group: ConceptGroup, kwargs: dict[str, Any]):
        output = await scout.run(group, **kwargs)
        return group.key, scout.name, output

    async def _cluster_and_generate(
        self, group: ConceptGroup, signals: list[Signal]
    ) -> tuple[ConceptGroup, Cluster | None, list[CandidateVideo]]:
        assert self.runner is not None
        cluster_out: ClusterOutput | None = await Clusterer(self.db, self.runner).run(
            group, signals=signals
        )
        cluster = to_cluster(cluster_out, group) if cluster_out else None

        translation = await Translator(self.db, self.runner).run(group, cluster=cluster_out)
        generated = await CandidateGenerator(self.db, self.runner).run(
            group,
            cluster=cluster_out,
            translation=translation,
            signals=signals,
            count=int(self.settings.pipeline.get("candidates_per_cluster", 3)),
        )
        fallback = sorted({s for sig in signals for s in sig.source_ids})[:12]
        candidates = [
            to_candidate_video(g, cluster.id if cluster else None, fallback)
            for g in (generated.candidates if generated else [])
        ]
        return group, cluster, candidates

    def _group_index(self, result: DiscoveryResult) -> dict[str, ConceptGroup]:
        """Map candidate id -> the group it came from, via its cluster."""
        cluster_to_group: dict[str, ConceptGroup] = {}
        for cluster in result.clusters:
            for group in result.groups:
                if set(cluster.concept_ids) & set(group.concept_ids):
                    cluster_to_group[cluster.id] = group
                    break
        out: dict[str, ConceptGroup] = {}
        for cand in result.candidates:
            group = cluster_to_group.get(cand.cluster_id or "")
            if group:
                out[cand.id] = group
        return out

    def _trend_context(self, group: ConceptGroup | None) -> str:
        if not group:
            return ""
        lines = []
        for m in group.metrics[:4]:
            velocity = (
                f"{m.velocity_ratio:.2f}x baseline"
                if m.velocity_ratio is not None
                else "velocity NOT COMPUTED (insufficient history)"
            )
            lines.append(
                f"- {m.concept_label}: {m.mentions_7d} mentions/7d, "
                f"{m.unique_authors_7d} authors, {m.independent_source_count} "
                f"INDEPENDENT sources (ratio {m.independence_ratio:.2f}), {velocity}, "
                f"momentum {m.insider_momentum:.3f}"
            )
            for note in m.notes:
                lines.append(f"  NOTE: {note}")
        return "\n".join(lines)

    def _coverage_saturation(
        self, groups: list[ConceptGroup], coverage_blocks: dict[str, str]
    ) -> dict[str, float | None]:
        """Saturation stays None unless YouTube data actually existed.

        None means "we do not know", which the scorer treats as no adjustment.
        Guessing here would silently kill good topics or greenlight covered
        ones, and either failure is invisible after the fact.
        """
        return {g.key: None for g in groups if not coverage_blocks.get(g.key)}

    async def fetch_public_interest(
        self, groups: list[ConceptGroup]
    ) -> dict[int, str]:
        """Public search interest per gated concept.

        Runs after the gate because SerpAPI bills per search against a monthly
        quota. Feeds `google_search_change` and the public-confusion dimension
        only -- never INSIDER_MOMENTUM. The whole point is that insider
        attention and public interest are different signals: a concept
        insiders are converging on while public search stays flat or falls is
        the editorial opportunity, and merging them would erase it.
        """
        from ..ingest.serpapi_trends import SerpApiTrendsConnector

        readings: dict[int, str] = {}
        async with SerpApiTrendsConnector(self.settings) as trends:
            if not trends.enabled:
                return readings
            for group in groups:
                for metrics in group.metrics[:1]:  # one query per group
                    result = await trends.interest(metrics.concept_label)
                    summary = result.summary()
                    readings[metrics.concept_id] = summary
                    metrics.google_search_change = summary
                    if result.status == ConnectorStatus.OK and result.direction:
                        # The divergence IS the finding, so state it plainly
                        # where the scouts and the editor will both see it.
                        if result.direction in ("falling", "flat") and metrics.insider_momentum > 0.15:
                            metrics.notes.append(
                                f"DIVERGENCE: insider momentum {metrics.insider_momentum:.2f} "
                                f"while public search interest is {result.direction} "
                                f"({result.change_7d_vs_90d:+.0%} vs its 90-day average). "
                                "Insiders are moving on this ahead of the public."
                            )
        return readings

    async def fetch_coverage(self, groups: list[ConceptGroup]) -> dict[str, str]:
        """Pull existing YouTube coverage for gated groups only.

        Runs AFTER the momentum gate because search.list is capped at 100 calls
        per day as its own budget -- it is the scarcest quota in the system.
        """
        from ..ingest.youtube import YouTubeConnector, render_coverage_block

        if not self.settings.youtube_api_key:
            return {}
        if not self.settings.section("youtube", {}).get("enabled", True):
            return {}

        blocks: dict[str, str] = {}
        async with YouTubeConnector(self.settings) as yt:
            for group in groups:
                query = group.labels[0] if group.labels else ""
                if not query:
                    continue
                result = await yt.coverage_for(query)
                if result.ok and result.items:
                    blocks[group.key] = render_coverage_block(result.items)
                    for item in result.items:
                        self.db.insert_source_item(item, run_id=self.run_id)
                elif result.status == ConnectorStatus.BUDGET_EXHAUSTED:
                    log.warning("youtube search budget spent; remaining groups unchecked")
                    break
        return blocks


def write_artifacts(result: DiscoveryResult, runs_dir: Path) -> Path:
    """Persist every intermediate artifact as inspectable JSON/JSONL."""
    out = runs_dir / result.run_id
    out.mkdir(parents=True, exist_ok=True)

    def dump_jsonl(name: str, rows: list[Any]) -> None:
        with (out / name).open("w") as fh:
            for row in rows:
                payload = row.model_dump() if hasattr(row, "model_dump") else row
                fh.write(json.dumps(payload, default=str) + "\n")

    def dump_json(name: str, obj: Any) -> None:
        (out / name).write_text(json.dumps(obj, default=str, indent=2))

    dump_jsonl("01_metrics.jsonl", result.metrics)
    dump_jsonl("02_signals.jsonl", result.signals)
    dump_json("03_clusters.json", [c.model_dump() for c in result.clusters])
    dump_json("04_candidates.json", [c.model_dump() for c in result.candidates])
    dump_json("05_evaluations.json", [e.model_dump() for e in result.evaluations])
    dump_json(
        "06_pairwise.json",
        [r.model_dump() for r in (result.tournament.results if result.tournament else [])],
    )
    dump_json("07_red_team.json", [f.model_dump() for f in result.red_team])
    dump_json(
        "08_finalists.json",
        [
            {
                "candidate": f.candidate.model_dump(),
                "evaluation": f.evaluation.model_dump() if f.evaluation else None,
                "opportunity": f.opportunity.value if f.opportunity else None,
                "opportunity_explain": f.opportunity.explain() if f.opportunity else None,
                "pairwise_record": f.pairwise_record,
                "red_team": [r.model_dump() for r in f.red_team],
            }
            for f in result.finalists
        ],
    )
    dump_json("cost.json", result.cost)
    return out
