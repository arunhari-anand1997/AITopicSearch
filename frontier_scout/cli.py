"""Frontier Scout command line.

    frontier-scout collect      gather real source material
    frontier-scout trends       compute momentum (no LLM, no cost)
    frontier-scout discover     multi-agent editorial discovery
    frontier-scout review       the human gate
    frontier-scout export       run artifacts
    frontier-scout daily        collect -> trends -> discover -> brief
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .config import Settings
from .db import Database
from .ingest.collector import collect_all
from .ingest.files import ingest_file
from .llm.base import LLMUnavailable
from .models import HumanReview
from .pipeline.brief import write_daily_brief, write_run_summary
from .pipeline.discovery import DiscoveryPipeline, write_artifacts
from .trends.engine import TrendEngine
from .trends.registry import ConceptRegistry

app = typer.Typer(
    add_completion=False,
    help="Live editorial intelligence for the technological frontier.",
)
console = Console()


def _setup() -> tuple[Settings, Database]:
    settings = Settings.load()
    return settings, Database(settings.db_path)


def _provider(settings: Settings):
    """Build the LLM provider, or return None with a clear explanation."""
    from .llm.anthropic_provider import AnthropicProvider

    try:
        return AnthropicProvider(settings)
    except LLMUnavailable as exc:
        console.print(f"[yellow]LLM unavailable:[/yellow] {exc}")
        return None


def _run_id(prefix: str = "run") -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}_%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------


@app.command()
def collect(
    backfill: bool = typer.Option(
        False, "--backfill", help="Also pull historical search results for baselines"
    ),
    only: Optional[str] = typer.Option(
        None, "--only", help="Comma-separated connector names"
    ),
) -> None:
    """Gather real source material from the configured sources."""
    settings, db = _setup()
    run_id = _run_id("collect")
    db.start_run(run_id, "collect", settings.config_hash())

    names = {n.strip() for n in only.split(",")} if only else None
    with console.status("[bold]collecting from live sources..."):
        summary = asyncio.run(
            collect_all(settings, db, run_id, include_backfill=backfill, only=names)
        )

    table = Table(title=f"Collection — {run_id}")
    table.add_column("connector")
    table.add_column("target")
    table.add_column("status")
    table.add_column("items", justify="right")
    for connector, target, status, count in summary.table():
        style = "green" if status in ("OK", "PARTIAL") else "red"
        table.add_row(connector, target[:40], f"[{style}]{status}[/{style}]", str(count))
    console.print(table)

    console.print(
        f"\n[bold]{summary.new_items}[/bold] new items "
        f"({summary.seen_items} seen, {len(summary.failures)} source failures)"
    )
    for failure in summary.failures:
        console.print(f"  [red]{failure.target}[/red]: {failure.error or failure.status}")

    registry = ConceptRegistry(db)
    registry.seed(settings.seed_concepts)
    mentions = registry.backfill_mentions()
    registry.update_first_observed()
    console.print(f"[dim]recorded {mentions} concept mentions[/dim]")

    nominations = registry.nominate_phrases()
    if nominations:
        console.print(
            f"\n[bold]{len(nominations)} phrases nominated[/bold] "
            "(status=nominated; promote in config/concepts.yaml to track them):"
        )
        for phrase, count, authors in nominations[:12]:
            console.print(f"  {phrase:40s} {count:3d} docs, {authors} authors")

    db.finish_run(run_id)


@app.command()
def trends(
    top: int = typer.Option(25, "--top", help="How many concepts to show"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON"),
    nominated: bool = typer.Option(
        False, "--nominated", help="Show auto-nominated phrases awaiting promotion"
    ),
) -> None:
    """Compute and show momentum. No LLM calls, no cost."""
    settings, db = _setup()
    registry = ConceptRegistry(db)
    registry.seed(settings.seed_concepts)
    registry.backfill_mentions()
    registry.update_first_observed()

    engine = TrendEngine(db, settings, registry)
    run_id = _run_id("trends")
    db.start_run(run_id, "trends", settings.config_hash())
    metrics = engine.compute_all(run_id=run_id)
    db.finish_run(run_id)

    # Auto-nominated phrases are a staging area, not the day's intelligence.
    # Mixing them in lets a thin n-gram headline the table.
    tracked = [m for m in metrics if m.status == "tracked"]
    staged = [m for m in metrics if m.status != "tracked"]
    shown = staged if nominated else tracked

    if json_out:
        console.print_json(json.dumps([m.model_dump() for m in shown[:top]], default=str))
        return

    if nominated:
        title = f"Auto-nominated phrases — {len(staged)} awaiting promotion"
    else:
        title = f"Insider momentum — {len(tracked)} tracked concepts"
    table = Table(title=title)
    table.add_column("concept", max_width=32, overflow="fold")
    table.add_column("mom", justify="right")
    table.add_column("raw", justify="right", style="dim")
    table.add_column("24h", justify="right")
    table.add_column("7d", justify="right")
    table.add_column("auth", justify="right")
    table.add_column("ind", justify="right")
    table.add_column("ratio", justify="right")
    table.add_column("plt", justify="right")
    table.add_column("vel", justify="right")

    for m in shown[:top]:
        ratio_style = "red" if m.independence_ratio < 0.5 else "white"
        velocity = (
            f"{m.velocity_ratio:.1f}x" if m.velocity_ratio is not None else "[dim]n/a[/dim]"
        )
        table.add_row(
            m.concept_label,
            f"{m.insider_momentum:.3f}",
            f"{m.insider_momentum_raw:.3f}",
            str(m.mentions_24h),
            str(m.mentions_7d),
            str(m.unique_authors_7d),
            str(m.independent_source_count),
            f"[{ratio_style}]{m.independence_ratio:.2f}[/{ratio_style}]",
            str(m.cross_platform_count),
            velocity,
        )
    console.print(table)
    console.print(
        "\n[dim]ratio < 0.50 (red) means the sources share an origin: an echo, "
        "not independent convergence. Momentum is already discounted for it.[/dim]"
    )
    if not nominated and staged:
        console.print(
            f"[dim]{len(staged)} auto-nominated phrases are excluded from this table "
            "and from editorial discovery until promoted in config/concepts.yaml. "
            "See them with --nominated.[/dim]"
        )
    flagged = [m for m in shown[:top] if m.notes]
    if flagged:
        console.print("\n[bold]Notes[/bold]")
        for m in flagged[:8]:
            for note in m.notes:
                console.print(f"  [dim]{m.concept_label}:[/dim] {note}")


@app.command()
def discover(
    top: Optional[int] = typer.Option(None, "--top", help="Concept groups through the gate"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore the LLM call cache"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Stop at the gate; show what would be sent"
    ),
) -> None:
    """Run multi-agent editorial discovery."""
    settings, db = _setup()
    provider = None if dry_run else _provider(settings)
    pipeline = DiscoveryPipeline(db, settings, provider, use_cache=not no_cache)

    if dry_run:
        metrics, groups = pipeline.compute_trends()
        console.print(f"[bold]{len(metrics)}[/bold] concepts scored; "
                      f"[bold]{len(groups)}[/bold] groups would pass the gate:\n")
        for group in groups[: (top or settings.gate_width)]:
            console.print(f"  [cyan]{group.title}[/cyan]  momentum={group.momentum:.3f} "
                          f"({len(group.item_ids)} source items)")
        console.print("\n[dim]No LLM calls made.[/dim]")
        return

    with console.status("[bold]running discovery (scouts run in parallel)..."):
        result = asyncio.run(pipeline.run(top_n=top))

    out_dir = write_artifacts(result, settings.runs_dir)
    brief = write_daily_brief(
        db, result, settings.runs_dir,
        signal_count=int(settings.pipeline.get("brief_signal_count", 12)),
    )
    write_run_summary(db, result, settings.runs_dir)

    for note in result.notes:
        console.print(f"[yellow]{note}[/yellow]")

    if result.finalists:
        console.print(f"\n[bold]{len(result.finalists)} finalists[/bold]\n")
        for idx, f in enumerate(result.finalists, start=1):
            score = f.opportunity.value if f.opportunity else 0.0
            console.print(f"[bold]{idx}. {f.candidate.working_title}[/bold]  "
                          f"[dim]({score:.3f})[/dim]")
            console.print(f"   {f.candidate.central_question}")

    if result.cost:
        console.print(
            f"\n[dim]cost: ${result.cost.get('estimated_cost_usd', 0):.4f} "
            f"over {result.cost.get('calls', 0)} calls "
            f"({result.cost.get('cache_hits', 0)} cached)[/dim]"
        )
    console.print(f"\nartifacts: [cyan]{out_dir}[/cyan]")
    console.print(f"brief:     [cyan]{brief}[/cyan]")
    console.print(f"\nreview with: [bold]frontier-scout review {result.run_id}[/bold]")


@app.command()
def review(run_id: Optional[str] = typer.Argument(None)) -> None:
    """The human gate. The editor decides; the system only surfaces."""
    settings, db = _setup()
    if run_id is None:
        row = db.conn.execute(
            "SELECT id FROM collection_runs WHERE mode='discover'"
            " ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            console.print("[red]no discovery runs found[/red]")
            raise typer.Exit(1)
        run_id = row["id"]

    rows = db.conn.execute(
        "SELECT c.*, e.mean_score, e.strongest_failure_reason, e.viewer_understanding_delta,"
        " e.cares_before_knowing_term, e.interests_non_ai_person"
        " FROM candidates c LEFT JOIN evaluations e ON e.candidate_id = c.id"
        " WHERE c.run_id = ? ORDER BY COALESCE(e.mean_score,0) DESC",
        (run_id,),
    ).fetchall()
    if not rows:
        console.print(f"[red]no candidates for run {run_id}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Reviewing {len(rows)} candidates from {run_id}[/bold]\n")

    for row in rows:
        _print_candidate(db, row)
        decision = Prompt.ask(
            "decision",
            choices=["approve", "reject", "watch", "research", "skip", "quit"],
            default="skip",
        )
        if decision == "quit":
            break
        if decision == "skip":
            console.print()
            continue

        excitement = IntPrompt.ask("personal excitement (1-10)", default=5)
        two_weeks = Confirm.ask("would you spend two weeks investigating this?", default=False)
        notes = Prompt.ask("notes", default="")
        favorite = Confirm.ask("favorite?", default=False)

        db.save_human_review(
            HumanReview(
                candidate_id=row["id"],
                decision={
                    "approve": "APPROVE", "reject": "REJECT",
                    "watch": "WATCH", "research": "RESEARCH_MORE",
                }[decision],
                favorite=favorite,
                personal_excitation_score=excitement,
                would_spend_two_weeks=two_weeks,
                notes=notes or None,
            ),
            run_id=run_id,
        )
        console.print("[green]recorded[/green]\n")

    counts = db.conn.execute(
        "SELECT decision, COUNT(*) n FROM human_reviews WHERE run_id=? GROUP BY decision",
        (run_id,),
    ).fetchall()
    if counts:
        console.print("\n[bold]Decisions[/bold]")
        for row in counts:
            console.print(f"  {row['decision']}: {row['n']}")


def _print_candidate(db: Database, row) -> None:
    red = db.conn.execute(
        "SELECT attack_key, verdict, severity, argument FROM red_team"
        " WHERE candidate_id=? AND verdict IN ('WOUNDED','FATAL')"
        " ORDER BY severity DESC LIMIT 4",
        (row["id"],),
    ).fetchall()
    wins = db.conn.execute(
        "SELECT COUNT(*) n FROM pairwise_results WHERE winner=?", (row["id"],)
    ).fetchone()["n"]
    played = db.conn.execute(
        "SELECT COUNT(*) n FROM pairwise_results WHERE candidate_a=? OR candidate_b=?",
        (row["id"], row["id"]),
    ).fetchone()["n"]
    sources = db.conn.execute(
        "SELECT COUNT(*) n FROM candidate_sources WHERE candidate_id=?", (row["id"],)
    ).fetchone()["n"]

    body = [
        f"[bold cyan]{row['working_title']}[/bold cyan]",
        "",
        f"[bold]CENTRAL QUESTION[/bold]\n{row['central_question']}",
        "",
        f"[bold]WHY NOW[/bold]\n{row['why_now']}",
        "",
        f"[bold]TECHNICAL MECHANISM[/bold]\n{row['technical_mechanism']}",
        "",
        f"[bold]OLD → NEW MODEL[/bold]\n{row['old_mental_model']}\n  →  {row['new_mental_model']}",
        "",
        f"[bold]HUMAN STAKE[/bold]\n{row['human_consequence']}",
        "",
        f"[bold]DISAGREEMENT[/bold]\n{row['credible_disagreement']}",
        "",
        f"[bold]COVERAGE GAP[/bold]\n{row['coverage_gap'] or 'unknown'}",
        "",
        f"[bold]PAIRWISE[/bold] {wins}/{played} wins   "
        f"[bold]EVALUATOR[/bold] {row['mean_score'] or 0:.1f}/10   "
        f"[bold]SOURCES[/bold] {sources}",
    ]
    if row["cares_before_knowing_term"] is not None:
        body += [
            "",
            f"[bold]BINARIES[/bold] cares-before-jargon: "
            f"{'YES' if row['cares_before_knowing_term'] else '[red]NO[/red]'}   "
            f"interests-non-AI: "
            f"{'YES' if row['interests_non_ai_person'] else '[red]NO[/red]'}",
        ]
    if row["strongest_failure_reason"]:
        body += ["", f"[bold]LIKELIEST FAILURE[/bold]\n{row['strongest_failure_reason']}"]
    if red:
        body += ["", "[bold]RED TEAM[/bold]"]
        for r in red:
            body.append(f"  [{r['attack_key']}] {r['verdict']} ({r['severity']}/10): {r['argument']}")

    console.print(Panel("\n".join(body), border_style="blue"))


@app.command()
def export(run_id: str) -> None:
    """Show where a run's artifacts live."""
    settings, _ = _setup()
    out = settings.runs_dir / run_id
    if not out.exists():
        console.print(f"[red]no artifacts for {run_id}[/red]")
        raise typer.Exit(1)
    for path in sorted(out.iterdir()):
        console.print(f"  {path}  [dim]({path.stat().st_size:,} bytes)[/dim]")


@app.command("ingest")
def ingest_cmd(path: str) -> None:
    """Ingest a JSON, JSONL, CSV or text file of manually gathered items."""
    settings, db = _setup()
    run_id = _run_id("ingest")
    db.start_run(run_id, "collect", settings.config_hash())
    result = ingest_file(path)
    if not result.ok:
        console.print(f"[red]{result.status}[/red]: {result.error}")
        raise typer.Exit(1)
    new = sum(db.insert_source_item(item, run_id=run_id) for item in result.items)
    db.record_fetch(result, run_id, new)
    db.finish_run(run_id)
    console.print(f"ingested [bold]{new}[/bold] new items of {len(result.items)}")


@app.command()
def daily(
    backfill: bool = typer.Option(False, "--backfill"),
    top: Optional[int] = typer.Option(None, "--top"),
) -> None:
    """COLLECT → NORMALIZE → DETECT → CLUSTER → SCOUT → GENERATE → EVALUATE → PRESENT."""
    settings, db = _setup()
    run_id = _run_id("daily")
    db.start_run(run_id, "collect", settings.config_hash())

    with console.status("[bold]1/3 collecting..."):
        summary = asyncio.run(
            collect_all(settings, db, run_id, include_backfill=backfill)
        )
    console.print(f"collected [bold]{summary.new_items}[/bold] new items "
                  f"({len(summary.failures)} source failures)")
    for failure in summary.failures:
        console.print(f"  [red]{failure.target}[/red]: {failure.status}")
    db.finish_run(run_id)

    provider = _provider(settings)
    pipeline = DiscoveryPipeline(db, settings, provider)
    with console.status("[bold]2/3 scoring and discovering..."):
        result = asyncio.run(pipeline.run(top_n=top))

    with console.status("[bold]3/3 writing brief..."):
        write_artifacts(result, settings.runs_dir)
        brief = write_daily_brief(
            db, result, settings.runs_dir,
            signal_count=int(settings.pipeline.get("brief_signal_count", 12)),
        )
        write_run_summary(db, result, settings.runs_dir)

    console.print(f"\nbrief: [cyan]{brief}[/cyan]")
    if result.cost:
        console.print(f"[dim]cost: ${result.cost.get('estimated_cost_usd', 0):.4f}[/dim]")


if __name__ == "__main__":
    app()
