# Frontier Scout — Roadmap

Status as of 2026-09-20. Ordered by leverage, not by ease.

---

## Where the system stands

Shipped and running on live data:

- Four credential-free connectors (Substack, Hacker News, arXiv, lab blogs) —
  8,915 articles, zero source failures
- Concept registry with stable lexical identity and auto-nomination staging
- Independence graph (convergence vs. echo) discounting momentum arithmetically
- Trend engine, gate, eight parallel scouts, clusterer, translator, generator
- Independent evaluation, pairwise tournament, red team, human review
- Two LLM providers (Anthropic, DeepSeek) behind one abstraction
- Public-interest signal via SerpAPI; X activity derived from citations
- 91 tests

A full multi-agent run costs about $1.60 and takes roughly 20 minutes.

---

## The diagnosis that drives this roadmap

The corpus holds **8.26M characters (~2M tokens)** of article text. A scout sees
roughly **14k tokens per topic — about 0.7% of it** — selected by recency rather
than relevance.

Worse: the system stores *raw text* and *counts*, but never **what an article
argued**. Every run re-reads text, re-derives understanding, emits signals, and
discards the understanding. There is memory of counts and no memory of
comprehension.

A note on the medallion framing, since it prompted this work: the layering
already exists implicitly — `source_items` is bronze, `concept_mentions` /
`links` / `paper_refs` are silver, `concept_metrics` / `clusters` are gold.
Adopting the vocabulary would be relabeling. What is genuinely missing is not a
*cleaned* layer but a **semantic** one.

---

## P1 — The claim layer — SHIPPED (2026-09-20)

Built and backfilled. Measured on live data: long-form articles yield about 8
claims each against 4.8 for short forum comments, so extraction is selective by
source type and length rather than spending equally on an essay and a two-line
comment. The full corpus would be ~$11; the long-form pass ~$2.60.

Already surfacing what it was built for — named authors on both sides of real
splits (multi-agent orchestration: Anthropic and Nathan Lambert skeptical
against Ruiping Liu and Wenjie Liao advocating), a genuine public
self-correction, and one-sided concepts correctly reported as one-sided rather
than dressed up as balance.

Two false positives found and fixed during rollout: organisational bylines were
being read as one mind changing ("Anthropic moved from skeptical to advocating"
across two posts a day apart), and stance flips needed a minimum time gap.

**The one that unlocks the rest.**

Extract once per article at ingestion, persist permanently:

| Stored | Purpose |
|---|---|
| `claims` — proposition, stance, hedging, evidence type | what was actually asserted |
| `positions` — author's stance per tracked concept | how views move over time |
| `entities` — people, orgs, models, papers | who and what is involved |
| markers — "contrary to", "I was wrong", "for the first time" | surprise and self-correction |

Why this changes the system rather than merely improving it:

1. **Disagreement stops being a guess.** Today the Disagreement Scout skims ~40
   excerpts and forms a view. With a claim store it becomes a query: *find
   contradictory propositions about concept X from independent authors*, across
   **all** articles rather than the few that fit in a prompt.
2. **Self-correction becomes visible.** Same author, opposite stance, months
   apart. "Why [respected person] changed their mind about Y" is editorially
   valuable and is currently undetectable.
3. **The dossier gets roughly 20x denser** — distilled positions from hundreds
   of articles in the token budget that currently holds forty excerpts.
4. **Understanding accumulates** instead of being rebuilt nightly.

Cost: roughly $3-8 one-time for the existing backlog on a cheap model, then
pennies per day. It also makes later runs cheaper, because agents reason over
structure instead of prose.

## P2 — Author lead-time scoring

**Free. No new data, no model calls.** The database already holds 3,698 authors
and 8,025 dated mentions, which is enough to compute who reaches an idea before
the crowd does.

A proof run over current data:

```
author                concepts early on    avg days ahead
Jack Clark                      8                 31
Ethan Mollick                   7                238
Nathan Lambert                  6                 54
UC Berkeley BAIR                8                 68
```

38 authors have four or more early calls.

This makes authority **earned and observable** rather than asserted in config,
which was the original brief's requirement. Editorially, "three of our
highest-lead-time authors just started discussing X" is a far stronger signal
than raw volume.

Known calibration issue to fix during implementation: sources with deep
archives (OpenAI's blog shows 973 days ahead) score high because of corpus
depth, not prescience. Normalise by observation window.

## P3 — Ground truth

**The biggest hole in the system.** Nothing validates any of it. `video_outcomes`
is empty and the scoring weights are informed guesses.

Two moves:

- **Backtest** — freeze the corpus to 90 days ago, run the pipeline, and check
  whether it surfaces what actually became significant. The historical backfill
  makes this runnable today. Until it runs, "plausible" is the honest word for
  the system; afterwards you would know.
- **Human review as labels** — every APPROVE/REJECT plus excitement score is a
  labelled example. After ~50 reviews the momentum weights can be fitted to the
  editor's actual taste rather than the defaults in `config/editorial.yaml`.

Do not tune scoring weights further before this exists; there is nothing to tune
against.

## P4 — Two new sources (not "more data")

More raw volume into a system that discards comprehension only costs more. These
two are a different *kind* of signal:

- **Podcast transcripts** — where unguarded expert opinion lives. People say
  things in conversation they would never publish. Reachable via caption tracks.
- **Job postings** — a genuine leading indicator of corporate belief. A lab
  hiring six inference-optimisation engineers precedes anyone writing about it,
  and it feeds Capital Scout directly.

Later, in rough order: Reddit (free API, different community from HN), GitHub
release notes and star velocity, earnings-call transcripts, Bluesky curated
accounts (`getAuthorFeed` works unauthenticated; search does not).

## P5 — Retrieval instead of recency

The dossier currently selects source items by recency and engagement. It should
select by relevance to the specific scout's question, over the claim layer.
Blocked on P1.

## P6 — Concept lifecycle and state transitions

Each run is currently a snapshot that re-ranks everything. Concepts should carry
a lifecycle stage — emerging, accelerating, peaking, mainstream, fading — and
the brief should lead with **transitions**. "X moved from emerging to
accelerating" is more actionable than "X is ranked fourth today."

---

## Engineering debt worth clearing

- ~~**Durability of long runs.**~~ FIXED. Each artifact now persists the moment
  it exists: scouts save their own signals, a cluster is saved before
  generation is attempted, evaluations save per candidate, and the tournament
  commits each judgement as it lands. A `run_stages` table records where a run
  reached. Four tests pin it by failing a stage mid-flight and asserting the
  earlier work survived.
- ~~**Per-call timeouts.**~~ FIXED. A wall-clock ceiling per LLM call, because
  httpx read timeouts do not fire when a server trickles bytes.
- **Resumability.** A dead run can be re-run cheaply (the content-hash cache
  replays completed calls) but cannot yet skip finished stages outright.
- **Evaluator binaries are not discriminating** — 12 of 12 candidates passed
  both, in two consecutive runs. This is the same class of fault as the
  title-scope attack below: a check that always returns the same answer
  carries no information. Needs the same treatment — an explicit calibration
  scale with worked examples of a NO.
- ~~**Red team title-scope attack was a constant.**~~ FIXED. It returned FATAL
  on 10 of 10 candidates across two runs, with zero WOUNDED and zero
  SURVIVES, because it treated any gap between a general-audience title and a
  specific finding as fatal — an unfalsifiable standard, since every such
  title is broader than its evidence. It now separates an ASSERTION of an
  undemonstrated fact from a QUESTION the evidence can explore. Verified
  against controls: assertion-titles on thin evidence score FATAL 10/10,
  the same-evidence question scores WOUNDED 5/10.

  Worth remembering as a general lesson: the fix for "every candidate fails
  this check" was to the check, not the candidates.

## Explicitly not doing

- Renaming tables to bronze/silver/gold for its own sake.
- Adding sources before P1 lands.
- Further weight tuning before P3 exists.
