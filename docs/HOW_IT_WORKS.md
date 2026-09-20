# How Frontier Scout works

Written for someone with no prior context. No code required.

---

## The job

The channel makes documentaries about technology for intelligent people who do
**not** work in tech. Not tutorials, not AI news — the kind of video that leaves
a viewer able to form their own opinion about a shift.

The hard part is not making the video. It is **knowing what to make a video
about**, early — while an idea is still moving among insiders and before it is
obvious to everyone else.

That is this system's entire job: find those topics, with evidence, before they
are obvious.

## What it refuses to do

The naive version is "ask an AI for video ideas". That fails for a specific
reason: a model answers from what it absorbed during training, which is the
past. It has no idea what happened this month, and it will confidently hand you
last year's topics.

The second-naive version is "find trending words". That fails too, and we have
proof. `loop engineering` — one of the terms in the original brief — measured
**78 mentions in 90 days and 0 in the last 7**. It is dead. A keyword counter
would have proposed a video about it.

So the governing rule is:

```
TRENDING TERM  ≠  GOOD VIDEO
```

Instead the system traces a chain:

```
signals → underlying change → technical mechanism
        → human consequence → disagreement → public question
```

Five buzzwords do not become five videos. They become **one underlying change**
and **one public question**.

---

## The pipeline, stage by stage

### 1. Read the real world

Every run reads what knowledgeable people actually published:

- 11 Substack newsletters (full article text, free, via RSS)
- Hacker News (live front page plus full searchable history)
- arXiv (research papers)
- 8 lab and company blogs (OpenAI, DeepMind, Anthropic, NVIDIA, …)

A representative run: **8,915 articles, zero source failures.**

Two deliberate omissions. **Meta AI** is not collected at all — its robots.txt
prohibits automated collection. Two newsletters sat behind bot-protection
challenges; rather than working around them, the system names the problem and
uses the authors' own domains, which serve the same feed openly.

### 2. Count things, carefully

The unglamorous stage everything else depends on.

The system tracks a fixed registry of **concepts** — "world models", "agent
harness", "verification bottleneck" — each with alias spellings, so "harness
engineering" and "agent scaffolding" count as the same idea.

Why fix the list rather than let a model extract concepts freely? Because a
model would say "context engineering" on Monday and "prompt scaffolding" on
Tuesday, and the 30-day trend line would silently shatter into disconnected
fragments. Every number built on top would be fiction. **Counting is never done
by a model.**

The system does *propose* new phrases it sees spreading across independent
authors — recent examples include "millennium prize problem", "regulatory
capture" and "pace the frontier" — but those land in a staging list for the
editor to promote. The system does not get to decide on its own what is worth
spending money on.

### 3. Independence: five people agreeing, or one person echoed?

This is the stage that separates the system from a dashboard.

Counting authors cannot answer that question, so independence is computed as a
graph. Two articles are joined if they share an author, cite each other, link
the same source, reference the same paper, or use near-identical wording. The
system then counts the separate islands.

```
5 posts, 1 author                  → 1 independent source
4 writers all citing one essay     → 1 independent source   ← the trap
4 writers, no shared origin        → 4 independent sources  ← the buy signal
```

Not theoretical. On live data, **`recursive self-improvement` had the fastest
growth of anything measured** — 120 mentions a week, 3.2× its baseline. Any
trending tool would rank it first. Its independence ratio was **0.36**: 103
authors collapsing into 69 real sources, largely echoing one another. The
system demoted it to 9th automatically, by arithmetic rather than argument.

### 4. The gate

Everything above is free: no model calls, no cost. The system now selects the
top ~12 topics, and **only those** proceed to the expensive stage.

That is the entire reason the counting layer exists — to decide what deserves a
model's attention. A run with no LLM budget at all still produces a useful
ranked table.

### 5. Eight specialists, working blind

Surviving topics go to eight agents, each asked **one** narrow question and each
given real articles with real URLs. None sees the others' output, so where they
disagree, that disagreement is information rather than something averaged away.

| Agent | Its one question |
|---|---|
| Trend | What is genuinely accelerating — a new *idea* or just a new *word*? |
| Frontier | What actually changed technically? |
| Disagreement | Where do credible people read the same evidence differently? |
| Consequence | If this is real, what changes for someone outside the field? |
| Mechanism | What must the audience understand for this to land? |
| Capital | What are serious organisations spending on? |
| Story | What makes this a story rather than a lecture? |
| Coverage | Has someone already explained this well to a general audience? |

Real output from the Consequence scout:

> *"Reliability becomes a purchasable tier, and the buyer of the tier is usually
> the counterparty, not you."*

### 6. Provenance: the unbreakable rule

Every statement an agent makes is tagged either **SOURCE_FACT** (something an
article actually said, with that article's ID attached) or **INFERENCE** (the
model's own reasoning, citing nothing).

This is enforced mechanically, not by asking politely. If an agent cites a
source ID it was never shown, that is detected automatically, the agent gets one
chance to repair it, and otherwise the claim is stripped and demoted to
inference. A representative run: **281 citations, 0 fabricated.**

It matters because the alternative — a model's guess sitting beside a sourced
fact, indistinguishable — is how you confidently publish something untrue.

### 7. Translate insider language into a public question

```
"loop engineering"      → "What Happens When AI Stops Waiting for Instructions?"
"inference cost"        → "What Happens When AI Becomes Almost Free?"
"multi-agent"           → "What Happens When One AI Becomes a Team?"
```

A generator then writes full video concepts: title, central question, the
mechanism to teach, what the viewer believes before and after, the human stake,
the real disagreement, and the strongest argument for **not** making it.

Hard rule: a concept needs **both something worth learning and something at
stake**. Mechanism with no consequence is a lecture. Consequence with no
mechanism is punditry.

### 8. Judgement, deliberately adversarial

Three separate checks, because a model asked to rate its own idea always likes
it:

1. **An independent evaluator** scores 15 dimensions, never knowing which
   generator produced the candidate.
2. **A tournament**: every candidate against every other, three questions each
   ("which would you publish next week?"), judged twice with the order flipped
   so position bias cancels.
3. **A red team** told plainly: *assume this will fail, explain why.*

The red team is the sharpest part. On one run it killed all five finalists on
the same charge — the title claimed more than the evidence supported:

> Evidence: one benchmark on file-review agents.
> Title: *"When AI Claims a Job Is Done, How Do We Know It Did the Work?"* —
> **too broad.**

It also supplies a checkable exit test: *"If no independent economic analysis is
found, drop it."* That is the system catching its own generator's systematic
flaw.

### 9. The human decides

The system produces a brief and a review screen. The editor approves, rejects,
watches, or sends for more research, and records how much they would personally
want to spend two weeks on it.

**The system never picks.** It is a research assistant that surfaces unusually
strong ideas with receipts.

---

## The finding that best explains the point

Insider attention and public attention are measured **separately** and never
mixed. Here is why that matters:

| concept | insiders | the public |
|---|---|---|
| world models | 49 mentions/week, 46 authors, accelerating | **down 76%** |
| test-time compute | rising | **down 51%** |
| context engineering | rising | **down 53%** |

Every topic that passed the gate showed insiders converging while public search
interest fell. That gap — people who know, moving on something the public has
not noticed — *is* the editorial opportunity. Merge those two numbers into one
"trend score" and you erase the only thing you were looking for.

## What it costs

| | |
|---|---|
| Reading the world plus all the counting | **$0** |
| A full multi-agent analysis run | **~$1.60** |
| Wall-clock time | ~20 minutes |

## Where to look next

- `README.md` — install, commands, how to add sources
- `docs/ARCHITECTURE.md` — the full design, verified API findings, and the
  measurement traps the system is built to avoid
