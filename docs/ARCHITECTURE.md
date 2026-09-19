# Frontier Scout — Proposed Architecture (v0.1, for approval)

**Status: PROPOSAL. No implementation code written yet, pending editorial approval.**

Date: 2026-09-19. All API-access claims below were verified against live
documentation and live endpoint probes on this date, not recalled from memory.
Verification notes are marked `[VERIFIED 2026-09-19]`.

---

## 0. The one-line summary

Frontier Scout is a **measurement system with an LLM interpretation layer bolted
on top** — not an agent swarm that brainstorms. Anything that can be counted is
counted deterministically in SQL. LLMs are only allowed to *interpret* numbers
and *cite* source rows they were actually shown. Every editorial claim traces
back to a `source_item.id` with a URL and a timestamp, or it is labelled
`INFERENCE` and stored in a different column.

---

## 1. Architecture restatement

The system is a daily, resumable pipeline with a hard cost gate in the middle.

```
  ┌─ CHEAP / DETERMINISTIC / NO LLM ────────────────────────────────┐
  │  1. COLLECT      connectors → raw payloads (idempotent, cached) │
  │  2. NORMALIZE    → source_items + authors + publications        │
  │                    + engagement_snapshots + links + paper_refs  │
  │  3. RESOLVE      concept registry matching (stable IDs)         │
  │                    + novel-phrase nomination                    │
  │  4. INDEPENDENCE attribution graph → independent_source_count   │
  │  5. TREND MATH   daily stats → velocity / acceleration /        │
  │                    novelty / cross-platform → INSIDER_MOMENTUM  │
  │  6. PRE-CLUSTER  co-occurrence graph → candidate concept groups │
  └─────────────────────────────────────────────────────────────────┘
                              ↓
            ╔═══ GATE: top N groups only (default N=12) ═══╗
            ║  Everything above is free. Everything below  ║
            ║  costs money. The gate is where the budget   ║
            ║  is actually enforced.                       ║
            ╚══════════════════════════════════════════════╝
                              ↓
  ┌─ EXPENSIVE / LLM / PARALLEL ────────────────────────────────────┐
  │  7. COVERAGE     YouTube search (scarce quota, see §3)          │
  │  8. SCOUTS       8 scouts run concurrently per group, blind to  │
  │                    each other, must cite source_ids             │
  │  9. CLUSTERER    consolidates groups → underlying_change        │
  │ 10. TRANSLATOR   insider concept → public question              │
  │ 11. GENERATOR    2–5 candidates per cluster (higher temp)       │
  │ 12. EVALUATOR    independent call, 15 dimensions (low temp)     │
  │ 13. PAIRWISE     top 8–12, 3 questions per pair                 │
  │ 14. RED TEAM     top 5 attacked, one revision pass              │
  └─────────────────────────────────────────────────────────────────┘
                              ↓
  15. BRIEF + HUMAN GATE (approve / reject / watch / research-more)
```

### Three architectural commitments

**(a) The gate is the design.** Scouts are the expensive part. Running them on
everything is how this becomes a $200/day toy. The quantitative layer exists to
decide *what deserves an LLM*, and nothing else. A run with zero LLM budget
still produces a ranked momentum table — which is genuinely useful on its own.

**(b) Concept identity must be stable across days, or the time series is
fiction.** This is the failure mode that silently kills trend systems: you ask an
LLM to extract concepts today and it says "context engineering", tomorrow it says
"prompt scaffolding", and your 30-day series is three disconnected fragments. So
concept identity is **lexical and deterministic** (a registry of canonical labels
+ alias sets, matched by normalized phrase). The LLM's job is to *propose* new
canonical concepts and *merge aliases into the registry* — a periodic
registry-maintenance job, not a per-item extraction job. Counting is never done
by a model.

**(c) Independence is computed, not vibed.** See §6. "Five writers are saying
this" is worthless if four of them are quoting the fifth. This is the system's
main epistemic contribution and it is a first-class module, not a prompt
instruction.

---

## 2. Ingestion architecture, source by source

Every connector implements one interface and returns a `CollectionResult`
carrying an explicit status. **No connector is ever allowed to return
fabricated or substituted data.** A failed source is recorded as failed, the
run continues, and every downstream score carries a `data_completeness` value
naming which sources were missing.

```python
class ConnectorStatus(StrEnum):
    OK = "OK"
    PARTIAL = "PARTIAL"                  # some targets failed
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"  # network/5xx/feed gone
    ACCESS_DENIED = "ACCESS_DENIED"      # 401/403 — tier doesn't permit
    RATE_LIMITED = "RATE_LIMITED"
    NOT_CONFIGURED = "NOT_CONFIGURED"    # no credential present
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
```

### 2.1 Substack — `source_role: EARLY_SIGNAL | INDEPENDENT_ANALYSIS`

Public RSS only: `https://<pub>.substack.com/feed`, driven by
`config/substack_feeds.yaml`.

`[VERIFIED 2026-09-19]` Live-probed 5 feeds. All returned HTTP 200. Feeds carry
**full article text** in `<content:encoded>` — a sampled Interconnects post gave
11,080 characters of plain text plus 15 outbound links, with `<pubDate>`, from a
single unauthenticated request. 20 items per feed. `substack.com/robots.txt`
disallows `/subscribe`, `/action/`, `/p/*/comment/*` and `/feed/private` — it
does **not** disallow `/feed`, so this usage is compliant.

Paywall handling: if `<content:encoded>` is truncated or carries a
"this post is for paid subscribers" marker, we set `paywalled=true` and store
**only the public excerpt**. We never authenticate, never bypass. A paywalled
post still counts as a mention (we know it exists and who wrote it) but
contributes no body text to scouts.

This is the highest-value free source in the system: full text, named authors,
dates, outbound link graph, and it is exactly where insider argument happens.

### 2.2 Hacker News — `source_role: COMMUNITY_REACTION`

**Two** connectors, because they do different jobs:

- **Firebase API** (`hacker-news.firebaseio.com/v0/`) for the live front page:
  `topstories`, `beststories`, `newstories`, `showstories`, `askstories`, and
  `item/<id>`. `[VERIFIED 2026-09-19]` HTTP 200, no key, and the official docs
  state *"There is currently no rate limit."*
- **Algolia HN Search** (`hn.algolia.com/api/v1/`) for **historical backfill and
  concept counting** — `search`, `search_by_date`, with
  `numericFilters=created_at_i>...`. `[VERIFIED 2026-09-19]` Live-probed: real,
  current, on-topic results with `points`, `num_comments`, `author`,
  `created_at`, `objectID`.

Algolia is what makes day-1 baselines possible (§5.3). Two measured caveats in §4.

### 2.3 arXiv — `source_role: PRIMARY_EVIDENCE`

`https://export.arxiv.org/api/query` (**https**; plain http returns 301).
`[VERIFIED 2026-09-19]` HTTP 200, no key required. Official terms of use:
*"make no more than one request every three seconds, and limit requests to a
single connection at a time."* The connector enforces a global 3s-spaced,
single-connection queue — not per-worker, globally, since the terms bind all
machines under our control.

Driven by `config/arxiv_queries.yaml`: category feeds (cs.AI, cs.LG, cs.CL,
cs.CV, cs.RO, cs.HC, stat.ML) plus phrase searches.

Crucially, a paper's *existence* is a weak signal. Papers earn attention score
from **inbound references** — arXiv IDs extracted from Substack bodies, HN
submissions, X posts, and lab blogs, resolved against `paper_refs`. A paper that
three independent newsletters argue about outranks a paper with a nice abstract.

### 2.4 Primary-source blogs — `source_role: PRIMARY_EVIDENCE` (for their own claims)

`[VERIFIED 2026-09-19]` Probed:

| Org | Feed | Result |
|---|---|---|
| OpenAI | `openai.com/news/rss.xml` | 200 RSS |
| Google DeepMind | `deepmind.google/blog/rss.xml` | 200 RSS |
| Microsoft Research | `microsoft.com/en-us/research/feed/` | 200 RSS |
| NVIDIA Developer | `developer.nvidia.com/blog/feed/` | 200 RSS |
| Hugging Face | `huggingface.co/blog/feed.xml` | 200 RSS |
| Google Research | `research.google/blog/rss/` | 200 RSS |
| BAIR | `bair.berkeley.edu/blog/feed.xml` | 200 RSS |
| Simon Willison | `simonwillison.net/atom/everything/` | 200 Atom |
| **Anthropic** | no RSS found (`/rss.xml`, `/news/rss.xml`, `/engineering/rss.xml`, `/feed.xml` all 404) | **sitemap fallback** |
| **Meta AI** | `ai.meta.com/blog/rss/` and `/feed/` both 404 | **sitemap fallback** |

**Sitemap-diff fallback connector** (the general pattern for no-RSS orgs):
`[VERIFIED 2026-09-19]` `anthropic.com/sitemap.xml` → HTTP 200, 535 URLs, each
with a `<lastmod>`, including 259 `/news/`, 162 `/research/`, 25 `/engineering/`.
We poll the sitemap, diff against seen URLs, and fetch only *new* pages, at a
polite rate. `anthropic.com/robots.txt` is `Allow: /` and advertises the sitemap;
`openai.com/robots.txt` is `Allow: /` except `/microsoft-for-startups/`. The
connector reads and caches `robots.txt` per host and refuses disallowed paths.

**Editorial rule enforced in schema, not prose:** these items get
`primary_or_secondary = PRIMARY` and `source_role = PRIMARY_EVIDENCE`, but a
company blog is primary evidence *for what the company claims*, never evidence
the claim is true. Benchmark claims are stored in a `claimed_capabilities` field
that scout prompts are required to render as "X says Y" — never as "Y".

### 2.5 YouTube — `source_role: EXISTING_COVERAGE`

Official YouTube Data API v3, `YOUTUBE_API_KEY`. Two-step, as specified:
`search.list` to find videos, then `videos.list` (batched, up to 50 IDs) to get
real statistics.

`[VERIFIED 2026-09-19]` **This is the tightest quota in the system.** Official
docs: *"Projects that enable the YouTube Data API have a default quota
allocation of 100 `search.list` calls, 100 `videos.insert` calls, and 10,000
units per day combined for all other endpoints."* So `search.list` is capped at
**100 calls/day as its own budget**, separate from the 10,000-unit pool.
`videos.list` costs 1 unit and comes from the 10k pool, so enrichment is
effectively free; **search is the scarce resource.**

Design consequence: coverage checks run **only after the momentum gate**, on
~12 concept groups/day × 3 orderings (`date`, `viewCount`, `relevance`) = 36
calls, leaving headroom. Coverage results are cached for 7 days per query.

Coverage quality is judged **semantically by the Coverage Scout**, never by
result count. Each video is labelled `tutorial | product_news | interview |
prediction | general_explainer | documentary | conference_talk`, and the gap
metric only counts `general_explainer` and `documentary` with meaningful reach.
100 videos titled "Loop Engineering Tutorial" leave the general-audience gap
wide open, and the schema is built so that falls out arithmetically.

### 2.6 X — `source_role: EARLY_SIGNAL | COMMUNITY_REACTION`

**This is the source whose access model changed most, and the brief's
assumptions about it are now out of date. Details and cost model in §3.2.
It is built, but it is OFF BY DEFAULT in V1.**

Connector design is endpoint-capability-driven exactly as the brief asks:
`config/x_endpoints.yaml` declares which endpoints the authenticated account may
use; the connector probes on startup, records what it actually got, and degrades
to whatever is permitted. Keyword monitoring (`config/x_queries.yaml`) and
curated-account monitoring (`config/x_accounts.yaml`, categorised
AI_RESEARCHERS / AI_BUILDERS / FOUNDERS / INFRASTRUCTURE / ROBOTICS / BIOTECH /
ENERGY / TECH_POLICY / TECH_JOURNALISTS). Follower count is stored as one
observable field among several and is explicitly **not** treated as authority —
authority comes from the curated list tier.

### 2.7 Google Trends — `source_role: PUBLIC_INTEREST`

`[VERIFIED 2026-09-19]` The official Google Trends API exists but is **alpha and
application-gated** (announced 2025-07-24; access granted to approved alpha
testers on a rolling basis; no self-serve key, no published pricing, no GA
timeline). There is no compliant self-serve path today.

Therefore: implemented behind an interface, **default status
`NOT_AVAILABLE`**, and the daily brief prints `Public interest: NOT_AVAILABLE
(Google Trends API access not granted)`. We do **not** ship unofficial scraping,
and we never substitute another metric and call it public interest. The
insider-attention vs public-interest distinction matters too much to fake: it is
the core of the coverage-gap thesis.

### 2.8 Deferred to V2

Reddit, Semantic Scholar, Crossref, PubMed, podcast transcripts, GitHub. **I did
not verify Reddit's current API access terms in this pass** and will not assert
anything about them (§4).

---

## 3. API access status — what works today vs what needs credentials

### 3.1 Summary table

| Source | Credential | Status today | Verification |
|---|---|---|---|
| Substack RSS | none | **WORKS NOW**, full article text | live probe, 5/5 feeds HTTP 200 |
| Hacker News Firebase | none | **WORKS NOW**, no rate limit | live probe + official docs |
| HN Algolia Search | none | **WORKS NOW**, full history | live probe |
| arXiv API | none | **WORKS NOW**, 1 req / 3 s | live probe + official ToU |
| Lab/company RSS | none | **WORKS NOW** (8 of 10 probed) | live probe |
| Sitemap fallback | none | **WORKS NOW** (Anthropic, Meta AI) | live probe, robots checked |
| YouTube Data API | `YOUTUBE_API_KEY` (free) | **Needs key**, free to obtain; 100 search calls/day | official quota docs |
| Anthropic API | `ANTHROPIC_API_KEY` | **Needs key**, paid | — |
| **X API** | credits | **Needs paid credits — no free tier** | official docs |
| **Google Trends** | alpha application | **NOT AVAILABLE** self-serve | official announcement |

**Six of the eleven data sources need no credentials at all and work today.**
That is the whole basis of the V1 recommendation in §9.

### 3.2 X API — the assumption in the brief that no longer holds

The brief says "Use the official X API wherever the authenticated account's
current access tier permits it" and lists `X_BEARER_TOKEN` / `X_API_KEY` /
`X_API_SECRET`. The tier model it implies is gone.

`[VERIFIED 2026-09-19, docs.x.com]`:

- X API is now **pay-per-usage with credits purchased upfront**. Official
  wording: *"The X API uses pay-per-usage pricing. No subscriptions—pay only for
  what you use."*
- **There is no free tier.** The old Free / Basic / Pro tier ladder has been
  retired in favour of credits.
- **Reads are billed per returned resource, not per request**: **$0.005 per
  Post**, $0.010 per user, $0.001 per like. A single search call returning 100
  posts costs **$0.50**.
- **Deduplication:** requesting the same resource again within the same 24-hour
  UTC window is **not** charged again; the window resets at midnight UTC.
- Pay-per-usage is capped at 3M Post reads per monthly billing cycle.
- **Rate limits are separate from billing** and are not tiered: Recent Search
  `/2/tweets/search/recent` is 450/15min per app, 300/15min per user; user
  timelines `/2/users/:id/tweets` 10,000/15min per app.
- **Recent Search covers the last 7 days only.** Full-archive search (back to
  2006, 500 results/request) is **restricted** to pay-per-use and Enterprise
  customers and may require Enterprise.

**Three consequences that shape the architecture:**

1. **Cost governor is mandatory, not optional.** A naive implementation of the
   brief's §D (9 tracked queries, 100 posts each, daily) costs
   `9 × 100 × $0.005 = $4.50/day ≈ $135/month` — and that is the *floor*, before
   curated-account timelines. The connector must therefore: estimate cost before
   every call from `max_results`, check it against a hard daily and monthly USD
   ceiling in config, refuse and return `BUDGET_EXHAUSTED` rather than overspend,
   and log actual spend per call to `source_fetches.cost_usd`.
2. **Exploit the 24h dedup window.** Re-reading the same post on the same UTC day
   is free, so collection is scheduled as **one pass per UTC day** with an
   aggressive local cache — never polled hourly.
3. **X cannot supply its own baseline.** Recent Search sees 7 days; the brief's
   `baseline_mentions` needs ~28. So X baselines are **grown in our own
   database** over the first month (which is what the brief already implies with
   "Store first observed timestamp in our own database"). Until ~14 days of X
   history exist, `x_velocity` reports `INSUFFICIENT_HISTORY` rather than a
   number. **We never compute a velocity ratio against a baseline we don't have.**

**Recommendation: ship V1 with X off.** The free sources already carry the
signal (§8 demonstrates this on real data). Turn X on deliberately, with a
budget, once the rest of the pipeline is earning its keep.

---

## 4. Things I am uncertain about, and measurement traps I found by probing

Stated plainly rather than papered over.

**(1) HN Algolia "first seen" is unreliable for common terms — confirmed trap.**
I probed `search_by_date` for `"prompt engineering"`: it reports
`nbHits=4983` but returns `hitsPerPage=1000, nbPages=1`. Taking `min(created_at)`
over that response yields **2025-10-12**, which is obviously wrong — it is merely
the 1000th-most-recent match. Any naive first-seen implementation would silently
report a years-old term as brand new and rank it as maximally novel.
**Mitigation:** first-observation is authoritative only from our own
`concepts.first_observed_at`; external history is used for *baselines* (counts in
a window, which paginate correctly) and never for novelty claims. Where an
external first-seen is wanted, we paginate ascending rather than trusting a
single page.

**(2) Literal phrase match ≠ same meaning — confirmed trap.** I checked whether
Algolia's quoted search returns literal phrases: it does (18/18 sampled hits
contained the exact phrase). But `"loop engineering"` has HN hits back to
**2012**, which is *control-loop* engineering, a completely unrelated sense.
So a term can look ancient while the *idea* is new. **Mitigation:** every
`concept_mention` stores a `context_snippet`; the novelty computation uses
sense-checked mentions, and the Trend Scout is explicitly asked to separate
`NEW TERM / NEW IDEA / OLD IDEA NEW LABEL / ACTUAL CAPABILITY CHANGE` with the
snippets in front of it.

**(3) Unique-author counts are capped by page size.** My first probe reported
`authors90` values of 79–95 for every term, which is an artifact of
`hitsPerPage=100`, not a finding. **Mitigation:** author counts come from fully
paginated fetches persisted to our DB, never from a single response page. I
flag this because it is exactly the kind of number that looks plausible and is
meaningless.

**(4) Generic tokens pollute counts.** Scanning Substack for `harness` matched
23 posts across all 5 publications — but bare "harness" also matches "harness
the power of". **Mitigation:** the concept registry gates phrases on a
specificity test and prefers multi-word phrases; single generic words require an
explicit alias binding to a canonical concept plus a context check.

**(5) Genuinely uncertain — X full-archive eligibility.** Docs say full-archive
is for "pay-per-use and Enterprise customers" but also hint Enterprise may be
required. I could not resolve this without an authenticated account. **Plan:**
the connector probes the endpoint at startup and records the real answer in
`source_fetches` rather than assuming either way.

**(6) Not verified — Reddit.** The second brief mentions Reddit; I did not check
its current API terms and am not going to guess. Deferred to V2 pending a
verification pass of its own.

**(7) Substack full-text availability varies by publication** and by the
author's RSS settings; some emit excerpt-only. Detected per item at runtime and
recorded in `paywalled` / `text_completeness`, not assumed.

---

## 5. Database schema (SQLite)

Abbreviated DDL; types and constraints are real. All timestamps UTC ISO-8601.

### 5.1 Provenance core

```sql
CREATE TABLE collection_runs (
  id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
  mode TEXT NOT NULL,                 -- collect|trends|discover|daily
  status TEXT NOT NULL, git_sha TEXT, config_hash TEXT, notes TEXT);

-- per-source availability: this is how SOURCE_UNAVAILABLE is made auditable
CREATE TABLE source_fetches (
  id INTEGER PRIMARY KEY, run_id TEXT REFERENCES collection_runs(id),
  connector TEXT NOT NULL, target TEXT NOT NULL,
  status TEXT NOT NULL,               -- ConnectorStatus
  http_status INTEGER, items_seen INTEGER DEFAULT 0, items_new INTEGER DEFAULT 0,
  cost_usd REAL DEFAULT 0, error TEXT, started_at TEXT, finished_at TEXT);

CREATE TABLE publications (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
  homepage_url TEXT, feed_url TEXT, source_role TEXT NOT NULL,
  curated_tier INTEGER,               -- from config only; never LLM-assigned
  robots_checked_at TEXT, notes TEXT);

CREATE TABLE authors (
  id INTEGER PRIMARY KEY, platform TEXT NOT NULL, handle TEXT,
  display_name TEXT, publication_id INTEGER REFERENCES publications(id),
  curated_tier INTEGER, expert_lists TEXT,   -- JSON array of list names
  follower_count INTEGER, follower_count_observed_at TEXT,
  author_reputation_signal TEXT,      -- OBSERVED metadata + curated list only
  credibility_notes TEXT,
  UNIQUE(platform, handle));

CREATE TABLE source_items (
  id TEXT PRIMARY KEY,                -- stable: sha256(source_type|external_id)
  source_type TEXT NOT NULL,          -- x|substack|arxiv|hacker_news|youtube|
                                      -- company_blog|research_blog|podcast|manual
  source_name TEXT NOT NULL, external_id TEXT,
  publication_id INTEGER REFERENCES publications(id),
  author_id INTEGER REFERENCES authors(id),
  title TEXT, text TEXT, url TEXT, canonical_url TEXT,
  published_at TEXT, collected_at TEXT NOT NULL,
  source_role TEXT NOT NULL,          -- EARLY_SIGNAL|PRIMARY_EVIDENCE|...
  primary_or_secondary TEXT NOT NULL,
  paywalled INTEGER DEFAULT 0, text_completeness TEXT,  -- full|excerpt|title_only
  lang TEXT, content_hash TEXT, raw_json TEXT,
  run_id TEXT REFERENCES collection_runs(id),
  UNIQUE(source_type, external_id));

-- engagement is a TIME SERIES, not a column: acceleration needs ≥2 observations
CREATE TABLE engagement_snapshots (
  id INTEGER PRIMARY KEY, source_item_id TEXT REFERENCES source_items(id),
  observed_at TEXT NOT NULL, likes INTEGER, reposts INTEGER, replies INTEGER,
  quotes INTEGER, views INTEGER, score INTEGER, comment_count INTEGER);

CREATE TABLE links (               -- outbound link graph → independence
  id INTEGER PRIMARY KEY, source_item_id TEXT REFERENCES source_items(id),
  url TEXT NOT NULL, canonical_url TEXT,
  target_source_item_id TEXT REFERENCES source_items(id),
  link_type TEXT);                 -- citation|quote_ancestry|paper|self

CREATE TABLE paper_refs (
  id INTEGER PRIMARY KEY, source_item_id TEXT REFERENCES source_items(id),
  arxiv_id TEXT, doi TEXT, title TEXT);

CREATE TABLE org_mentions (
  id INTEGER PRIMARY KEY, source_item_id TEXT REFERENCES source_items(id),
  org_name TEXT NOT NULL, context_snippet TEXT);
```

### 5.2 Concepts and longitudinal measurement

```sql
CREATE TABLE concepts (
  id INTEGER PRIMARY KEY, canonical_label TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,                 -- term|claim|capability|debate
  status TEXT NOT NULL,               -- tracked|nominated|retired|merged
  merged_into INTEGER REFERENCES concepts(id),
  first_observed_at TEXT,             -- OUR observation; authoritative (see §4.1)
  first_source_item_id TEXT REFERENCES source_items(id),
  specificity_score REAL, created_at TEXT, notes TEXT);

CREATE TABLE concept_aliases (        -- what makes IDs stable across days
  id INTEGER PRIMARY KEY, concept_id INTEGER REFERENCES concepts(id),
  alias TEXT NOT NULL, normalized TEXT NOT NULL,
  added_by TEXT NOT NULL,             -- config|auto|llm_registry_maintenance
  UNIQUE(normalized));

CREATE TABLE concept_mentions (       -- exactly the brief's required fields
  id INTEGER PRIMARY KEY,
  concept_id INTEGER REFERENCES concepts(id),
  source_item_id TEXT REFERENCES source_items(id),
  author_id INTEGER REFERENCES authors(id),
  timestamp TEXT NOT NULL,            -- = source_items.published_at
  context TEXT,                       -- snippet; required for sense-checking
  char_offset INTEGER,
  sentiment_or_stance TEXT,           -- advocating|skeptical|neutral|reporting
  confidence REAL, extractor TEXT,    -- lexical|llm:<version>
  independent_cluster_id INTEGER,
  UNIQUE(concept_id, source_item_id, char_offset));

CREATE TABLE concept_daily_stats (    -- precomputed series; makes trends cheap
  concept_id INTEGER, day TEXT, platform TEXT,
  mentions INTEGER, unique_authors INTEGER, unique_publications INTEGER,
  independent_clusters INTEGER, engagement_sum REAL, engagement_median REAL,
  PRIMARY KEY (concept_id, day, platform));

CREATE TABLE concept_metrics (        -- one row per concept per computation
  id INTEGER PRIMARY KEY, concept_id INTEGER REFERENCES concepts(id),
  run_id TEXT, computed_at TEXT NOT NULL,
  mentions_24h INTEGER, mentions_7d INTEGER, mentions_30d INTEGER,
  unique_authors_24h INTEGER, unique_authors_7d INTEGER, unique_publications_7d INTEGER,
  independent_source_count INTEGER, independence_ratio REAL,
  baseline_daily REAL, baseline_days_available INTEGER,
  velocity_ratio REAL, acceleration REAL,
  engagement_sum REAL, engagement_median REAL, engagement_acceleration REAL,
  cross_platform_count INTEGER, cross_role_count INTEGER,
  respected_source_share REAL, novelty REAL,
  paper_count INTEGER, youtube_recent_count INTEGER,
  google_search_change TEXT,          -- nullable; 'NOT_AVAILABLE' is a real value
  insider_momentum REAL, components_json TEXT NOT NULL,
  data_completeness REAL, missing_sources TEXT);

CREATE TABLE signal_propagation (
  id INTEGER PRIMARY KEY, concept_id INTEGER REFERENCES concepts(id),
  origin_source_item_id TEXT REFERENCES source_items(id), origin_date TEXT,
  platform_sequence TEXT,             -- JSON: [{platform,first_seen,item_id},...]
  independent_author_count INTEGER, time_to_cross_platform_hours REAL,
  coverage_gap REAL, computed_at TEXT);
```

### 5.3 Editorial layer — every table joins back to source_items

```sql
CREATE TABLE clusters (
  id TEXT PRIMARY KEY, run_id TEXT REFERENCES collection_runs(id),
  cluster_name TEXT NOT NULL, underlying_change TEXT NOT NULL,
  evidence TEXT, disagreements TEXT, potential_public_relevance TEXT,
  breadth_warning INTEGER DEFAULT 0, created_at TEXT);
CREATE TABLE cluster_concepts (
  cluster_id TEXT, concept_id INTEGER, weight REAL,
  PRIMARY KEY (cluster_id, concept_id));

CREATE TABLE signals (                -- scout output
  id TEXT PRIMARY KEY, run_id TEXT, agent TEXT NOT NULL, cluster_id TEXT,
  short_label TEXT, claim TEXT, concept TEXT, what_is_new TEXT, why_now TEXT,
  actors TEXT, evidence_summary TEXT, disagreement_summary TEXT,
  technical_mechanism TEXT, possible_human_consequence TEXT,
  possible_economic_consequence TEXT, possible_societal_consequence TEXT,
  possible_visuals TEXT, raw_quotes TEXT, tags TEXT,
  confidence REAL, claim_kind TEXT NOT NULL,   -- SOURCE_FACT | INFERENCE
  llm_call_id INTEGER, created_at TEXT);
CREATE TABLE signal_sources (         -- provenance join; validated, see §7
  signal_id TEXT, source_item_id TEXT, role TEXT,
  PRIMARY KEY (signal_id, source_item_id));

CREATE TABLE candidates (
  id TEXT PRIMARY KEY, run_id TEXT, cluster_id TEXT,
  working_title TEXT NOT NULL, alternative_titles TEXT,
  central_question TEXT NOT NULL, one_sentence_premise TEXT NOT NULL,
  real_current_trigger TEXT NOT NULL, why_now TEXT NOT NULL,
  target_viewer TEXT, viewer_problem_solved TEXT,
  technical_mechanism TEXT NOT NULL,
  old_mental_model TEXT NOT NULL, new_mental_model TEXT NOT NULL,
  technical_concepts_to_teach TEXT,
  human_consequence TEXT NOT NULL, economic_consequence TEXT,
  societal_consequence TEXT, stakes TEXT, human_angle TEXT,
  credible_disagreement TEXT NOT NULL, story_engine TEXT,
  people_orgs TEXT, evidence_available TEXT, coverage_gap TEXT,
  visual_opportunities TEXT, strongest_counterargument TEXT,
  what_we_need_to_investigate TEXT, possible_ending_insight TEXT,
  revision_of TEXT REFERENCES candidates(id), created_at TEXT);
CREATE TABLE candidate_sources (
  candidate_id TEXT, source_item_id TEXT, PRIMARY KEY (candidate_id, source_item_id));

CREATE TABLE evaluations (
  id INTEGER PRIMARY KEY, candidate_id TEXT REFERENCES candidates(id),
  evaluator_model TEXT, prompt_version TEXT, run_id TEXT,
  buzz_timeliness INTEGER, public_curiosity INTEGER, stakes INTEGER,
  novelty INTEGER, technical_depth INTEGER, accessibility INTEGER,
  human_relevance INTEGER, serious_disagreement INTEGER, story_motion INTEGER,
  visual_potential INTEGER, evidence_availability INTEGER, coverage_gap INTEGER,
  retellability INTEGER, sponsor_audience_quality INTEGER, expansion_potential INTEGER,
  cares_before_knowing_term INTEGER,   -- binary
  interests_non_ai_person INTEGER,     -- binary
  strongest_failure_reason TEXT, viewer_understanding_delta TEXT,
  old_model TEXT, new_model TEXT, scores_json TEXT, created_at TEXT);

CREATE TABLE pairwise_results (        -- one row PER QUESTION, as specified
  id INTEGER PRIMARY KEY, run_id TEXT,
  candidate_a TEXT, candidate_b TEXT, question_key TEXT NOT NULL,
  winner TEXT, confidence REAL, explanation TEXT,
  judge_model TEXT, order_swapped INTEGER, created_at TEXT);

CREATE TABLE red_team (
  id INTEGER PRIMARY KEY, candidate_id TEXT, attack_key TEXT, verdict TEXT,
  argument TEXT, severity INTEGER, abandon_evidence TEXT,
  judge_model TEXT, created_at TEXT);

CREATE TABLE human_reviews (
  id INTEGER PRIMARY KEY, candidate_id TEXT, run_id TEXT, reviewer TEXT,
  decision TEXT NOT NULL,              -- APPROVE|REJECT|WATCH|RESEARCH_MORE
  favorite INTEGER DEFAULT 0,
  personal_excitation_score INTEGER,   -- 1-10
  would_spend_two_weeks INTEGER,       -- boolean
  notes TEXT, created_at TEXT);

CREATE TABLE youtube_coverage (
  id INTEGER PRIMARY KEY, concept_id INTEGER, candidate_id TEXT,
  topic_query TEXT NOT NULL, search_order TEXT, video_id TEXT,
  title TEXT, channel_title TEXT, channel_id TEXT, subscriber_count INTEGER,
  published_at TEXT, views INTEGER, likes INTEGER, comment_count INTEGER,
  duration_seconds INTEGER, description TEXT,
  content_kind TEXT,                   -- tutorial|product_news|interview|...
  audience TEXT, quality_estimate TEXT, fetched_at TEXT);

CREATE TABLE llm_calls (
  id INTEGER PRIMARY KEY, run_id TEXT, agent TEXT, model TEXT,
  prompt_version TEXT, input_hash TEXT, input_tokens INTEGER,
  output_tokens INTEGER, cost_usd REAL, latency_ms INTEGER,
  cache_hit INTEGER, retry_count INTEGER, created_at TEXT);

-- empty in V1; exists so outcome learning can attach later without migration
CREATE TABLE video_outcomes (
  id INTEGER PRIMARY KEY, candidate_id TEXT REFERENCES candidates(id),
  final_title TEXT, published_video_id TEXT, published_at TEXT,
  ctr REAL, avg_view_duration_s REAL, views_30d INTEGER,
  comment_count INTEGER, sponsor_revenue REAL, qualitative_notes TEXT);
```

---

## 6. Independence: the attribution graph

The brief's hardest requirement — *distinguish "one essay people are copying"
from "multiple independent people arriving at related ideas"* — is not solvable
by counting authors. It is a graph problem.

For a given concept and window, build a graph over the items mentioning it:

| Edge | Evidence | Meaning |
|---|---|---|
| same author | `author_id` | not independent |
| explicit citation | `links.target_source_item_id` resolves to another item | derived |
| shared origin URL | two items link the same `canonical_url` | common ancestor |
| shared paper | same `arxiv_id` in `paper_refs` | common ancestor |
| quote/repost ancestry | X referenced-tweet chain | derived |
| near-duplicate phrasing | shingled Jaccard > threshold on snippets | copied |

`independent_source_count` = number of **weakly connected components**.
`independence_ratio` = `independent_source_count / distinct_authors` ∈ (0,1].

- Five posts, one author → 1 component. Ratio 0.2.
- Four writers all citing one origin essay → **1 component**, because the shared
  origin links them. Ratio 0.2. *This is the case the brief cares most about.*
- Four writers, no shared citations, converging independently → 4 components.
  Ratio 1.0. **This is the buy signal.**

`independence_ratio` then multiplies `INSIDER_MOMENTUM` (§7), so a viral
single-origin cascade is arithmetically suppressed rather than talked down by a
prompt. Both the raw and discounted momentum are stored, so an editor can always
see what the discount did.

---

## 7. The scoring model

### 7.1 Component definitions

All components normalised to [0,1]; ratios are log-squashed so one 40× spike
doesn't dominate. `squash(r) = min(1, ln(1+r) / ln(1+CAP))`, default `CAP=10`.

```
baseline_daily   = median(daily mentions over trailing 28d, excluding last 3d)
velocity_ratio   = mentions_24h / max(baseline_daily, 0.5)
normalized_velocity = squash(velocity_ratio)
    → NOT computed if baseline_days_available < 14; emits INSUFFICIENT_HISTORY

unique_author_growth = squash(unique_authors_7d / max(unique_authors_prev7d,1))

cross_platform_presence = 0.6 * (platforms_with_independent_mention_7d / 5)
                        + 0.4 * (distinct_source_roles_present / 4)
    -- roles matter more than platforms: EARLY_SIGNAL + PRIMARY_EVIDENCE +
    -- COMMUNITY_REACTION is a real propagation chain; three EARLY_SIGNALs is an echo

respected_source_presence = share of independent clusters containing ≥1
    curated-tier author/publication          -- curated lists ONLY, never LLM

engagement_acceleration = squash(median_engagement_per_mention_7d /
                                 max(median_engagement_per_mention_prev28d, ε))

novelty = 0.5 * clamp(1 - age_days(first_observed_at)/180)
        + 0.5 * (1 - corpus_share_before_90d)    -- sense-checked mentions only
```

### 7.2 Insider momentum

```
INSIDER_MOMENTUM_raw =
    0.30 * normalized_velocity
  + 0.20 * unique_author_growth
  + 0.15 * cross_platform_presence
  + 0.15 * respected_source_presence
  + 0.10 * engagement_acceleration
  + 0.10 * novelty

INSIDER_MOMENTUM = INSIDER_MOMENTUM_raw * independence_ratio^0.5
```

Weights live in `config/editorial.yaml`, not in code. Every component is stored
in `concept_metrics.components_json`, so when outcome data eventually arrives
the weights can be refit — which is impossible if only the composite is kept.

If a component is unavailable, it is **excluded and the weights renormalise**,
and `data_completeness` drops. A missing input never silently becomes zero: a
concept with no baseline yet must not be rankable as "not accelerating".

### 7.3 Editorial opportunity

Momentum ranks what is *moving*. It must not rank what is *worth making*.

```
EDITORIAL_OPPORTUNITY =
    weighted_geometric_mean(
        insider_momentum        w=0.20,
        consequence             w=0.25,   -- LLM-judged, source-cited
        public_confusion        w=0.20,   -- LLM + Trends when available
        story_potential         w=0.15,   -- Story Scout
        technical_substance     w=0.20    -- Frontier/Mechanism Scout
    ) * coverage_adjustment

coverage_adjustment = 1 - 0.7 * general_audience_saturation
```

Geometric mean implements the brief's `×` semantics — a near-zero on *any*
dimension collapses the score, which is correct: a huge trend with no
consequence is not a video, and a big consequence with no technical substance is
punditry. Components are clamped to [0.05, 1.0] so one missing judgement can't
zero out an otherwise strong candidate outright.

`general_audience_saturation` counts only YouTube results labelled
`general_explainer` or `documentary` **with meaningful reach**, weighted by
recency. Tutorials and product-news videos contribute ~0. This is how "100 Loop
Engineering Tutorials" correctly yields a saturation near 0 and leaves the
opportunity wide open.

### 7.4 Provenance enforcement (mechanical, not aspirational)

Every scout returns claims shaped as:

```python
class Claim(BaseModel):
    text: str
    kind: Literal["SOURCE_FACT", "INFERENCE"]
    source_ids: list[str]
    confidence: float
    @model_validator(mode="after")
    def _check(self):
        if self.kind == "SOURCE_FACT" and not self.source_ids:
            raise ValueError("SOURCE_FACT requires ≥1 source_id")
        return self
```

A post-validation pass rejects any `source_id` **not present in the input set**
for that call — i.e. hallucinated citations are caught structurally, not by
reading the output. Violations trigger one repair retry with the error fed back,
then the claim is dropped and the drop is logged. `SOURCE_FACT` and `INFERENCE`
live in the same table with a discriminating column, so no rendering path can
blur them: the brief calls this requirement absolute, and a column is the only
way to actually make it so.

---

## 8. Worked example on real data collected today

Following the brief's requested example — the *loop / graph / context / harness
engineering* family. **The numbers below are real**, measured 2026-09-19 from
free sources during this architecture pass.

### Step 1 — Ingestion (what the connectors actually returned)

Hacker News via Algolia, phrase-matched, stories+comments:

| phrase | all-time | 90d | 7d |
|---|---|---|---|
| `agent harness` | 912 | 404 | **30** |
| `prompt engineering` | 4983 | 188 | 7 |
| `harness engineering` | 266 | 104 | 6 |
| `agentic loop` | 308 | 67 | 6 |
| `context rot` | 287 | 44 | 8 |
| `context engineering` | 787 | 93 | 3 |
| `loop engineering` | 111 | 78 | 0 |
| `verification bottleneck` | 7 | 2 | 0 |

Substack, 5 curated feeds, 20 most recent posts each (full text):

| token | posts | publications |
|---|---|---|
| `harness` | 23 | **5 of 5** — Interconnects, OneUsefulThing, PragmaticEngineer, SimonWillison, Zvi |
| `verification` | 11 | 4 |
| `orchestrat*` | 7 | 3 |
| `subagent` | 5 | 3 |
| `loop engineering` | 3 | 1 (PragmaticEngineer) |
| `context engineering` | 2 | 1 (PragmaticEngineer) |

arXiv, most recent cs.AI: *"Coding Agents with an Obstacle-Aware **Harness** for
Safe Robot Manipulation"* — the vocabulary has reached the paper titles.

Also caught today (2026-09-19, Interconnects): *"Why I still haven't bought into
true RSI"* — opening on *"a few organizations using **thousands of concurrent
agents** to improve their processes"*. Note this is a **skeptical** post, which
is stance-tagged, not counted as advocacy.

### Step 2 — Computed momentum (honest about what day 1 can't do)

`agent harness`: 30 mentions/7d vs a 90d run-rate of ~4.5/day → an
elevated but not explosive ratio; `harness engineering` at 6/7d over a small
base is the sharper riser. **But** `baseline_days_available = 0` for our own DB
on day 1, so `normalized_velocity` is computed from the Algolia backfill and
flagged `source: external_backfill`, not from our own accumulated history.

The `loop engineering` row is the instructive one: **78 mentions in 90 days, 0
in the last 7.** Whatever this phrase was doing, it is not what is moving now.
A system that took the brief's example list at face value would have generated a
video about a decaying term. The measurement layer is what prevents that — this
is precisely the "trending term ≠ good video" failure, caught arithmetically.

`verification bottleneck` as a *phrase* is nearly absent (7 all-time). But
`verification` appears in 11 posts across 4 of 5 newsletters. So the **idea** is
live while the **label** is not — exactly the `NEW IDEA` vs `NEW TERM`
distinction the Trend Scout must draw, and a case where phrase-counting alone
would return a false negative.

### Step 3 — Independence check

Five publications mention `harness`. The attribution graph asks whether they are
echoing one origin. Provisionally the Substack hits span Apr 2025 → Sep 2026
across unrelated posts with no common cited origin → multiple components →
`independence_ratio` high → **this is convergence, not a cascade.** That is the
finding that earns an LLM call.

### Step 4 — Clustering

Deterministic pre-cluster groups {agent harness, harness engineering, agentic
loop, context engineering, subagent, orchestration, verification} by
co-occurrence. The LLM Clusterer is then asked for the *underlying change*:

> **Underlying change:** the unit of AI engineering is shifting from the message
> to the surrounding apparatus — the harness, the loop, the context, the
> verifier. The scarce skill is no longer phrasing a request but building a
> structure that keeps a model working and checks whether its output is right.

Note what this correctly *excludes*: `loop engineering` is dropped for lack of
current movement, and `prompt engineering` is retained only as the **old mental
model**, not as a live trend.

### Step 5 — Scout interpretations (with required provenance)

- **Trend Scout** → `NEW IDEA, PARTIALLY NEW TERM`. "harness" is spreading across
  independent publications; "loop engineering" is decaying. *Trap flagged: the
  2012 HN hits for "loop engineering" are control-loop engineering, a different
  sense.*
- **Frontier Scout** → the real change is that agents now run long enough for the
  scaffolding to dominate outcomes; the harness vocabulary has reached arXiv
  titles. Labelled `INFERENCE` where not directly stated by a source.
- **Disagreement Scout** → a genuine, sourced split. *View A:* scaffolding is
  where the remaining gains are. *View B* (Interconnects, 2026-09-19, explicitly
  skeptical of true RSI): the loop is not the bottleneck and concurrency is being
  oversold. This disagreement is **found, not manufactured** — the brief forbids
  manufacturing balance, and here both sides have real proponents with URLs.
- **Mechanism Scout** → the concept the audience actually needs is **verification**
  — how a system knows its own output is correct — not "harness".
- **Human Consequence Scout** → the reviewable unit of work changes: people
  increasingly specify objectives and acceptance criteria rather than steps.
- **Coverage Scout** → *pending YouTube key.* Reports
  `SOURCE_UNAVAILABLE: YOUTUBE_API_KEY not configured`, and every downstream
  coverage-gap score is marked incomplete. **It does not guess.**

### Step 6 — Candidates

Applying the Translator's insider→public rule:

1. **"Who Checks the AI?"** — central question: *if machines now produce more
   work than humans can review, what happens to the idea of checking?* Mechanism:
   verification. Old model: AI is a thing you ask. New model: AI is a process you
   supervise. Disagreement: real and sourced. Coverage gap: unknown pending key.
2. **"The End of Prompting"** — the brief's own exemplar. Momentum data says the
   *prompting* half is decaying (that's the point — it's the old model) while the
   *harness* half is rising. Strong title, but Red Team will press on whether the
   evidence supports the finality the title claims.
3. **"What Happens When One AI Becomes a Team?"** — carried by `subagent` (3
   pubs) + concurrency claims. Weaker: it is closer to an old orchestration idea
   with new vocabulary, and the system should say so rather than sell it.

### Step 7 — Evaluation

Candidate 1 scores highest on *stakes*, *serious disagreement* and *mechanism*,
and passes both binaries — an intelligent non-specialist cares about "who checks
the work" without knowing the word "harness". Candidate 3 fails
`cares_before_knowing_term`. Red Team's strongest attack on candidate 1 is
evidence burden: the claim is currently carried by essays and one skeptical post,
not by a measured result — so the brief is required to state what evidence would
settle it.

**Conclusion of the walkthrough:** the free sources alone produced a real,
defensible, non-obvious editorial finding today — including correctly *killing*
one of the brief's own example terms. That is the case for the V1 below.

---

## 9. Recommended V1 — real data this week

**Principle: ship the half that needs no credentials, because it is the half
that carries the signal.**

### In scope

1. SQLite schema + Pydantic models + migrations (§5)
2. Connectors: **Substack RSS, Hacker News (Firebase + Algolia), arXiv, primary
   blogs RSS + sitemap fallback** — all four need zero credentials
3. Normalizer, concept registry + alias matching, novel-phrase nomination
4. Attribution graph / independence (§6)
5. Trend engine + `INSIDER_MOMENTUM` + propagation detection (§7)
6. **90-day backfill on first run** — Algolia + arXiv + Substack history mean
   day-1 baselines are real, not empty
7. LLM layer: provider abstraction, cache, cost tracking, provenance validator
8. Scouts (all 8), Clusterer, Translator, Generator, Evaluator, Pairwise, Red Team
9. `daily_brief.md` + `frontier-scout collect | trends | discover | review | daily`
10. Tests: schema validation, provenance preservation, hallucinated-citation
    rejection, cache hits, pairwise ranking, malformed-output retry, persistence

### Deliberately deferred

- **YouTube** — one config line away; add the day you have the key. Until then
  coverage reports `SOURCE_UNAVAILABLE` and scores carry reduced completeness.
- **X** — built behind a flag, **off by default**, with the budget governor
  (§3.2) in place before the first paid call is ever made.
- **Google Trends** — interface only; `NOT_AVAILABLE` until alpha access.
- Reddit, Semantic Scholar, Crossref, PubMed, podcasts.

### Build order

Days 1–2: schema + the four free connectors + normalizer → *real rows in a real
database*. Day 3: concept registry + independence + trend math → *a ranked
momentum table with no LLM at all.* Days 4–5: LLM layer + scouts + clusterer.
Day 6: candidates + evaluation + pairwise + red team. Day 7: brief + review CLI.

The system is useful at the end of day 3, before a single token is spent.

---

## 10. Decisions I need from you before writing code

1. **Ship V1 with X off?** (My recommendation: yes — no free tier, ~$135/month at
   the brief's stated query volume, and today's walkthrough found the signal
   without it. Say the word and I'll wire it on with a hard budget cap.)
2. **Clustering order.** The brief describes scouts→clusterer in one place and
   clusterer→translator→generator in another. I propose **deterministic
   pre-cluster → scouts on groups → LLM clusterer consolidates**, because scouts
   reason better with a concept family than a single term, and it cuts LLM calls
   by roughly the cluster size. Confirm or override.
3. **Gate width.** Default `N=12` concept groups/day through the expensive stage.
4. **Daily LLM budget ceiling** (USD) — drives model choice per stage.
5. **Seed lists.** I'll draft `substack_feeds.yaml` / `arxiv_queries.yaml` /
   `primary_feeds.yaml` from the verified-working feeds above; you'll want to
   edit them. Any must-have publications to include from the start?
6. **Backfill depth** — 90 days proposed. Deeper costs only patience (arXiv is
   3s/request).

---

**Awaiting approval. No implementation will begin until you approve.**
