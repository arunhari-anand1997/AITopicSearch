"""SQLite persistence.

Everything the system observes or infers lands here, and every editorial row
joins back to `source_items`. The schema is deliberately wide rather than
normalised-to-death: this is an inspectable research database, and being able
to read a table with the sqlite3 CLI is worth more than saving a few columns.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .models import (
    CandidateVideo,
    Cluster,
    CollectionResult,
    ConceptMetrics,
    Evaluation,
    HumanReview,
    LLMCall,
    PairwiseResult,
    RedTeamFinding,
    Signal,
    SignalPropagation,
    SourceItem,
    iso,
)

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS collection_runs (
  id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  git_sha TEXT,
  config_hash TEXT,
  notes TEXT);

-- Per-source availability. This is what makes SOURCE_UNAVAILABLE auditable
-- instead of a silently empty result set.
CREATE TABLE IF NOT EXISTS source_fetches (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES collection_runs(id),
  connector TEXT NOT NULL,
  target TEXT NOT NULL,
  status TEXT NOT NULL,
  http_status INTEGER,
  items_seen INTEGER DEFAULT 0,
  items_new INTEGER DEFAULT 0,
  cost_usd REAL DEFAULT 0,
  error TEXT,
  started_at TEXT,
  finished_at TEXT);

CREATE TABLE IF NOT EXISTS publications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  homepage_url TEXT,
  feed_url TEXT,
  source_role TEXT NOT NULL,
  curated_tier INTEGER,
  robots_checked_at TEXT,
  notes TEXT);

CREATE TABLE IF NOT EXISTS authors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  platform TEXT NOT NULL,
  handle TEXT,
  display_name TEXT,
  publication_id INTEGER REFERENCES publications(id),
  curated_tier INTEGER,
  expert_lists TEXT,
  follower_count INTEGER,
  follower_count_observed_at TEXT,
  author_reputation_signal TEXT,
  credibility_notes TEXT,
  UNIQUE(platform, handle));

CREATE TABLE IF NOT EXISTS source_items (
  id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_name TEXT NOT NULL,
  external_id TEXT,
  publication_id INTEGER REFERENCES publications(id),
  author_id INTEGER REFERENCES authors(id),
  title TEXT,
  text TEXT,
  url TEXT,
  canonical_url TEXT,
  published_at TEXT,
  collected_at TEXT NOT NULL,
  source_role TEXT NOT NULL,
  primary_or_secondary TEXT NOT NULL,
  paywalled INTEGER DEFAULT 0,
  text_completeness TEXT,
  lang TEXT,
  content_hash TEXT,
  raw_json TEXT,
  run_id TEXT REFERENCES collection_runs(id),
  UNIQUE(source_type, external_id));
CREATE INDEX IF NOT EXISTS idx_items_published ON source_items(published_at);
CREATE INDEX IF NOT EXISTS idx_items_type ON source_items(source_type);

CREATE TABLE IF NOT EXISTS engagement_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_item_id TEXT REFERENCES source_items(id),
  observed_at TEXT NOT NULL,
  likes INTEGER, reposts INTEGER, replies INTEGER, quotes INTEGER,
  views INTEGER, score INTEGER, comment_count INTEGER);
CREATE INDEX IF NOT EXISTS idx_eng_item ON engagement_snapshots(source_item_id);

CREATE TABLE IF NOT EXISTS links (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_item_id TEXT REFERENCES source_items(id),
  url TEXT NOT NULL,
  canonical_url TEXT,
  target_source_item_id TEXT REFERENCES source_items(id),
  link_type TEXT);
CREATE INDEX IF NOT EXISTS idx_links_item ON links(source_item_id);
CREATE INDEX IF NOT EXISTS idx_links_canon ON links(canonical_url);

CREATE TABLE IF NOT EXISTS paper_refs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_item_id TEXT REFERENCES source_items(id),
  arxiv_id TEXT, doi TEXT, title TEXT);
CREATE INDEX IF NOT EXISTS idx_paper_arxiv ON paper_refs(arxiv_id);

CREATE TABLE IF NOT EXISTS org_mentions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_item_id TEXT REFERENCES source_items(id),
  org_name TEXT NOT NULL,
  context_snippet TEXT);

-- Concept identity is lexical and stable. Models may propose new concepts and
-- aliases, but they never do the counting: a re-labelled concept would silently
-- fragment its own time series.
CREATE TABLE IF NOT EXISTS concepts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_label TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  merged_into INTEGER REFERENCES concepts(id),
  first_observed_at TEXT,
  first_source_item_id TEXT REFERENCES source_items(id),
  specificity_score REAL,
  created_at TEXT,
  notes TEXT);

CREATE TABLE IF NOT EXISTS concept_aliases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id INTEGER REFERENCES concepts(id),
  alias TEXT NOT NULL,
  normalized TEXT NOT NULL UNIQUE,
  added_by TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS concept_mentions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id INTEGER REFERENCES concepts(id),
  source_item_id TEXT REFERENCES source_items(id),
  author_id INTEGER REFERENCES authors(id),
  timestamp TEXT,
  context TEXT,
  char_offset INTEGER,
  sentiment_or_stance TEXT,
  confidence REAL,
  extractor TEXT,
  independent_cluster_id INTEGER,
  UNIQUE(concept_id, source_item_id, char_offset));
CREATE INDEX IF NOT EXISTS idx_cm_concept ON concept_mentions(concept_id);
CREATE INDEX IF NOT EXISTS idx_cm_time ON concept_mentions(timestamp);

CREATE TABLE IF NOT EXISTS concept_daily_stats (
  concept_id INTEGER, day TEXT, platform TEXT,
  mentions INTEGER, unique_authors INTEGER, unique_publications INTEGER,
  independent_clusters INTEGER, engagement_sum REAL, engagement_median REAL,
  PRIMARY KEY (concept_id, day, platform));

CREATE TABLE IF NOT EXISTS concept_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id INTEGER REFERENCES concepts(id),
  run_id TEXT,
  computed_at TEXT NOT NULL,
  mentions_24h INTEGER, mentions_7d INTEGER, mentions_30d INTEGER,
  unique_authors_24h INTEGER, unique_authors_7d INTEGER,
  unique_publications_7d INTEGER,
  independent_source_count INTEGER, independence_ratio REAL,
  baseline_daily REAL, baseline_days_available INTEGER,
  velocity_ratio REAL, acceleration REAL,
  engagement_sum REAL, engagement_median REAL, engagement_acceleration REAL,
  cross_platform_count INTEGER, cross_role_count INTEGER,
  respected_source_share REAL, novelty REAL,
  paper_count INTEGER, youtube_recent_count INTEGER,
  google_search_change TEXT,
  insider_momentum REAL, insider_momentum_raw REAL,
  components_json TEXT NOT NULL,
  data_completeness REAL, missing_sources TEXT, notes TEXT);
CREATE INDEX IF NOT EXISTS idx_metrics_concept ON concept_metrics(concept_id);

CREATE TABLE IF NOT EXISTS signal_propagation (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id INTEGER REFERENCES concepts(id),
  origin_source_item_id TEXT REFERENCES source_items(id),
  origin_date TEXT,
  platform_sequence TEXT,
  independent_author_count INTEGER,
  time_to_cross_platform_hours REAL,
  coverage_gap REAL,
  computed_at TEXT);

CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY,
  run_id TEXT REFERENCES collection_runs(id),
  cluster_name TEXT NOT NULL,
  underlying_change TEXT NOT NULL,
  raw_terms TEXT,
  evidence TEXT,
  disagreements TEXT,
  potential_public_relevance TEXT,
  breadth_warning INTEGER DEFAULT 0,
  momentum REAL,
  created_at TEXT);

CREATE TABLE IF NOT EXISTS cluster_concepts (
  cluster_id TEXT REFERENCES clusters(id),
  concept_id INTEGER REFERENCES concepts(id),
  weight REAL,
  PRIMARY KEY (cluster_id, concept_id));

CREATE TABLE IF NOT EXISTS signals (
  id TEXT PRIMARY KEY,
  run_id TEXT REFERENCES collection_runs(id),
  agent TEXT NOT NULL,
  cluster_id TEXT,
  short_label TEXT, claim TEXT, concept TEXT,
  what_is_new TEXT, why_now TEXT, actors TEXT,
  evidence_summary TEXT, disagreement_summary TEXT,
  technical_mechanism TEXT,
  possible_human_consequence TEXT,
  possible_economic_consequence TEXT,
  possible_societal_consequence TEXT,
  possible_visuals TEXT, raw_quotes TEXT, tags TEXT,
  claims_json TEXT,
  confidence REAL,
  created_at TEXT);

CREATE TABLE IF NOT EXISTS signal_sources (
  signal_id TEXT REFERENCES signals(id),
  source_item_id TEXT REFERENCES source_items(id),
  role TEXT,
  PRIMARY KEY (signal_id, source_item_id));

CREATE TABLE IF NOT EXISTS candidates (
  id TEXT PRIMARY KEY,
  run_id TEXT REFERENCES collection_runs(id),
  cluster_id TEXT,
  working_title TEXT NOT NULL,
  alternative_titles TEXT,
  central_question TEXT NOT NULL,
  one_sentence_premise TEXT NOT NULL,
  real_current_trigger TEXT,
  why_now TEXT,
  target_viewer TEXT,
  viewer_problem_solved TEXT,
  technical_mechanism TEXT,
  old_mental_model TEXT,
  new_mental_model TEXT,
  technical_concepts_to_teach TEXT,
  stakes TEXT,
  human_consequence TEXT,
  economic_consequence TEXT,
  societal_consequence TEXT,
  human_angle TEXT,
  credible_disagreement TEXT,
  story_engine TEXT,
  people_orgs TEXT,
  evidence_available TEXT,
  coverage_gap TEXT,
  visual_opportunities TEXT,
  strongest_counterargument TEXT,
  what_we_need_to_investigate TEXT,
  possible_ending_insight TEXT,
  revision_of TEXT REFERENCES candidates(id),
  created_at TEXT);

CREATE TABLE IF NOT EXISTS candidate_sources (
  candidate_id TEXT REFERENCES candidates(id),
  source_item_id TEXT REFERENCES source_items(id),
  PRIMARY KEY (candidate_id, source_item_id));

CREATE TABLE IF NOT EXISTS evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id TEXT REFERENCES candidates(id),
  run_id TEXT,
  evaluator_model TEXT, prompt_version TEXT,
  buzz_timeliness INTEGER, public_curiosity INTEGER, stakes INTEGER,
  novelty INTEGER, technical_depth INTEGER, accessibility INTEGER,
  human_relevance INTEGER, serious_disagreement INTEGER, story_motion INTEGER,
  visual_potential INTEGER, evidence_availability INTEGER, coverage_gap INTEGER,
  retellability INTEGER, sponsor_audience_quality INTEGER,
  expansion_potential INTEGER,
  cares_before_knowing_term INTEGER,
  interests_non_ai_person INTEGER,
  strongest_failure_reason TEXT,
  viewer_understanding_delta TEXT,
  old_model TEXT, new_model TEXT,
  mean_score REAL,
  scores_json TEXT,
  created_at TEXT);

CREATE TABLE IF NOT EXISTS pairwise_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT,
  candidate_a TEXT, candidate_b TEXT,
  question_key TEXT NOT NULL,
  winner TEXT, confidence REAL, explanation TEXT,
  judge_model TEXT, order_swapped INTEGER,
  created_at TEXT);

CREATE TABLE IF NOT EXISTS red_team (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id TEXT REFERENCES candidates(id),
  run_id TEXT,
  attack_key TEXT, verdict TEXT, argument TEXT,
  severity INTEGER, abandon_evidence TEXT,
  judge_model TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS human_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id TEXT REFERENCES candidates(id),
  run_id TEXT,
  reviewer TEXT,
  decision TEXT NOT NULL,
  favorite INTEGER DEFAULT 0,
  personal_excitation_score INTEGER,
  would_spend_two_weeks INTEGER,
  notes TEXT,
  created_at TEXT);

CREATE TABLE IF NOT EXISTS youtube_coverage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id INTEGER, candidate_id TEXT, cluster_id TEXT,
  topic_query TEXT NOT NULL, search_order TEXT,
  video_id TEXT, title TEXT, channel_title TEXT, channel_id TEXT,
  subscriber_count INTEGER, published_at TEXT,
  views INTEGER, likes INTEGER, comment_count INTEGER,
  duration_seconds INTEGER, description TEXT,
  content_kind TEXT, audience TEXT, quality_estimate TEXT,
  fetched_at TEXT);
CREATE INDEX IF NOT EXISTS idx_yt_query ON youtube_coverage(topic_query);

CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT, agent TEXT, model TEXT, prompt_version TEXT,
  input_hash TEXT, input_tokens INTEGER, output_tokens INTEGER,
  cost_usd REAL, latency_ms INTEGER, cache_hit INTEGER,
  retry_count INTEGER, created_at TEXT);

-- Empty in V1. Exists so outcome learning can attach without a migration:
-- the point of storing every scoring component is to refit weights against
-- these columns once real videos have shipped.
CREATE TABLE IF NOT EXISTS video_outcomes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id TEXT REFERENCES candidates(id),
  final_title TEXT, published_video_id TEXT, published_at TEXT,
  ctr REAL, avg_view_duration_s REAL, views_30d INTEGER,
  comment_count INTEGER, sponsor_revenue REAL, qualitative_notes TEXT);
"""


def _j(value: Any) -> str | None:
    """JSON-encode lists/dicts for storage; pass through None."""
    if value is None:
        return None
    return json.dumps(value, default=str)


class Database:
    def __init__(self, path: str | Path = "data/frontier_scout.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    # -- lifecycle ---------------------------------------------------------

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        self.conn.close()

    # -- runs --------------------------------------------------------------

    def start_run(self, run_id: str, mode: str, config_hash: str | None = None) -> str:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO collection_runs(id,started_at,mode,status,config_hash)"
                " VALUES(?,?,?,?,?)",
                (run_id, datetime.now(timezone.utc).isoformat(), mode, "running", config_hash),
            )
        return run_id

    def finish_run(self, run_id: str, status: str = "ok", notes: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE collection_runs SET finished_at=?,status=?,notes=? WHERE id=?",
                (datetime.now(timezone.utc).isoformat(), status, notes, run_id),
            )

    # -- publications / authors -------------------------------------------

    def upsert_publication(
        self,
        name: str,
        kind: str,
        source_role: str,
        homepage_url: str | None = None,
        feed_url: str | None = None,
        curated_tier: int | None = None,
    ) -> int:
        with self.tx() as c:
            c.execute(
                "INSERT INTO publications(name,kind,source_role,homepage_url,feed_url,curated_tier)"
                " VALUES(?,?,?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET kind=excluded.kind,"
                " source_role=excluded.source_role, feed_url=COALESCE(excluded.feed_url,feed_url),"
                " curated_tier=COALESCE(excluded.curated_tier,curated_tier)",
                (name, kind, source_role, homepage_url, feed_url, curated_tier),
            )
            row = c.execute("SELECT id FROM publications WHERE name=?", (name,)).fetchone()
        return int(row["id"])

    def upsert_author(
        self,
        platform: str,
        handle: str | None,
        display_name: str | None = None,
        publication_id: int | None = None,
        curated_tier: int | None = None,
        expert_lists: list[str] | None = None,
        follower_count: int | None = None,
    ) -> int | None:
        if not handle and not display_name:
            return None
        handle = handle or display_name
        with self.tx() as c:
            c.execute(
                "INSERT INTO authors(platform,handle,display_name,publication_id,curated_tier,"
                "expert_lists,follower_count,follower_count_observed_at) VALUES(?,?,?,?,?,?,?,?)"
                " ON CONFLICT(platform,handle) DO UPDATE SET"
                " display_name=COALESCE(excluded.display_name,display_name),"
                " publication_id=COALESCE(excluded.publication_id,publication_id),"
                " curated_tier=COALESCE(excluded.curated_tier,curated_tier),"
                " expert_lists=COALESCE(excluded.expert_lists,expert_lists),"
                " follower_count=COALESCE(excluded.follower_count,follower_count)",
                (
                    platform,
                    handle,
                    display_name,
                    publication_id,
                    curated_tier,
                    _j(expert_lists),
                    follower_count,
                    datetime.now(timezone.utc).isoformat() if follower_count is not None else None,
                ),
            )
            row = c.execute(
                "SELECT id FROM authors WHERE platform=? AND handle=?", (platform, handle)
            ).fetchone()
        return int(row["id"]) if row else None

    # -- source items ------------------------------------------------------

    def insert_source_item(
        self, item: SourceItem, run_id: str | None = None, curated_tier: int | None = None
    ) -> bool:
        """Insert an item. Returns True if it was new.

        Engagement is appended as a snapshot every time, even for items we have
        seen before, so that engagement acceleration has a series to work with.
        """
        pub_id = None
        if item.publication:
            pub_id = self.upsert_publication(
                item.publication,
                kind=str(item.source_type),
                source_role=str(item.source_role),
                curated_tier=curated_tier,
            )
        author_id = self.upsert_author(
            platform=str(item.source_type),
            handle=item.author_handle,
            display_name=item.author_name,
            publication_id=pub_id,
            curated_tier=curated_tier,
        )
        with self.tx() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO source_items(id,source_type,source_name,external_id,"
                "publication_id,author_id,title,text,url,canonical_url,published_at,collected_at,"
                "source_role,primary_or_secondary,paywalled,text_completeness,lang,content_hash,"
                "raw_json,run_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item.id,
                    str(item.source_type),
                    item.source_name,
                    item.external_id,
                    pub_id,
                    author_id,
                    item.title,
                    item.text,
                    item.url,
                    item.canonical_url or item.url,
                    iso(item.published_at),
                    iso(item.collected_at),
                    str(item.source_role),
                    item.primary_or_secondary,
                    int(item.paywalled),
                    str(item.text_completeness),
                    item.lang,
                    None,
                    _j(item.metadata),
                    run_id,
                ),
            )
            is_new = cur.rowcount > 0

            if item.engagement:
                e = item.engagement
                c.execute(
                    "INSERT INTO engagement_snapshots(source_item_id,observed_at,likes,reposts,"
                    "replies,quotes,views,score,comment_count) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        item.id,
                        iso(e.observed_at),
                        e.likes,
                        e.reposts,
                        e.replies,
                        e.quotes,
                        e.views,
                        e.score,
                        e.comment_count,
                    ),
                )

            if is_new:
                for url in item.outbound_links:
                    c.execute(
                        "INSERT INTO links(source_item_id,url,canonical_url,link_type)"
                        " VALUES(?,?,?,?)",
                        (item.id, url, url, "citation"),
                    )
                for ref in item.arxiv_refs:
                    c.execute(
                        "INSERT INTO paper_refs(source_item_id,arxiv_id) VALUES(?,?)",
                        (item.id, ref),
                    )
        return is_new

    def record_fetch(self, result: CollectionResult, run_id: str | None, items_new: int) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO source_fetches(run_id,connector,target,status,http_status,items_seen,"
                "items_new,cost_usd,error,started_at,finished_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    result.connector,
                    result.target,
                    str(result.status),
                    result.http_status,
                    len(result.items),
                    items_new,
                    result.cost_usd,
                    result.error,
                    iso(result.started_at),
                    iso(result.finished_at),
                ),
            )

    def items_since(self, since: datetime, limit: int | None = None) -> list[sqlite3.Row]:
        sql = (
            "SELECT * FROM source_items WHERE published_at >= ? ORDER BY published_at DESC"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.conn.execute(sql, (iso(since),)).fetchall()

    def get_items(self, ids: Iterable[str]) -> dict[str, sqlite3.Row]:
        ids = list(ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM source_items WHERE id IN ({marks})", ids
        ).fetchall()
        return {r["id"]: r for r in rows}

    # -- persistence for editorial artifacts -------------------------------

    def save_metrics(self, m: ConceptMetrics, run_id: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO concept_metrics(concept_id,run_id,computed_at,mentions_24h,mentions_7d,"
                "mentions_30d,unique_authors_24h,unique_authors_7d,unique_publications_7d,"
                "independent_source_count,independence_ratio,baseline_daily,baseline_days_available,"
                "velocity_ratio,acceleration,engagement_sum,engagement_median,engagement_acceleration,"
                "cross_platform_count,cross_role_count,respected_source_share,novelty,paper_count,"
                "youtube_recent_count,google_search_change,insider_momentum,insider_momentum_raw,"
                "components_json,data_completeness,missing_sources,notes)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    m.concept_id, run_id, iso(m.computed_at), m.mentions_24h, m.mentions_7d,
                    m.mentions_30d, m.unique_authors_24h, m.unique_authors_7d,
                    m.unique_publications_7d, m.independent_source_count, m.independence_ratio,
                    m.baseline_daily, m.baseline_days_available, m.velocity_ratio, m.acceleration,
                    m.engagement_sum, m.engagement_median, m.engagement_acceleration,
                    m.cross_platform_count, m.cross_role_count, m.respected_source_share,
                    m.components.novelty, m.paper_count, m.youtube_recent_count,
                    m.google_search_change, m.insider_momentum, m.insider_momentum_raw,
                    _j(m.components.model_dump()), m.data_completeness, _j(m.missing_sources),
                    _j(m.notes),
                ),
            )

    def save_propagation(self, p: SignalPropagation) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO signal_propagation(concept_id,origin_source_item_id,origin_date,"
                "platform_sequence,independent_author_count,time_to_cross_platform_hours,"
                "coverage_gap,computed_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    p.concept_id, p.origin_source_item_id, iso(p.origin_date),
                    _j(p.platform_sequence), p.independent_author_count,
                    p.time_to_cross_platform_hours, p.coverage_gap,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def save_cluster(self, cl: Cluster, run_id: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO clusters(id,run_id,cluster_name,underlying_change,raw_terms,"
                "evidence,disagreements,potential_public_relevance,breadth_warning,momentum,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cl.id, run_id, cl.cluster_name, cl.underlying_change, _j(cl.raw_terms),
                    _j([e.model_dump() for e in cl.evidence]),
                    _j([d.model_dump() for d in cl.disagreements]),
                    cl.potential_public_relevance, int(cl.breadth_warning), cl.momentum,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            for cid in cl.concept_ids:
                c.execute(
                    "INSERT OR REPLACE INTO cluster_concepts(cluster_id,concept_id,weight)"
                    " VALUES(?,?,?)",
                    (cl.id, cid, 1.0),
                )

    def save_signal(self, s: Signal, run_id: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO signals(id,run_id,agent,cluster_id,short_label,claim,concept,"
                "what_is_new,why_now,actors,evidence_summary,disagreement_summary,technical_mechanism,"
                "possible_human_consequence,possible_economic_consequence,possible_societal_consequence,"
                "possible_visuals,raw_quotes,tags,claims_json,confidence,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    s.id, run_id, s.agent, s.cluster_id, s.short_label, s.claim, s.concept,
                    s.what_is_new, s.why_now, _j(s.actors), s.evidence_summary,
                    s.disagreement_summary, s.technical_mechanism, s.possible_human_consequence,
                    s.possible_economic_consequence, s.possible_societal_consequence,
                    _j(s.possible_visuals), _j(s.raw_quotes), _j(s.tags),
                    _j([c_.model_dump() for c_ in s.claims]), s.confidence,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            for sid in s.source_ids:
                c.execute(
                    "INSERT OR IGNORE INTO signal_sources(signal_id,source_item_id,role)"
                    " VALUES(?,?,?)",
                    (s.id, sid, s.agent),
                )

    def save_candidate(self, cand: CandidateVideo, run_id: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO candidates(id,run_id,cluster_id,working_title,"
                "alternative_titles,central_question,one_sentence_premise,real_current_trigger,"
                "why_now,target_viewer,viewer_problem_solved,technical_mechanism,old_mental_model,"
                "new_mental_model,technical_concepts_to_teach,stakes,human_consequence,"
                "economic_consequence,societal_consequence,human_angle,credible_disagreement,"
                "story_engine,people_orgs,evidence_available,coverage_gap,visual_opportunities,"
                "strongest_counterargument,what_we_need_to_investigate,possible_ending_insight,"
                "revision_of,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cand.id, run_id, cand.cluster_id, cand.working_title,
                    _j(cand.alternative_titles), cand.central_question, cand.one_sentence_premise,
                    cand.real_current_trigger, cand.why_now, cand.target_viewer,
                    cand.viewer_problem_solved, cand.technical_mechanism, cand.old_mental_model,
                    cand.new_mental_model, _j(cand.technical_concepts_to_teach), cand.stakes,
                    cand.human_consequence, cand.economic_consequence, cand.societal_consequence,
                    cand.human_angle, cand.credible_disagreement, cand.story_engine,
                    _j(cand.people_orgs), cand.evidence_available, cand.coverage_gap,
                    _j(cand.visual_opportunities), cand.strongest_counterargument,
                    _j(cand.what_we_need_to_investigate), cand.possible_ending_insight,
                    cand.revision_of, datetime.now(timezone.utc).isoformat(),
                ),
            )
            for sid in cand.source_ids:
                c.execute(
                    "INSERT OR IGNORE INTO candidate_sources(candidate_id,source_item_id)"
                    " VALUES(?,?)",
                    (cand.id, sid),
                )

    def save_evaluation(self, ev: Evaluation, run_id: str) -> None:
        cols = list(ev.scores.keys())
        with self.tx() as c:
            c.execute(
                "INSERT INTO evaluations(candidate_id,run_id,evaluator_model,prompt_version,"
                + ("".join(f"{k}," for k in cols))
                + "cares_before_knowing_term,interests_non_ai_person,strongest_failure_reason,"
                "viewer_understanding_delta,old_model,new_model,mean_score,scores_json,created_at)"
                " VALUES(?,?,?,?," + ("?," * len(cols)) + "?,?,?,?,?,?,?,?,?)",
                (
                    ev.candidate_id, run_id, ev.evaluator_model, ev.prompt_version,
                    *[ev.scores[k] for k in cols],
                    int(ev.cares_before_knowing_term), int(ev.interests_non_ai_person),
                    ev.strongest_failure_reason, ev.viewer_understanding_delta,
                    ev.old_model, ev.new_model, ev.mean_score(), _j(ev.scores),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def save_pairwise(self, pr: PairwiseResult, run_id: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO pairwise_results(run_id,candidate_a,candidate_b,question_key,winner,"
                "confidence,explanation,judge_model,order_swapped,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, pr.candidate_a, pr.candidate_b, pr.question_key, pr.winner,
                    pr.confidence, pr.explanation, pr.judge_model, int(pr.order_swapped),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def save_red_team(self, f: RedTeamFinding, run_id: str, judge_model: str = "") -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO red_team(candidate_id,run_id,attack_key,verdict,argument,severity,"
                "abandon_evidence,judge_model,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    f.candidate_id, run_id, f.attack_key, f.verdict, f.argument, f.severity,
                    f.abandon_evidence, judge_model, datetime.now(timezone.utc).isoformat(),
                ),
            )

    def save_human_review(self, r: HumanReview, run_id: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO human_reviews(candidate_id,run_id,reviewer,decision,favorite,"
                "personal_excitation_score,would_spend_two_weeks,notes,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    r.candidate_id, run_id, r.reviewer, r.decision, int(r.favorite),
                    r.personal_excitation_score,
                    None if r.would_spend_two_weeks is None else int(r.would_spend_two_weeks),
                    r.notes, iso(r.created_at),
                ),
            )

    def save_llm_call(self, call: LLMCall) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO llm_calls(run_id,agent,model,prompt_version,input_hash,input_tokens,"
                "output_tokens,cost_usd,latency_ms,cache_hit,retry_count,created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    call.run_id, call.agent, call.model, call.prompt_version, call.input_hash,
                    call.input_tokens, call.output_tokens, call.cost_usd, call.latency_ms,
                    int(call.cache_hit), call.retry_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def run_cost(self, run_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) usd, COALESCE(SUM(input_tokens),0) inp,"
            " COALESCE(SUM(output_tokens),0) out, COALESCE(SUM(cache_hit),0) hits"
            " FROM llm_calls WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else {}
