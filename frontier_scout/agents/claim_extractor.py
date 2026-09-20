"""Claim extraction: read each article once, keep what it argued.

This is the semantic layer. Without it the system stores raw text and counts
but no comprehension, so every run rebuilds its understanding from prose and
then throws it away.

Design notes:

- Runs ONCE per article, at ingestion or via backfill, on the cheapest model.
  The whole point is that the expensive reasoning downstream reads structure
  instead of prose.
- Persistence is per-article and immediate, so a backfill over thousands of
  articles is resumable and a crash costs one article rather than the run.
- Concepts are resolved through the registry, not invented, so claims join
  cleanly onto the existing trend tables.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import load_prompt, prompt_version
from ..db import Database
from ..llm.base import LLMRequest
from ..llm.runner import AgentRunner
from ..trends.registry import ConceptRegistry

log = logging.getLogger(__name__)

#: Articles longer than this are truncated. Claims cluster near the top of a
#: piece, and paying for 20k characters of a link roundup is waste.
MAX_ARTICLE_CHARS = 9000
MIN_ARTICLE_CHARS = 400


class ExtractedClaim(BaseModel):
    proposition: str
    stance: Literal["asserts", "disputes", "questions", "reports"]
    hedging: Literal["confident", "hedged", "speculative"] = "confident"
    evidence_type: Literal[
        "benchmark", "measurement", "anecdote", "argument", "announcement", "none"
    ] = "argument"
    subject: str = ""
    quote: str | None = None
    is_surprising: bool = False
    is_self_correction: bool = False
    confidence: float = 0.6


class ExtractedPosition(BaseModel):
    concept: str = Field(description="Must be one of the tracked concepts listed")
    stance: Literal["advocating", "skeptical", "neutral", "reporting"]
    rationale: str = ""
    confidence: float = 0.6


class ExtractedEntity(BaseModel):
    name: str
    entity_type: Literal["person", "org", "model", "paper", "product"] = "org"
    role: str = ""


class ClaimExtractionOutput(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)
    positions: list[ExtractedPosition] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)


class ClaimExtractor:
    """Extracts and persists the claim layer, one article at a time."""

    prompt_name = "claim_extractor"
    role = "extractor"

    def __init__(self, db: Database, runner: AgentRunner, registry: ConceptRegistry | None = None):
        self.db = db
        self.runner = runner
        self.registry = registry or ConceptRegistry(db)
        self._concept_by_label = {
            self.registry.label(cid).lower(): cid for cid in self.registry.concept_ids
        }

    # -- selection ---------------------------------------------------------

    def pending(
        self,
        limit: int | None = None,
        min_chars: int = MIN_ARTICLE_CHARS,
        source_types: list[str] | None = None,
    ) -> list[Any]:
        """Articles with real text that have not been extracted yet.

        Selective on purpose. Extraction costs the same per article whether the
        article is a 15,000-character essay or a two-line forum comment, and
        the essay yields far more. `min_chars` and `source_types` let a backfill
        spend where the arguments actually are.

        Ordered newest first, so an interrupted backfill has already covered the
        most editorially relevant articles.
        """
        # 'ok' and 'empty' are settled outcomes. 'failed' is not: a timeout or
        # a token overrun should be retried on the next pass, not written off.
        sql = (
            "SELECT s.id, s.title, s.text, s.source_name, s.published_at,"
            " s.author_id, s.source_type, s.primary_or_secondary"
            " FROM source_items s"
            " LEFT JOIN claim_extractions e ON e.source_item_id = s.id"
            " WHERE (e.source_item_id IS NULL OR e.status = 'failed')"
            "   AND LENGTH(COALESCE(s.text,'')) >= ?"
        )
        params: list[Any] = [min_chars]
        if source_types:
            marks = ",".join("?" * len(source_types))
            sql += f" AND s.source_type IN ({marks})"
            params.extend(source_types)
        sql += " ORDER BY s.published_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.db.conn.execute(sql, params).fetchall()

    def coverage(self) -> dict[str, int]:
        row = self.db.conn.execute(
            "SELECT"
            " (SELECT COUNT(*) FROM source_items WHERE LENGTH(COALESCE(text,'')) >= ?) eligible,"
            " (SELECT COUNT(*) FROM claim_extractions) extracted,"
            " (SELECT COUNT(*) FROM article_claims) claims,"
            " (SELECT COUNT(*) FROM article_positions) positions,"
            " (SELECT COUNT(*) FROM article_entities) entities",
            (MIN_ARTICLE_CHARS,),
        ).fetchone()
        return dict(row)

    # -- extraction --------------------------------------------------------

    async def extract_one(self, row: Any) -> ClaimExtractionOutput | None:
        text = (row["text"] or "")[:MAX_ARTICLE_CHARS]
        tracked = ", ".join(sorted(self._concept_by_label))

        task = (
            f"ARTICLE\n"
            f"publication: {row['source_name']}\n"
            f"date: {str(row['published_at'])[:10]}\n"
            f"title: {row['title'] or '(untitled)'}\n"
            f"type: {row['source_type']}"
            + (
                "\nNOTE: this is a PRIMARY source -- the organisation's own claims "
                "about itself. Record its capability statements as stance=reports, "
                "evidence_type=announcement."
                if row["primary_or_secondary"] == "PRIMARY"
                else ""
            )
            + f"\n\nBODY\n{text}\n\n"
            f"TRACKED CONCEPTS (use these exact labels for positions, or omit):\n{tracked}\n\n"
            "Extract the claims, positions and entities as json."
        )

        request = LLMRequest(
            agent="claim_extractor",
            instructions=load_prompt(self.prompt_name),
            task=task,
            schema=ClaimExtractionOutput,
            prompt_version=prompt_version(self.prompt_name),
            effort="low",
            # Dense essays produce twenty-plus claims. At 8000 the model ran
            # out of room mid-JSON and the whole article was lost -- 16 of 620
            # on the first backfill. Headroom is far cheaper than re-reading.
            max_tokens=16000,
            temperature=0.0,
            metadata={"role": self.role, "item": row["id"]},
        )
        return await self.runner.run(request)

    def persist(self, row: Any, output: ClaimExtractionOutput | None) -> tuple[int, int]:
        """Write one article's extraction. Committed immediately, so a long
        backfill is resumable and a crash costs one article."""
        now = datetime.now(timezone.utc).isoformat()
        model = self.runner.provider.model_for(self.role)
        version = prompt_version(self.prompt_name)

        if output is None:
            with self.db.tx() as c:
                c.execute(
                    "INSERT OR REPLACE INTO claim_extractions"
                    "(source_item_id,extracted_at,model,prompt_version,n_claims,"
                    "n_positions,status,error) VALUES(?,?,?,?,0,0,'failed',?)",
                    (row["id"], now, model, version, "extraction returned nothing"),
                )
            return 0, 0

        with self.db.tx() as c:
            for claim in output.claims:
                concept_id = self._resolve(claim.subject, claim.proposition)
                c.execute(
                    "INSERT INTO article_claims(source_item_id,proposition,stance,hedging,"
                    "evidence_type,subject,concept_id,quote,is_surprising,is_self_correction,"
                    "confidence,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row["id"], claim.proposition, claim.stance, claim.hedging,
                        claim.evidence_type, claim.subject.lower()[:120], concept_id,
                        claim.quote, int(claim.is_surprising),
                        int(claim.is_self_correction), claim.confidence, now,
                    ),
                )
            for position in output.positions:
                concept_id = self._concept_by_label.get(position.concept.strip().lower())
                if concept_id is None:
                    continue  # model named a concept we do not track; drop it
                c.execute(
                    "INSERT OR REPLACE INTO article_positions(source_item_id,concept_id,"
                    "author_id,stance,rationale,confidence,observed_at)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (
                        row["id"], concept_id, row["author_id"], position.stance,
                        position.rationale, position.confidence, row["published_at"],
                    ),
                )
            for entity in output.entities:
                c.execute(
                    "INSERT OR IGNORE INTO article_entities(source_item_id,name,entity_type,role)"
                    " VALUES(?,?,?,?)",
                    (row["id"], entity.name.strip()[:160], entity.entity_type, entity.role),
                )
            c.execute(
                "INSERT OR REPLACE INTO claim_extractions(source_item_id,extracted_at,model,"
                "prompt_version,n_claims,n_positions,status,error) VALUES(?,?,?,?,?,?,?,NULL)",
                (
                    row["id"], now, model, version, len(output.claims),
                    len(output.positions), "ok" if output.claims else "empty",
                ),
            )
        return len(output.claims), len(output.positions)

    def _resolve(self, subject: str, proposition: str) -> int | None:
        """Attach a claim to a tracked concept where the registry recognises it."""
        hits = self.registry.find_mentions(f"{subject} {proposition}")
        return hits[0][0] if hits else None
