"""Agent base.

An agent is a prompt, an output schema, and a role that decides which model and
effort level it gets. Keeping them this thin is what makes them testable: each
one is small enough to read, re-prompt and diff in isolation, which a single
giant prompt never is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ..config import load_prompt, prompt_version
from ..db import Database
from ..llm.base import LLMRequest
from ..llm.runner import AgentRunner
from .dossier import ConceptGroup, build_dossier, source_ids_for

T = TypeVar("T", bound=BaseModel)

#: Prepended to every agent prompt. The provenance rule is repeated here rather
#: than trusted to each prompt file, because it is the one rule that must never
#: be accidentally dropped when a prompt is edited.
PROVENANCE_RULES = """
# NON-NEGOTIABLE RULES

1. PROVENANCE. Every claim you make is either:
   - kind "SOURCE_FACT": something actually stated in the source material.
     It MUST cite one or more source_ids that appear in [brackets] above.
   - kind "INFERENCE": your own reasoning. It cites nothing.
   Never present your interpretation as something a source said.

2. NEVER INVENT A SOURCE ID. Only use IDs that literally appear in the source
   material. A fabricated ID is detected and your answer is rejected.

3. PREFER "UNKNOWN" TO FABRICATION. If the evidence does not support an answer,
   say so. An empty list is a valid answer. A confident guess is not.

4. PRIMARY SOURCES ARE CLAIMS. A company blog is evidence of what that company
   SAYS, never proof that it is true. Write "OpenAI says X", not "X".

5. RESPECT THE MEASUREMENTS. The trend statistics above are computed from real
   observations. Do not contradict them from intuition. If velocity says NOT
   COMPUTED, do not assert that something is accelerating.

6. THE AUDIENCE is an intelligent non-specialist: curious, educated, not living
   on AI Twitter, unable to read every paper. They want to understand well
   enough to form their own opinion.

Return ONLY valid JSON matching the schema.
"""


class Agent(ABC, Generic[T]):
    """One narrow question, asked of real source material."""

    #: Prompt file stem in frontier_scout/prompts/.
    prompt_name: str = ""
    #: Pipeline role: picks the model and effort level.
    role: str = "scout"
    #: Output schema.
    schema: type[BaseModel]

    def __init__(self, db: Database, runner: AgentRunner):
        self.db = db
        self.runner = runner

    @property
    def name(self) -> str:
        return self.prompt_name or self.__class__.__name__

    def instructions(self) -> str:
        return load_prompt(self.prompt_name) + PROVENANCE_RULES

    @abstractmethod
    def task(self, group: ConceptGroup, **kwargs: Any) -> str:
        """The specific question for this group."""

    async def run(self, group: ConceptGroup, **kwargs: Any) -> T | None:
        dossier = build_dossier(self.db, group)
        allowed = source_ids_for(self.db, group)
        request = LLMRequest(
            agent=self.name,
            instructions=self.instructions(),
            task=self.task(group, **kwargs),
            schema=self.schema,
            shared_context=dossier,
            prompt_version=prompt_version(self.prompt_name),
            effort=self.runner.provider.effort_for(self.role)
            if hasattr(self.runner.provider, "effort_for")
            else "medium",
            metadata={"role": self.role, "group": group.key},
        )
        return await self.runner.run(request, allowed_source_ids=allowed)
