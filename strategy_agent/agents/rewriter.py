"""Rewriter Agent — revises drafts based on evaluator feedback.

The rewriter takes the current draft and the evaluator's structured scorecard
and produces an improved version that addresses every identified weakness
while preserving strengths.
"""

from __future__ import annotations

import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from strategy_agent.config import Settings, get_settings
from strategy_agent.memory.working_memory import WorkingMemory
from strategy_agent.models import build_writer_llm
from strategy_agent.prompts.rewriter import rewriter_prompt

logger = logging.getLogger(__name__)


class RewriterAgent:
    """Revises a draft using evaluator feedback."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._llm = llm or build_writer_llm(self._settings)
        self._chain = rewriter_prompt | self._llm | StrOutputParser()

    def run(self, memory: WorkingMemory) -> WorkingMemory:
        """Produce a revised draft addressing the latest evaluation feedback."""
        evaluation = memory.latest_evaluation
        if evaluation is None:
            raise ValueError("No evaluation to base the rewrite on")

        logger.info(
            "Rewriting draft (iteration %d → %d) based on score %s",
            memory.current_iteration,
            memory.current_iteration + 1,
            evaluation.summary,
        )

        revised = self._chain.invoke({
            "brief": memory.brief,
            "document_type": memory.document_type,
            "research_synthesis": memory.research_synthesis,
            "current_iteration": memory.current_iteration,
            "draft": memory.latest_draft,
            "overall_score": f"{evaluation.overall_score:.1f}",
            "storytelling_score": f"{evaluation.storytelling_score:.1f}",
            "narrative_cohesion_score": f"{evaluation.narrative_cohesion_score:.1f}",
            "data_integration_score": f"{evaluation.data_integration_score:.1f}",
            "style_compliance_score": f"{evaluation.style_compliance_score:.1f}",
            "strengths": "\n".join(f"- {s}" for s in evaluation.strengths),
            "weaknesses": "\n".join(f"- {w}" for w in evaluation.weaknesses),
            "rewrite_instructions": evaluation.rewrite_instructions,
        })

        memory.drafts.append(revised)
        memory.current_iteration += 1

        logger.info(
            "Rewrite complete (iteration %d, %d chars)",
            memory.current_iteration,
            len(revised),
        )
        return memory
