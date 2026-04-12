"""Narrative Evaluator Agent — scores drafts and generates rewrite instructions.

The evaluator uses a potentially heavier model (e.g. Nemotron-120B) to provide
a rigorous, structured assessment of each draft against the house style rubric.
Its JSON output is parsed into an ``EvaluationResult`` that drives the
accept/rewrite decision in the orchestrator.
"""

from __future__ import annotations

import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from strategy_agent.config import Settings, get_settings
from strategy_agent.memory.working_memory import EvaluationResult, WorkingMemory
from strategy_agent.models import build_eval_llm
from strategy_agent.prompts.evaluator import evaluator_prompt

logger = logging.getLogger(__name__)


def _parse_evaluation(raw: str) -> EvaluationResult:
    """Extract a JSON scorecard from the LLM response.

    Models sometimes wrap JSON in markdown code fences — we strip those.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse evaluator JSON — returning default scores")
        return EvaluationResult(
            rewrite_instructions=raw,
            weaknesses=["Evaluator output was not valid JSON — manual review needed"],
        )

    return EvaluationResult(
        storytelling_score=float(data.get("storytelling_score", 0)),
        narrative_cohesion_score=float(data.get("narrative_cohesion_score", 0)),
        data_integration_score=float(data.get("data_integration_score", 0)),
        style_compliance_score=float(data.get("style_compliance_score", 0)),
        overall_score=float(data.get("overall_score", 0)),
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
        rewrite_instructions=data.get("rewrite_instructions", ""),
        passes_threshold=bool(data.get("passes_threshold", False)),
    )


class EvaluatorAgent:
    """Scores a draft against the narrative quality rubric."""

    def __init__(
        self,
        llm: ChatOpenAI | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self._llm = llm or build_eval_llm(self._settings)
        self._chain = evaluator_prompt | self._llm | StrOutputParser()

    def run(self, memory: WorkingMemory) -> WorkingMemory:
        """Evaluate the latest draft and append the result to working memory."""
        draft = memory.latest_draft
        if not draft:
            raise ValueError("No draft to evaluate")

        logger.info("Evaluating draft (iteration %d)", memory.current_iteration)

        raw = self._chain.invoke({
            "brief": memory.brief,
            "document_type": memory.document_type,
            "research_synthesis": memory.research_synthesis,
            "current_iteration": memory.current_iteration,
            "draft": draft,
        })

        evaluation = _parse_evaluation(raw)
        memory.evaluations.append(evaluation)

        logger.info(
            "Evaluation complete: %s | Accepted: %s",
            evaluation.summary,
            evaluation.passes_threshold,
        )
        return memory
