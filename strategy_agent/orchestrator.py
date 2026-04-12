"""LangGraph orchestrator — the multi-agent pipeline that turns a brief into
a polished strategy document.

The pipeline follows this flow:

    ┌──────────┐     ┌────────┐     ┌───────────┐     ┌──────────┐
    │ RESEARCH │ ──▶ │ WRITE  │ ──▶ │ EVALUATE  │ ──▶ │ REWRITE  │
    └──────────┘     └────────┘     └───────────┘     └──────┬───┘
                                         ▲                    │
                                         │   (loop until      │
                                         │    accepted or     │
                                         │    max iterations) │
                                         └────────────────────┘
                                         │
                                         ▼
                                    ┌──────────┐
                                    │ FINALIZE │
                                    └──────────┘

The graph carries ``PipelineState`` (a TypedDict wrapping ``WorkingMemory``)
through each node.  Conditional edges implement the evaluate→rewrite loop.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from strategy_agent.agents.evaluator import EvaluatorAgent
from strategy_agent.agents.researcher import ResearchAgent
from strategy_agent.agents.rewriter import RewriterAgent
from strategy_agent.agents.writer import WriterAgent
from strategy_agent.config import Settings, get_settings
from strategy_agent.memory.vector_store import CorpusStore
from strategy_agent.memory.working_memory import WorkingMemory
from strategy_agent.models import build_eval_llm, build_writer_llm

logger = logging.getLogger(__name__)


# ── Graph state ───────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """The state object that flows through every node in the graph."""

    memory: WorkingMemory
    settings: Settings


# ── Node functions ────────────────────────────────────────────────────────────
# Each receives the full state dict and returns the fields it mutates.

def research_node(state: PipelineState) -> PipelineState:
    """Run the research agent to build a synthesis from the corpus."""
    memory = state["memory"]
    settings = state["settings"]

    store = CorpusStore(settings)
    llm = build_writer_llm(settings)
    agent = ResearchAgent(llm=llm, store=store, settings=settings)

    # Use agentic mode if the corpus has documents, else retrieval
    if store.count > 0:
        agent.run(memory)
    else:
        logger.warning("Corpus is empty — research phase will note the gap")
        memory.research_synthesis = (
            "The document corpus is currently empty.  The writer should rely "
            "on the brief and general strategic reasoning.  Flag any claims "
            "that would normally require corpus evidence."
        )

    return {"memory": memory, "settings": settings}


def write_node(state: PipelineState) -> PipelineState:
    """Run the writer agent to produce a draft."""
    memory = state["memory"]
    settings = state["settings"]

    llm = build_writer_llm(settings)
    agent = WriterAgent(llm=llm, settings=settings)
    agent.run(memory)

    return {"memory": memory, "settings": settings}


def evaluate_node(state: PipelineState) -> PipelineState:
    """Run the evaluator agent to score the latest draft."""
    memory = state["memory"]
    settings = state["settings"]

    llm = build_eval_llm(settings)
    agent = EvaluatorAgent(llm=llm, settings=settings)
    agent.run(memory)

    return {"memory": memory, "settings": settings}


def rewrite_node(state: PipelineState) -> PipelineState:
    """Run the rewriter agent to revise the draft based on feedback."""
    memory = state["memory"]
    settings = state["settings"]

    llm = build_writer_llm(settings)
    agent = RewriterAgent(llm=llm, settings=settings)
    agent.run(memory)

    return {"memory": memory, "settings": settings}


def finalize_node(state: PipelineState) -> PipelineState:
    """Terminal node — log completion metadata."""
    memory = state["memory"]
    evaluation = memory.latest_evaluation

    score_info = evaluation.summary if evaluation else "no evaluation"
    logger.info(
        "Pipeline complete — %d iteration(s), final score: %s",
        memory.current_iteration,
        score_info,
    )

    return state


# ── Conditional edge: should we rewrite or accept? ────────────────────────────

def should_rewrite(state: PipelineState) -> str:
    """Return the next node name based on evaluation outcome."""
    memory = state["memory"]
    settings = state["settings"]
    evaluation = memory.latest_evaluation

    if evaluation is None:
        return "finalize"

    if evaluation.passes_threshold:
        logger.info("Draft accepted at iteration %d", memory.current_iteration)
        return "finalize"

    if memory.current_iteration >= settings.max_rewrite_loops:
        logger.warning(
            "Max rewrite iterations (%d) reached — accepting current draft",
            settings.max_rewrite_loops,
        )
        return "finalize"

    logger.info(
        "Draft scored %.1f/10 — sending to rewriter (iteration %d/%d)",
        evaluation.overall_score,
        memory.current_iteration,
        settings.max_rewrite_loops,
    )
    return "rewrite"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_pipeline(settings: Settings | None = None) -> StateGraph:
    """Construct and compile the LangGraph strategy pipeline.

    Returns a compiled graph that can be invoked with::

        result = graph.invoke({
            "memory": WorkingMemory(brief="...", document_type="..."),
            "settings": settings,
        })
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("finalize", finalize_node)

    # Linear flow: research → write → evaluate
    graph.set_entry_point("research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "evaluate")

    # Conditional: evaluate → rewrite (loop) or → finalize (accept)
    graph.add_conditional_edges(
        "evaluate",
        should_rewrite,
        {
            "rewrite": "rewrite",
            "finalize": "finalize",
        },
    )

    # After rewrite, re-evaluate
    graph.add_edge("rewrite", "evaluate")

    # Finalize is terminal
    graph.add_edge("finalize", END)

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────────

def run_pipeline(
    brief: str,
    document_type: str = "strategy_memo",
    additional_instructions: str = "",
    settings: Settings | None = None,
) -> WorkingMemory:
    """One-call entry point: brief in, polished document out.

    Returns the final ``WorkingMemory`` containing all drafts, evaluations,
    and the accepted document in ``memory.latest_draft``.
    """
    s = settings or get_settings()
    memory = WorkingMemory(
        brief=brief,
        document_type=document_type,
        additional_instructions=additional_instructions,
    )

    pipeline = build_pipeline(s)
    result = pipeline.invoke({"memory": memory, "settings": s})

    return result["memory"]
