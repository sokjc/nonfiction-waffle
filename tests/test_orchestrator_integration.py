"""Integration tests for the LangGraph orchestrator pipeline.

Uses a mock LLM that returns canned responses so we can test the full
research → write → evaluate → rewrite flow without a running model server.
"""

import json
import os
from unittest.mock import patch

from strategy_agent.config import Settings
from strategy_agent.memory.working_memory import WorkingMemory
from strategy_agent.orchestrator import PipelineState, should_rewrite


def _make_settings() -> Settings:
    with patch.dict(os.environ, {}, clear=True):
        return Settings()


# ── should_rewrite conditional edge ──────────────────────────────────────────

def test_should_rewrite_accepts_passing_score():
    """When the evaluation passes threshold, should route to finalize."""
    from strategy_agent.memory.working_memory import EvaluationResult

    mem = WorkingMemory(brief="test")
    mem.evaluations.append(EvaluationResult(overall_score=8.5, passes_threshold=True))
    mem.current_iteration = 1

    state: PipelineState = {"memory": mem, "settings": _make_settings()}
    assert should_rewrite(state) == "finalize"


def test_should_rewrite_rejects_failing_score():
    """When the evaluation fails threshold and iterations remain, should route to rewrite."""
    from strategy_agent.memory.working_memory import EvaluationResult

    mem = WorkingMemory(brief="test")
    mem.evaluations.append(EvaluationResult(overall_score=5.0, passes_threshold=False))
    mem.current_iteration = 1

    settings = _make_settings()
    state: PipelineState = {"memory": mem, "settings": settings}
    assert should_rewrite(state) == "rewrite"


def test_should_rewrite_respects_max_iterations():
    """When max iterations reached, should route to finalize even with failing score."""
    from strategy_agent.memory.working_memory import EvaluationResult

    settings = _make_settings()
    mem = WorkingMemory(brief="test")
    mem.evaluations.append(EvaluationResult(overall_score=3.0, passes_threshold=False))
    mem.current_iteration = settings.max_rewrite_loops  # At the limit

    state: PipelineState = {"memory": mem, "settings": settings}
    assert should_rewrite(state) == "finalize"


def test_should_rewrite_handles_no_evaluation():
    """When no evaluation exists, should route to finalize."""
    mem = WorkingMemory(brief="test")
    state: PipelineState = {"memory": mem, "settings": _make_settings()}
    assert should_rewrite(state) == "finalize"


def test_should_rewrite_parse_failed_accepts():
    """When evaluation parse failed, passes_threshold=True so we finalize."""
    from strategy_agent.memory.working_memory import EvaluationResult

    mem = WorkingMemory(brief="test")
    mem.evaluations.append(
        EvaluationResult(overall_score=0.0, passes_threshold=True, parse_failed=True)
    )
    mem.current_iteration = 1

    state: PipelineState = {"memory": mem, "settings": _make_settings()}
    assert should_rewrite(state) == "finalize"
