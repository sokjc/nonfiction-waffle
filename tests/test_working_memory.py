"""Tests for the WorkingMemory data structures."""

from strategy_agent.memory.working_memory import EvaluationResult, WorkingMemory


def test_empty_memory():
    mem = WorkingMemory(brief="Test brief")
    assert mem.latest_draft == ""
    assert mem.latest_evaluation is None
    assert mem.is_accepted is False
    assert mem.current_iteration == 0


def test_draft_tracking():
    mem = WorkingMemory(brief="Test")
    mem.drafts.append("Draft 1")
    mem.drafts.append("Draft 2")
    assert mem.latest_draft == "Draft 2"
    assert len(mem.drafts) == 2


def test_evaluation_tracking():
    mem = WorkingMemory(brief="Test")
    ev1 = EvaluationResult(overall_score=5.0, passes_threshold=False)
    ev2 = EvaluationResult(overall_score=8.5, passes_threshold=True)
    mem.evaluations.append(ev1)
    mem.evaluations.append(ev2)
    assert mem.latest_evaluation is ev2
    assert mem.is_accepted is True


def test_evaluation_summary():
    ev = EvaluationResult(
        storytelling_score=7.5,
        narrative_cohesion_score=8.0,
        data_integration_score=6.5,
        style_compliance_score=9.0,
        overall_score=7.8,
    )
    summary = ev.summary
    assert "7.8" in summary
    assert "7.5" in summary
    assert "8.0" in summary
