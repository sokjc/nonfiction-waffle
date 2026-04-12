"""Tests that prompt templates render without errors."""

from strategy_agent.prompts.evaluator import evaluator_prompt
from strategy_agent.prompts.researcher import research_prompt
from strategy_agent.prompts.rewriter import rewriter_prompt
from strategy_agent.prompts.style_guide import FULL_STYLE_GUIDE
from strategy_agent.prompts.writer import writer_prompt


def test_style_guide_assembled():
    """The full style guide should contain all major sections."""
    assert "Voice & Tone Principles" in FULL_STYLE_GUIDE
    assert "Storytelling Framework" in FULL_STYLE_GUIDE
    assert "Formatting & Structure" in FULL_STYLE_GUIDE
    assert "American English" in FULL_STYLE_GUIDE


def test_research_prompt_renders():
    messages = research_prompt.format_messages(
        brief="Test brief",
        document_type="strategy_memo",
        additional_instructions="None",
        retrieved_context="Some context here.",
    )
    assert len(messages) == 2
    assert "Test brief" in messages[1].content


def test_writer_prompt_renders():
    messages = writer_prompt.format_messages(
        brief="Test brief",
        document_type="white_paper",
        research_synthesis="Research findings here.",
        additional_instructions="Focus on APAC markets.",
    )
    assert len(messages) == 2
    assert "Research findings here." in messages[1].content
    assert "APAC markets" in messages[1].content


def test_evaluator_prompt_renders():
    messages = evaluator_prompt.format_messages(
        brief="Test brief",
        document_type="strategy_memo",
        research_synthesis="Research here.",
        current_iteration=1,
        draft="This is a draft document.",
    )
    assert len(messages) == 2
    assert "iteration 1" in messages[1].content


def test_rewriter_prompt_renders():
    messages = rewriter_prompt.format_messages(
        brief="Test brief",
        document_type="strategy_memo",
        research_synthesis="Research here.",
        current_iteration=1,
        draft="Draft text.",
        overall_score="6.5",
        storytelling_score="7.0",
        narrative_cohesion_score="6.0",
        data_integration_score="6.5",
        style_compliance_score="7.0",
        strengths="- Good opening",
        weaknesses="- Weak data integration",
        rewrite_instructions="Add more data points.",
    )
    assert len(messages) == 2
    assert "Add more data points" in messages[1].content
