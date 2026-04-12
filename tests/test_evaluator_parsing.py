"""Tests for the evaluator's JSON parsing logic."""

import json

from strategy_agent.agents.evaluator import _parse_evaluation


def test_parse_clean_json():
    data = {
        "storytelling_score": 7.5,
        "narrative_cohesion_score": 8.0,
        "data_integration_score": 6.5,
        "style_compliance_score": 9.0,
        "overall_score": 7.8,
        "strengths": ["Strong opening hook", "Good data integration"],
        "weaknesses": ["Weak closing paragraph"],
        "rewrite_instructions": "Strengthen the closing section.",
        "passes_threshold": False,
    }
    result = _parse_evaluation(json.dumps(data))
    assert result.storytelling_score == 7.5
    assert result.overall_score == 7.8
    assert len(result.strengths) == 2
    assert result.passes_threshold is False


def test_parse_json_with_code_fences():
    data = {
        "storytelling_score": 9.0,
        "narrative_cohesion_score": 8.5,
        "data_integration_score": 8.0,
        "style_compliance_score": 8.5,
        "overall_score": 8.5,
        "strengths": ["Excellent narrative arc"],
        "weaknesses": [],
        "rewrite_instructions": "",
        "passes_threshold": True,
    }
    raw = f"```json\n{json.dumps(data)}\n```"
    result = _parse_evaluation(raw)
    assert result.overall_score == 8.5
    assert result.passes_threshold is True


def test_parse_invalid_json_fallback():
    raw = "This is not JSON at all, just the evaluator rambling."
    result = _parse_evaluation(raw)
    assert result.overall_score == 0.0
    assert "not valid JSON" in result.weaknesses[0]
    assert result.rewrite_instructions == raw
