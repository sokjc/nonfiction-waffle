"""Tests for error handling utilities and agent error boundaries."""

import json

import pytest

from strategy_agent.agents.evaluator import _parse_evaluation
from strategy_agent.errors import LLMConnectionError, invoke_llm


# ── invoke_llm helper ────────────────────────────────────────────────────────

class _FakeChain:
    """Mock chain that raises the given exception on invoke."""

    def __init__(self, exc: Exception | None = None, result: str = "ok"):
        self._exc = exc
        self._result = result

    def invoke(self, inputs):
        if self._exc:
            raise self._exc
        return self._result


def test_invoke_llm_passes_through_on_success():
    chain = _FakeChain(result="hello world")
    result = invoke_llm(chain, {}, endpoint_url="http://localhost:8000/v1")
    assert result == "hello world"


def test_invoke_llm_catches_connection_error():
    chain = _FakeChain(exc=ConnectionError("Connection refused"))
    with pytest.raises(LLMConnectionError, match="unreachable"):
        invoke_llm(chain, {}, endpoint_url="http://localhost:8000/v1")


def test_invoke_llm_catches_timeout():
    chain = _FakeChain(exc=TimeoutError("Request timeout"))
    with pytest.raises(LLMConnectionError, match="unreachable"):
        invoke_llm(chain, {}, endpoint_url="http://localhost:9000/v1")


def test_invoke_llm_includes_endpoint_in_message():
    chain = _FakeChain(exc=OSError("Connection refused by host"))
    with pytest.raises(LLMConnectionError, match="localhost:9999"):
        invoke_llm(chain, {}, endpoint_url="http://localhost:9999/v1")


def test_invoke_llm_reraises_non_connection_errors():
    chain = _FakeChain(exc=ValueError("Bad input"))
    with pytest.raises(ValueError, match="Bad input"):
        invoke_llm(chain, {}, endpoint_url="http://localhost:8000/v1")


# ── Evaluator parse failure behavior ─────────────────────────────────────────

def test_parse_evaluation_invalid_json_sets_parse_failed():
    result = _parse_evaluation("This is not JSON at all.")
    assert result.parse_failed is True
    assert result.passes_threshold is True  # Accepts to avoid looping


def test_parse_evaluation_missing_overall_score_sets_parse_failed():
    data = {"storytelling_score": 7.0, "strengths": ["Good opening"]}
    result = _parse_evaluation(json.dumps(data))
    assert result.parse_failed is True
    assert result.passes_threshold is True


def test_parse_evaluation_valid_json_no_parse_failed():
    data = {
        "storytelling_score": 8.0,
        "narrative_cohesion_score": 7.5,
        "data_integration_score": 7.0,
        "style_compliance_score": 8.5,
        "overall_score": 7.8,
        "strengths": ["Strong opening"],
        "weaknesses": ["Weak close"],
        "rewrite_instructions": "Fix the close.",
        "passes_threshold": False,
    }
    result = _parse_evaluation(json.dumps(data))
    assert result.parse_failed is False
    assert result.overall_score == 7.8
    assert result.passes_threshold is False
