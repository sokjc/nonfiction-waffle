"""Tests for the configuration layer."""

import os
from unittest.mock import patch

from strategy_agent.config import Settings


def test_defaults():
    """Settings should have sensible defaults without any env vars."""
    with patch.dict(os.environ, {}, clear=True):
        s = Settings()
    assert s.llm_model == "gemma4-31b"
    assert s.llm_base_url == "http://localhost:8000/v1"
    assert s.chunk_size == 1500
    assert s.chunk_overlap == 300
    assert s.max_rewrite_loops == 3


def test_env_override():
    """Environment variables should override defaults."""
    overrides = {
        "LLM_MODEL": "nemotron-120b",
        "LLM_BASE_URL": "http://gpu-server:9000/v1",
        "CHUNK_SIZE": "2000",
        "MAX_REWRITE_LOOPS": "5",
    }
    with patch.dict(os.environ, overrides, clear=True):
        s = Settings()
    assert s.llm_model == "nemotron-120b"
    assert s.llm_base_url == "http://gpu-server:9000/v1"
    assert s.chunk_size == 2000
    assert s.max_rewrite_loops == 5


def test_eval_model_fallback():
    """When no eval model is configured, it should fall back to the primary."""
    with patch.dict(os.environ, {}, clear=True):
        s = Settings()
    assert s.eval_model_resolved == "gemma4-31b"
    assert s.eval_base_url_resolved == s.llm_base_url


def test_eval_model_explicit():
    """When eval model is explicitly set, it should be used."""
    overrides = {
        "EVAL_MODEL": "nemotron-120b",
        "EVAL_BASE_URL": "http://big-gpu:8001/v1",
    }
    with patch.dict(os.environ, overrides, clear=True):
        s = Settings()
    assert s.eval_model_resolved == "nemotron-120b"
    assert s.eval_base_url_resolved == "http://big-gpu:8001/v1"


def test_kg_and_session_defaults():
    """New KG and session settings should have sensible defaults."""
    with patch.dict(os.environ, {}, clear=True):
        s = Settings()
    assert str(s.kg_gml_path).endswith("knowledge_graph.gml")
    assert str(s.session_db_path).endswith("sessions.db")


def test_kg_and_session_override():
    """KG and session paths should be overridable via env."""
    overrides = {
        "KNOWLEDGE_GRAPH_PATH": "/tmp/custom_kg.gml",
        "SESSION_DB_PATH": "/tmp/custom_sessions.db",
    }
    with patch.dict(os.environ, overrides, clear=True):
        s = Settings()
    assert str(s.kg_gml_path) == "/tmp/custom_kg.gml"
    assert str(s.session_db_path) == "/tmp/custom_sessions.db"
