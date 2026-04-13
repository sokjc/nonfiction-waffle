"""Tests for the configuration layer."""

import os
from unittest.mock import patch

from strategy_agent.config import Settings


def test_defaults():
    """Settings should have sensible defaults without any env vars."""
    with patch.dict(os.environ, {}, clear=True):
        s = Settings()
    # 3-model defaults
    assert s.writer_model == "gpt-oss"
    assert s.agent_model == "gemma4-31b"
    assert s.eval_model == "nemotron-120b"
    # Embedding default
    assert s.embedding_model == "nomic-ai/nomic-embed-text-v1.5"
    # Common defaults
    assert s.chunk_size == 1500
    assert s.chunk_overlap == 300
    assert s.max_rewrite_loops == 3
    assert s.context_stuffing_docs == 5


def test_env_override():
    """Environment variables should override defaults."""
    overrides = {
        "WRITER_MODEL": "llama-3-70b",
        "AGENT_MODEL": "qwen-72b",
        "AGENT_BASE_URL": "http://gpu-server:9000/v1",
        "CHUNK_SIZE": "2000",
        "MAX_REWRITE_LOOPS": "5",
    }
    with patch.dict(os.environ, overrides, clear=True):
        s = Settings()
    assert s.writer_model == "llama-3-70b"
    assert s.agent_model == "qwen-72b"
    assert s.agent_base_url == "http://gpu-server:9000/v1"
    assert s.chunk_size == 2000
    assert s.max_rewrite_loops == 5


def test_three_model_routing():
    """Each model role should be independently configurable."""
    overrides = {
        "WRITER_BASE_URL": "http://writer:8000/v1",
        "WRITER_MODEL": "gpt-oss",
        "AGENT_BASE_URL": "http://agent:8001/v1",
        "AGENT_MODEL": "gemma4-31b",
        "EVAL_BASE_URL": "http://eval:8002/v1",
        "EVAL_MODEL": "nemotron-120b",
    }
    with patch.dict(os.environ, overrides, clear=True):
        s = Settings()
    assert s.writer_base_url == "http://writer:8000/v1"
    assert s.agent_base_url == "http://agent:8001/v1"
    assert s.eval_base_url == "http://eval:8002/v1"
    assert s.writer_model != s.agent_model != s.eval_model


def test_kg_and_session_defaults():
    """KG and session settings should have sensible defaults."""
    with patch.dict(os.environ, {}, clear=True):
        s = Settings()
    assert str(s.kg_gml_path).endswith("knowledge_graph.gml")
    assert str(s.session_db_path).endswith("sessions.db")
    assert str(s.index_persist_dir).endswith("index")


def test_kg_and_session_override():
    """KG, session, and index paths should be overridable via env."""
    overrides = {
        "KNOWLEDGE_GRAPH_PATH": "/tmp/custom_kg.gml",
        "SESSION_DB_PATH": "/tmp/custom_sessions.db",
        "INDEX_PERSIST_DIR": "/tmp/custom_index",
    }
    with patch.dict(os.environ, overrides, clear=True):
        s = Settings()
    assert str(s.kg_gml_path) == "/tmp/custom_kg.gml"
    assert str(s.session_db_path) == "/tmp/custom_sessions.db"
    assert str(s.index_persist_dir) == "/tmp/custom_index"
