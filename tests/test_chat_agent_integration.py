"""Integration tests for the chat agent construction and tool binding."""

import os
import tempfile
from unittest.mock import patch

from strategy_agent.agents.chat_agent import _build_generate_tool
from strategy_agent.config import Settings
from strategy_agent.session import SessionManager


def _make_settings(tmp_path: str) -> Settings:
    overrides = {
        "SESSION_DB_PATH": os.path.join(tmp_path, "test_sessions.db"),
        "KNOWLEDGE_GRAPH_PATH": os.path.join(tmp_path, "test_kg.gml"),
        "CHROMA_PERSIST_DIR": os.path.join(tmp_path, "vectorstore"),
    }
    with patch.dict(os.environ, overrides, clear=True):
        return Settings()


def test_generate_document_tool_has_correct_name():
    """The generate_document tool should have the right name and docstring."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        tool = _build_generate_tool(settings)
        assert tool.name == "generate_document"
        assert "strategy" in tool.description.lower() or "document" in tool.description.lower()


def test_generate_document_tool_accepts_brief_arg():
    """The tool should accept 'brief' and 'document_type' arguments."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        tool = _build_generate_tool(settings)
        # Check that the tool schema includes the expected params
        schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
        props = schema.get("properties", {})
        assert "brief" in props
        assert "document_type" in props


def test_session_manager_provides_checkpointer():
    """The session manager should produce a valid checkpointer for the chat agent."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        sm = SessionManager(settings)
        assert sm.checkpointer is not None

        thread_id = sm.create_session("Test Chat")
        config = sm.get_config(thread_id)
        assert config["configurable"]["thread_id"] == thread_id
        sm.close()
