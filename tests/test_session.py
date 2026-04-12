"""Tests for the session management layer."""

import os
import tempfile
from unittest.mock import patch

from strategy_agent.config import Settings
from strategy_agent.session import SessionManager


def _make_settings(tmp_path: str) -> Settings:
    db_path = os.path.join(tmp_path, "test_sessions.db")
    with patch.dict(os.environ, {"SESSION_DB_PATH": db_path}, clear=True):
        return Settings()


def test_create_session():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        sm = SessionManager(settings)

        thread_id = sm.create_session("Test Session")
        assert isinstance(thread_id, str)
        assert len(thread_id) == 12
        sm.close()


def test_list_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        sm = SessionManager(settings)

        sm.create_session("Session A")
        sm.create_session("Session B")

        sessions = sm.list_sessions()
        assert len(sessions) == 2
        titles = [s["title"] for s in sessions]
        assert "Session A" in titles
        assert "Session B" in titles
        sm.close()


def test_session_exists():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        sm = SessionManager(settings)

        thread_id = sm.create_session("Exists Test")
        assert sm.session_exists(thread_id)
        assert not sm.session_exists("nonexistent")
        sm.close()


def test_touch_session():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        sm = SessionManager(settings)

        thread_id = sm.create_session("Touch Test")
        sessions_before = sm.list_sessions()
        ts_before = sessions_before[0]["updated_at"]

        sm.touch_session(thread_id)
        sessions_after = sm.list_sessions()
        ts_after = sessions_after[0]["updated_at"]

        assert ts_after >= ts_before
        sm.close()


def test_get_config():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        sm = SessionManager(settings)

        thread_id = sm.create_session()
        config = sm.get_config(thread_id)
        assert config == {"configurable": {"thread_id": thread_id}}
        sm.close()


def test_checkpointer_available():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _make_settings(tmp)
        sm = SessionManager(settings)

        checkpointer = sm.checkpointer
        assert checkpointer is not None
        sm.close()
