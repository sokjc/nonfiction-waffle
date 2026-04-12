"""Session management — SqliteSaver-backed cross-session persistence.

Each conversation is identified by a ``thread_id``.  The ``SessionManager``
owns the SQLite connection and exposes helpers for listing, creating, and
resuming sessions so the CLI REPL and (future) API server can share the
same session store.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from strategy_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages persistent chat sessions backed by SQLite."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._db_path = self._settings.session_db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)

        try:
            # Set up the LangGraph checkpoint tables
            self._checkpointer = SqliteSaver(self._conn)
            self._checkpointer.setup()

            # Set up our own sessions metadata table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    thread_id   TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
            """)
            self._conn.commit()
        except Exception:
            self._conn.close()
            raise

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def checkpointer(self) -> SqliteSaver:
        return self._checkpointer

    def create_session(self, title: str | None = None) -> str:
        """Create a new session and return its thread_id."""
        thread_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        display_title = title or f"Session {thread_id[:8]}"

        self._conn.execute(
            "INSERT INTO sessions (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (thread_id, display_title, now, now),
        )
        self._conn.commit()
        logger.info("Created session %s (%s)", thread_id, display_title)
        return thread_id

    def touch_session(self, thread_id: str) -> None:
        """Update the session's last-activity timestamp."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE thread_id = ?",
            (now, thread_id),
        )
        self._conn.commit()

    def list_sessions(self) -> list[dict]:
        """Return all sessions, most recently active first."""
        cursor = self._conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        )
        return [
            {
                "thread_id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in cursor.fetchall()
        ]

    def session_exists(self, thread_id: str) -> bool:
        cursor = self._conn.execute(
            "SELECT 1 FROM sessions WHERE thread_id = ?", (thread_id,)
        )
        return cursor.fetchone() is not None

    def get_config(self, thread_id: str) -> dict:
        """Return the LangGraph config dict for a given thread."""
        return {"configurable": {"thread_id": thread_id}}

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
