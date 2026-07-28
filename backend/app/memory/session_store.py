# memory/session_store.py

import sqlite3
import os
import json
from datetime import datetime
from config import DATA_DIR


class SessionStore:
    """
    Stores session-level metadata and long-term facts separately from
    the LangGraph checkpointer. The checkpointer already persists full
    conversation state (messages) per thread_id — this store is for
    things you want to query independently, like session history list,
    timestamps, or long-term user facts across threads.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(DATA_DIR / "sessions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                thread_id TEXT PRIMARY KEY,
                created_at TEXT,
                last_active_at TEXT,
                title TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                fact TEXT,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def create_or_update_session(self, thread_id: str, title: str = None):
        """Registers a session, or updates its last_active_at timestamp."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute("SELECT thread_id FROM sessions WHERE thread_id = ?", (thread_id,))
        exists = cursor.fetchone()

        if exists:
            cursor.execute(
                "UPDATE sessions SET last_active_at = ? WHERE thread_id = ?",
                (now, thread_id)
            )
        else:
            cursor.execute(
                "INSERT INTO sessions (thread_id, created_at, last_active_at, title) VALUES (?, ?, ?, ?)",
                (thread_id, now, now, title or thread_id)
            )

        conn.commit()
        conn.close()

    def delete_session(self, thread_id: str):
        conn = self._get_connection()
        conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()

    def list_sessions(self) -> list[dict]:
        """Returns all sessions, most recently active first."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT thread_id, created_at, last_active_at, title FROM sessions ORDER BY last_active_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {"thread_id": r[0], "created_at": r[1], "last_active_at": r[2], "title": r[3]}
            for r in rows
        ]

    def add_long_term_fact(self, thread_id: str, fact: str):
        """Stores a fact that should persist beyond a single conversation
        (e.g. a user preference learned during chat)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        cursor.execute(
            "INSERT INTO long_term_facts (thread_id, fact, created_at) VALUES (?, ?, ?)",
            (thread_id, fact, now)
        )

        conn.commit()
        conn.close()

    def get_long_term_facts(self, thread_id: str) -> list[str]:
        """Retrieves all long-term facts stored for a given thread."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT fact FROM long_term_facts WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [r[0] for r in rows]


_session_store_instance = None

def get_session_store() -> SessionStore:
    global _session_store_instance
    if _session_store_instance is None:
        _session_store_instance = SessionStore()
    return _session_store_instance