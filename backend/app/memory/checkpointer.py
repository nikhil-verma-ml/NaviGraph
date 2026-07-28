# memory/checkpointer.py

from langgraph.checkpoint.sqlite import SqliteSaver
from config import DATA_DIR
import sqlite3
import os


def get_checkpointer(db_path: str = None):
    """Returns a SQLite-based checkpointer so conversation state persists
    across requests, keyed by thread_id."""
    if db_path is None:
        db_path = str(DATA_DIR / "checkpoints.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)
