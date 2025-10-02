from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import aiosqlite

__all__ = [
    "HistoryEntry",
    "init_db",
    "save_user_message",
    "save_assistant_message",
    "fetch_messages",
    "delete_chat_history",
]


@dataclass(slots=True)
class HistoryEntry:
    ts: float
    role: str
    sender_id: Optional[int]
    sender_username: Optional[str]
    text: str


_DB_PATH: Optional[Path] = None
_DB_LOCK = asyncio.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    chat_type TEXT,
    ts REAL NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    sender_id INTEGER,
    sender_username TEXT,
    text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages(chat_id, ts);
"""


def _ensure_initialized() -> Path:
    if _DB_PATH is None:
        raise RuntimeError("Storage not initialized. Call init_db() first.")
    return _DB_PATH


async def init_db(db_path: str) -> None:
    """Initialise the SQLite database and remember its location."""
    global _DB_PATH

    resolved = Path(db_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    async with _DB_LOCK:
        async with aiosqlite.connect(resolved) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA foreign_keys=ON;")
            await db.executescript(_SCHEMA)
            await db.commit()

        _DB_PATH = resolved


async def save_user_message(
    chat_id: int,
    chat_type: str,
    ts: float,
    sender_id: Optional[int],
    sender_username: Optional[str],
    text: str,
) -> None:
    """Persist a user-authored message."""
    await _save_message(chat_id, chat_type, ts, "user", sender_id, sender_username, text)


async def save_assistant_message(
    chat_id: int,
    chat_type: str,
    ts: float,
    text: str,
) -> None:
    """Persist an assistant-authored message."""
    await _save_message(chat_id, chat_type, ts, "assistant", None, None, text)


async def _save_message(
    chat_id: int,
    chat_type: str,
    ts: float,
    role: str,
    sender_id: Optional[int],
    sender_username: Optional[str],
    text: str,
) -> None:
    cleaned = (text or "").strip()
    if not cleaned:
        return

    db_path = _ensure_initialized()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO messages (chat_id, chat_type, ts, role, sender_id, sender_username, text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (chat_id, chat_type, ts, role, sender_id, sender_username, cleaned),
        )
        await db.commit()


async def fetch_messages(
    chat_id: int,
    since_ts: float,
    until_ts: float,
    limit: int = 500,
) -> List[HistoryEntry]:
    """Return chronological messages for a chat between the given timestamps."""
    db_path = _ensure_initialized()
    bounded_limit = max(1, min(limit, 1000))

    query = """
        SELECT ts, role, sender_id, sender_username, text
        FROM messages
        WHERE chat_id = ? AND ts >= ? AND ts <= ?
        ORDER BY ts ASC
        LIMIT ?
    """

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, (chat_id, since_ts, until_ts, bounded_limit)) as cursor:
            rows = await cursor.fetchall()

    return [
        HistoryEntry(
            ts=row["ts"],
            role=row["role"],
            sender_id=row["sender_id"],
            sender_username=row["sender_username"],
            text=row["text"],
        )
        for row in rows
    ]


async def delete_chat_history(chat_id: int) -> None:
    """Erase persisted messages for a chat."""
    db_path = _ensure_initialized()

    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        await db.commit()
