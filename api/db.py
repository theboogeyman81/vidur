from __future__ import annotations

import json
import pathlib

import aiosqlite

DB_PATH = pathlib.Path("vidur.db")

_CREATE_PROGRESS = """
CREATE TABLE IF NOT EXISTS progress (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    topic       TEXT    NOT NULL,
    ts          DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    trace       TEXT    NOT NULL,
    ts          DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_PROGRESS)
        await db.execute(_CREATE_SESSIONS)
        await db.execute(_CREATE_TURNS)
        await db.commit()


async def log_progress(session_id: str, topic: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO progress (session_id, topic) VALUES (?, ?)",
            (session_id, topic),
        )
        await db.commit()


async def log_session(session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,))
        await db.commit()


async def log_turn(session_id: str, trace: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO turns (session_id, trace) VALUES (?, ?)",
            (session_id, json.dumps(trace)),
        )
        await db.commit()
