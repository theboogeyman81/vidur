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


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_PROGRESS)
        await db.commit()


async def log_progress(session_id: str, topic: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO progress (session_id, topic) VALUES (?, ?)",
            (session_id, topic),
        )
        await db.commit()
