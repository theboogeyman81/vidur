from __future__ import annotations

import logging
import os

import structlog
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli

load_dotenv()

# Plugins must be imported on the main thread before cli.run_app()
from livekit.plugins import google as _google_plugin  # noqa: E402, F401
from livekit.plugins import silero  # noqa: E402

_json_logging = os.getenv("LOG_FORMAT") == "json"

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer() if _json_logging else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)


def prewarm(proc: JobProcess) -> None:
    # Load the Silero VAD model once per worker process, not per job
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    from agent.session import run_session

    await run_session(ctx, vad=ctx.proc.userdata["vad"])


async def _startup() -> None:
    from api.db import init_db

    await init_db()


if __name__ == "__main__":
    import asyncio

    asyncio.run(_startup())
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
