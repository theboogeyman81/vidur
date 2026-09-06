from __future__ import annotations

import logging
import os

import structlog
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli

load_dotenv()

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


async def entrypoint(ctx: JobContext) -> None:
    from agent.session import run_session

    await run_session(ctx)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
