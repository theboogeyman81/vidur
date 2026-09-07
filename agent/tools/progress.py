from pydantic_ai import RunContext

from api.db import log_progress


async def progress_tool(ctx: RunContext[dict], topic: str) -> None:
    """Record that the student studied this topic in the current session."""
    session_id = ctx.deps.get("session_id", "unknown")
    await log_progress(session_id, topic)
