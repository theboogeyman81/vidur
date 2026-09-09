from __future__ import annotations

import time

from pydantic_ai import RunContext


async def retrieve_tool(ctx: RunContext[dict], query: str) -> str:
    """Look up relevant NCERT content for the student's question."""
    t0 = time.perf_counter()
    try:
        from rag.graph import run_graph

        chunks, meta = await run_graph(query)

        trace_ref = ctx.deps.get("active_trace", {}).get("ref")
        if trace_ref:
            trace_ref.span(name="rag", output=meta)

        if not chunks:
            return "[No relevant content found for this query]"
        return "\n\n".join(c.text for c in chunks)
    except Exception as e:
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        trace_ref = ctx.deps.get("active_trace", {}).get("ref")
        if trace_ref:
            trace_ref.span(name="rag", output={"error": str(e), "rag_total_ms": elapsed})
        return f"[RAG unavailable: {type(e).__name__}]"
