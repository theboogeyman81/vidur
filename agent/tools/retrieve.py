import time

from pydantic_ai import RunContext


async def retrieve_tool(ctx: RunContext[dict], query: str) -> str:
    """Look up relevant NCERT content for the student's question."""
    t0 = time.perf_counter()
    try:
        from rag.retriever import retrieve

        chunks = await retrieve(query, top_k=5)
        # TODO 2.10: log rag_total_ms to Langfuse span
        _ = round((time.perf_counter() - t0) * 1000, 1)
        return "\n\n".join(c.text for c in chunks)
    except ImportError:
        return "[RAG retriever not yet available]"
