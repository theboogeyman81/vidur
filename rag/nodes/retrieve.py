from __future__ import annotations

import time

from rag.retriever import retrieve
from rag.types import RAGState

_BAIL_MS = 2400


async def retrieve_node(state: RAGState) -> dict:
    elapsed_ms = (time.perf_counter() - state["start_time"]) * 1000
    if elapsed_ms > _BAIL_MS:
        return {"bailed_out": True}
    chunks = await retrieve(state["query"], top_k=5)
    return {"chunks": chunks}
