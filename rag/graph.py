from __future__ import annotations

import asyncio
import time
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from rag.nodes.generate import generate_node
from rag.nodes.grade import grade_node
from rag.nodes.retrieve import retrieve_node
from rag.nodes.rewrite import rewrite_node
from rag.retriever import Chunk
from rag.types import RAGState


def _route(state: RAGState) -> str:
    if state["bailed_out"]:
        return "generate"
    if state["passed_chunks"]:
        return "generate"
    if state["rewrites"] >= 2:
        return "generate"
    return "rewrite"


def _build_graph() -> object:
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)

    return g.compile(checkpointer=MemorySaver())


_graph = _build_graph()


async def run_graph(query: str) -> tuple[list[Chunk], dict]:
    """Public entry point for the retrieve tool and eval runner.

    Returns (chunks, metadata). The voice agent speaks its own answer;
    metadata is logged to Langfuse by the caller.
    """
    t0 = time.perf_counter()
    thread_id = str(uuid.uuid4())
    initial_state: RAGState = {
        "query": query,
        "original_query": query,
        "chunks": [],
        "passed_chunks": [],
        "rewrites": 0,
        "bailed_out": False,
        "start_time": t0,
        "answer": "",
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final = await asyncio.wait_for(
            _graph.ainvoke(initial_state, config=config),
            timeout=2.5,
        )
    except TimeoutError:
        snapshot = await _graph.aget_state(config)
        final = dict(snapshot.values) if snapshot and snapshot.values else dict(initial_state)
        final["bailed_out"] = True

    chunks: list[Chunk] = final.get("passed_chunks") or final.get("chunks", [])
    meta = {
        "rag_total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "rag_rewrites": final.get("rewrites", 0),
        "rag_bailed_out": final.get("bailed_out", False),
        "chunks_retrieved": len(final.get("chunks", [])),
        "chunks_passed_grading": len(final.get("passed_chunks", [])),
    }
    return chunks, meta
