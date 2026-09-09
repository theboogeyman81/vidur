from __future__ import annotations

from typing import TypedDict

from rag.retriever import Chunk  # re-export so nodes import from here


class RAGState(TypedDict):
    query: str
    original_query: str
    chunks: list[Chunk]
    passed_chunks: list[Chunk]
    rewrites: int
    bailed_out: bool
    start_time: float
    answer: str  # populated by generate_node for eval; unused by retrieve_tool
