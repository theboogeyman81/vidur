from __future__ import annotations

import asyncio
import os

import google.genai as genai

from rag.retriever import Chunk
from rag.types import RAGState

_GRADE_PROMPT = """\
You are a relevance grader. Answer YES if the chunk contains information that helps answer
the question, NO otherwise. Output only YES or NO.

Question: {question}
Chunk: {chunk}"""


async def _grade_one(
    client: genai.Client, question: str, chunk: Chunk
) -> tuple[Chunk, bool]:
    resp = await client.aio.models.generate_content(
        model="gemini-3.6-flash",
        contents=_GRADE_PROMPT.format(question=question, chunk=chunk.text),
    )
    verdict = resp.text.strip().upper()
    return chunk, verdict == "YES"


async def grade_node(state: RAGState) -> dict:
    if not state["chunks"]:
        return {"passed_chunks": []}
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results = await asyncio.gather(
        *[_grade_one(client, state["query"], c) for c in state["chunks"]]
    )
    passed = [chunk for chunk, ok in results if ok]
    return {"passed_chunks": passed}
