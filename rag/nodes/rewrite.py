from __future__ import annotations

import os

import google.genai as genai

from rag.types import RAGState

_REWRITE_PROMPT = """\
You are a query optimizer for a RAG system about Indian Constitutional law and governance
(NCERT textbook content).

The following search query returned no relevant chunks. Rewrite it to be more specific,
using different vocabulary or focusing on a narrower aspect of the question.

Original query: {original_query}
Current query: {query}
Rewrite attempt: {attempt}

Output only the rewritten query, nothing else."""


async def rewrite_node(state: RAGState) -> dict:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = await client.aio.models.generate_content(
        model="gemini-3.6-flash",
        contents=_REWRITE_PROMPT.format(
            original_query=state["original_query"],
            query=state["query"],
            attempt=state["rewrites"] + 1,
        ),
    )
    new_query = resp.text.strip()
    return {
        "query": new_query,
        "rewrites": state["rewrites"] + 1,
    }
