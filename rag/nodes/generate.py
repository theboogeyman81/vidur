from __future__ import annotations

import os

import google.genai as genai

from rag.types import RAGState

_GEN_PROMPT = """\
Using only the following context, answer the question in 1-3 sentences.
If the context does not contain the answer, say "I don't know based on the provided context."

Context:
{context}

Question: {question}"""


async def generate_node(state: RAGState) -> dict:
    chunks = state["passed_chunks"] or state["chunks"]
    if not chunks:
        return {"answer": "[No relevant context found]"}
    context = "\n\n".join(c.text for c in chunks)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = await client.aio.models.generate_content(
        model="gemini-3.6-flash",
        contents=_GEN_PROMPT.format(context=context, question=state["original_query"]),
    )
    return {"answer": resp.text.strip()}
