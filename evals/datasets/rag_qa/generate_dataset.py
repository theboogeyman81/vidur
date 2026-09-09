"""One-shot script to draft the RAG eval dataset from ingested NCERT content.

Writes dataset_draft.jsonl (~60 candidates).
Review it, delete bad rows, rename to dataset.jsonl.
Never overwrites dataset.jsonl.

Usage:
    uv run python -m evals.datasets.rag_qa.generate_dataset
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import random

import google.genai as genai
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION = "ncert"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
_TARGET = 60  # draft candidates to generate

_QUESTION_PROMPT = """\
You are creating a factoid QA dataset from NCERT textbook content about Indian governance
and the Constitution.

Given the following text chunk, generate:
1. One specific question whose answer is directly and completely contained in the chunk.
   The question should be answerable from this chunk alone.
2. A one-sentence ground-truth answer derived strictly from the chunk.

Chunk:
{chunk}

Respond in this exact JSON format (no markdown, no extra text):
{{"question": "...", "ground_truth_answer": "..."}}"""

_OUT = pathlib.Path(__file__).parent / "dataset_draft.jsonl"
_FINAL = pathlib.Path(__file__).parent / "dataset.jsonl"


def _fetch_chunks(limit: int) -> list[str]:
    """Scroll Qdrant directly — no Voyage embeddings needed for dataset generation."""
    qc = QdrantClient(url=QDRANT_URL)
    records, _ = qc.scroll(
        collection_name=COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    texts = [r.payload["text"] for r in records if r.payload and r.payload.get("text")]
    random.shuffle(texts)
    return texts


async def _generate_qa(client: genai.Client, chunk_text: str, idx: int) -> dict | None:
    try:
        resp = await client.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=_QUESTION_PROMPT.format(chunk=chunk_text[:1200]),
        )
        raw = resp.text.strip()
        # strip markdown fences if Gemini wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return {
            "id": f"q{idx:03d}",
            "question": parsed["question"],
            "ground_truth_chunks": [chunk_text],
            "ground_truth_answer": parsed["ground_truth_answer"],
        }
    except Exception as e:
        print(f"  [skip] chunk {idx}: {e}")
        return None


async def main() -> None:
    if _FINAL.exists():
        print(f"{_FINAL.name} already exists — will not overwrite. Delete it to regenerate.")
        return

    print(f"Fetching up to {_TARGET} chunks from Qdrant (no Voyage calls)...")
    try:
        chunk_texts = _fetch_chunks(_TARGET)
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        print("Is Qdrant running? Try: docker run -p 6333:6333 qdrant/qdrant")
        return

    if not chunk_texts:
        print("No chunks found in Qdrant. Run rag/embedder.py first.")
        return

    print(f"Generating QA pairs for {len(chunk_texts)} chunks...")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Process in batches of 10 to avoid hammering Gemini
    rows: list[dict] = []
    for batch_start in range(0, len(chunk_texts), 10):
        batch = chunk_texts[batch_start : batch_start + 10]
        tasks = [_generate_qa(client, text, batch_start + i + 1) for i, text in enumerate(batch)]
        results = await asyncio.gather(*tasks)
        for r in results:
            if r:
                rows.append(r)
        print(f"  {len(rows)} generated so far...")

    _OUT.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"\nWrote {len(rows)} candidates to {_OUT}")
    print("Review the file, remove bad rows, then rename to dataset.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
