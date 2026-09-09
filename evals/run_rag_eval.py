"""RAG eval runner — baseline (naive top-k) vs corrective (LangGraph graph).

Usage:
    uv run python -m evals.run_rag_eval --mode baseline [--limit N]
    uv run python -m evals.run_rag_eval --mode corrective [--limit N]

Writes evals/results/rag_{mode}_{timestamp}.json.
Never overwrites an existing results file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time
from datetime import UTC, datetime

import numpy as np
from dotenv import load_dotenv

load_dotenv()

_DATASET = pathlib.Path(__file__).parent / "datasets" / "rag_qa" / "dataset.jsonl"
_RESULTS_DIR = pathlib.Path(__file__).parent / "results"

_GEN_PROMPT = """\
Using only the following context, answer the question in 1-3 sentences. \
If the context does not contain the answer, say "I don't know based on the provided context."

Context:
{context}

Question: {question}"""


def _load_dataset(limit: int | None) -> list[dict]:
    if not _DATASET.exists():
        raise FileNotFoundError(
            f"{_DATASET} not found. Run generate_dataset.py first and rename the draft."
        )
    lines = _DATASET.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if limit:
        rows = rows[:limit]
    return rows


async def _generate_answer(client, context: str, question: str) -> str:
    resp = await client.aio.models.generate_content(
        model="gemini-3.6-flash",
        contents=_GEN_PROMPT.format(context=context, question=question),
    )
    return resp.text.strip()


async def _run_retrieval(rows: list[dict], mode: str) -> list[dict]:
    import google.genai as genai  # noqa: PLC0415

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    samples = []

    for row in rows:
        q = row["question"]
        print(f"  [{mode}] {row['id']}: {q[:60]}...")

        if mode == "baseline":
            from rag.retriever import retrieve  # noqa: PLC0415

            chunks = await retrieve(q, top_k=5)
            meta: dict = {}
        else:
            from rag.graph import run_graph  # noqa: PLC0415

            chunks, meta = await run_graph(q)

        contexts = [c.text for c in chunks]
        context_str = "\n\n".join(contexts)
        answer = await _generate_answer(client, context_str, q)

        samples.append(
            {
                "id": row["id"],
                "user_input": q,
                "retrieved_contexts": contexts,
                "response": answer,
                "reference": row["ground_truth_answer"],
                "reference_contexts": row["ground_truth_chunks"],
                "rag_meta": meta,
            }
        )

    return samples


def _evaluate_with_ragas(samples: list[dict]) -> tuple[list[dict], dict]:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    ragas_samples = [
        SingleTurnSample(
            user_input=s["user_input"],
            retrieved_contexts=s["retrieved_contexts"],
            response=s["response"],
            reference=s["reference"],
            reference_contexts=s["reference_contexts"],
        )
        for s in samples
    ]

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.environ["GEMINI_API_KEY"],
        )
    )
    embedder = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.environ["GEMINI_API_KEY"],
        )
    )

    dataset = EvaluationDataset(samples=ragas_samples)
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embedder,
    )

    result_df = result.to_pandas()
    per_question = []
    metric_cols = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    for i, row in result_df.iterrows():
        entry = {"id": samples[i]["id"]}
        for col in metric_cols:
            if col in row:
                entry[col] = float(row[col]) if row[col] is not None else None
        per_question.append(entry)

    aggregate: dict = {}
    for col in metric_cols:
        vals = [e[col] for e in per_question if e.get(col) is not None]
        if vals:
            aggregate[col] = {
                "mean": round(float(np.mean(vals)), 4),
                "p50": round(float(np.percentile(vals, 50)), 4),
                "p95": round(float(np.percentile(vals, 95)), 4),
            }

    return per_question, aggregate


def _write_results(mode: str, per_question: list[dict], aggregate: dict) -> pathlib.Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _RESULTS_DIR / f"rag_{mode}_{ts}.json"
    payload = {
        "timestamp": ts,
        "mode": mode,
        "n_questions": len(per_question),
        "aggregate": aggregate,
        "per_question": per_question,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


async def main(mode: str, limit: int | None) -> None:
    print(f"Loading dataset (limit={limit})...")
    rows = _load_dataset(limit)
    print(f"Running {mode} retrieval on {len(rows)} questions...")
    t0 = time.perf_counter()
    samples = await _run_retrieval(rows, mode)
    print(f"Retrieval done in {round((time.perf_counter() - t0)*1000)}ms. Running Ragas...")
    per_question, aggregate = _evaluate_with_ragas(samples)
    out_path = _write_results(mode, per_question, aggregate)
    print(f"\nResults written to {out_path}")
    print("\nAggregate scores:")
    for metric, stats in aggregate.items():
        print(f"  {metric}: mean={stats['mean']}, p50={stats['p50']}, p95={stats['p95']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "corrective"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()
    asyncio.run(main(args.mode, args.limit))
