# Phase 3 — `feat/corrective-rag`

**Goal:** Beat the naive top-k baseline with a corrective-RAG graph. The deliverable is two `results.json` files — before and after — with Ragas numbers showing the delta honestly. Eval dataset comes first; the graph only matters if you can measure it.

**Full spec:** `.claude/specs/phase_3_spec.md`

---

## Features

### 3.1 — RAG eval dataset
- Directory: `evals/datasets/rag_qa/`
- ~50 question + ground-truth-chunk pairs, drawn from what's actually ingested in Qdrant
- `dataset.jsonl`: `{"id": "q001", "question": "...", "ground_truth_chunks": [...], "ground_truth_answer": "..."}`
- Mix factual, multi-hop, and deliberately ambiguous questions — at least 10 ambiguous, so the corrective loop has something to fix
- `generate_dataset.py` drafts candidates from the chunk corpus; hand-review and trim to ~50

### 3.2 — Baseline eval run
- Run `evals/run_rag_eval.py --mode baseline` against the Phase 2 naive top-k retriever, before touching the graph
- Ragas metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, with a Gemini LLM judge
- Writes `results_baseline.json` — timestamped, per-question scores, aggregate p50/p95. Never overwrite.

### 3.3 — LangGraph corrective-RAG graph
- File: `rag/graph.py`
- `StateGraph` with four nodes (`retrieve`, `grade`, `rewrite`, `generate`) and one conditional edge
- Max 2 rewrite iterations; bail out if total graph time > 2500ms and return best chunks seen
- Invoked once per tool call, never per turn — not in the audio path
- Compile with `MemorySaver` checkpointer, scoped to one lookup only

### 3.4 — Node: retrieve
- File: `rag/nodes/retrieve.py`
- Calls `rag.retriever.retrieve(query, top_k=5)`
- Pre-call bail-out check: if elapsed > 2400ms, skip Qdrant, set `bailed_out: true`

### 3.5 — Node: grade
- File: `rag/nodes/grade.py`
- Gemini flash, temp=0, one YES/NO relevance call per chunk, run concurrently via `asyncio.gather`
- `passed_chunks` = chunks scored YES; if all fail, log the count rather than silently returning empty

### 3.6 — Node: rewrite
- File: `rag/nodes/rewrite.py`
- One Gemini call reformulates the query for better retrieval
- Increments `rewrites`, updates `query`; `original_query` stays fixed for the Ragas eval

### 3.7 — Node: generate
- File: `rag/nodes/generate.py`
- Composes a grounded answer from `passed_chunks` (or `chunks` if nothing passed)
- Only used by the eval runner — the agent still speaks its own response; this node exists so Ragas has a string to score faithfulness/relevancy against

### 3.8 — Bail-out logic
- Two-layer: pre-call check in `retrieve_node` (elapsed > 2400ms), plus a post-graph `asyncio.wait_for(..., timeout=2.5)` in `run_graph()`
- On timeout, catch it, log it, return the best chunks seen so far

### 3.9 — Wire graph into `retrieve` tool
- File: `agent/tools/retrieve.py`
- Replace the direct `rag.retriever.retrieve()` call with `rag.graph.run_graph()` — tool signature unchanged, agent sees no difference
- Logs `rag_total_ms`, `rag_rewrites`, `rag_bailed_out`, `chunks_retrieved`, `chunks_passed_grading` on the Langfuse span

### 3.10 — Corrective eval run
- Re-run `evals/run_rag_eval.py --mode corrective` against the live graph, same dataset, same metrics
- Writes `results_corrective.json`
- The delta vs. `results_baseline.json` is the deliverable — if the graph doesn't win, record that honestly in FINDINGS.md with why (e.g. grade node too strict)

### 3.11 — Langfuse RAG spans
- Every live turn that fires `retrieve` logs `rag_total_ms`, `rag_rewrites`, `rag_bailed_out`, `chunks_retrieved`, `chunks_passed_grading` on the `tool:retrieve` span
- Node-level spans (retrieve/grade/rewrite/generate) are a nice-to-have, not required for done

---

## Done when

- `evals/datasets/rag_qa/dataset.jsonl` exists with ~50 labelled Q&A pairs
- `results_baseline.json` and `results_corrective.json` both committed
- Delta documented in FINDINGS.md, even if negative
- Graph bails at 2500ms and logs `rag_bailed_out: true` — verifiable in a Langfuse trace
- `rag_rewrites` visible in every `tool:retrieve` Langfuse span
- Agent conversations still work end-to-end with the graph wired in

---

## Files created this phase

```
rag/
  graph.py
  nodes/
    retrieve.py
    grade.py
    rewrite.py
    generate.py

evals/
  run_rag_eval.py
  datasets/
    rag_qa/
      dataset.jsonl
      generate_dataset.py
      results_baseline.json
      results_corrective.json
  metrics/
    ragas_helpers.py   (only if Ragas needs a Voyage/Gemini adapter)

agent/
  tools/retrieve.py   (rewritten — replaces naive call with run_graph)
```

---

## Dependency notes

New packages: `langgraph`, `ragas`, `jiwer` (jiwer can be added now, needed properly in Phase 4).

Ragas needs an LLM judge and an embeddings model — Gemini flash for the judge (no extra cost, same as the agent), Voyage AI for embeddings. If Ragas doesn't support Voyage natively, wrap it via `ragas.llms` / `ragas.embeddings` override hooks and note the workaround in a comment.
