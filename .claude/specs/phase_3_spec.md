# Phase 3 — `feat/corrective-rag`

**Branch:** `feat/corrective-rag`
**Status:** not started
**Blocked by:** Phase 2 merged (`feat/tools-and-baseline-rag`)
**Goal:** Beat the naive top-k baseline with a corrective-RAG graph. The deliverable is two `results.json` files — before and after — with Ragas numbers showing the delta honestly.

> Eval dataset comes first. The graph only matters if you can measure it.

---

## Context: what Phase 2 left you

- `rag/retriever.py` — hybrid dense+sparse Qdrant retrieval, returns `list[Chunk]`
- `agent/tools/retrieve.py` — calls `retriever.retrieve()` naively, logs `rag_total_ms` stub
- Langfuse spans exist per turn but RAG fields (`rag_rewrites`, `chunks_passed_grading`, etc.) are unstubbed
- `rag/graph.py` and `rag/nodes/` do not yet exist

---

## Features

### 3.1 — RAG eval dataset

**File:** `evals/datasets/rag_qa/`

Build ~50 question + ground-truth-chunk pairs from the ingested NCERT content.

Format: one JSON file per question, or a single `dataset.jsonl` — either works.

```json
{
  "id": "q001",
  "question": "What were the main causes of the 1857 revolt?",
  "ground_truth_chunks": ["chunk text that should appear in the answer..."],
  "ground_truth_answer": "The 1857 revolt was caused by..."
}
```

- Questions must be drawn from what's actually in Qdrant — don't invent questions about content that isn't ingested
- Mix question types: factual, ambiguous (to force rewrite), multi-hop
- At least 10 questions should be deliberately ambiguous so the corrective loop has a chance to demonstrate value
- Ground-truth chunks are verbatim text from the ingested NCERT content

**Script to help:** `evals/datasets/rag_qa/generate_dataset.py` — optionally use Gemini to draft questions from the chunk corpus, then manually review and trim to ~50

---

### 3.2 — Baseline eval run

**Files:** `evals/run_rag_eval.py`, `evals/datasets/rag_qa/results_baseline.json`

Run Ragas against the Phase 2 naive top-k retriever. Record the numbers before touching the graph.

Metrics (via `ragas`):
- `context_precision` — are the retrieved chunks actually relevant?
- `context_recall` — do retrieved chunks cover the ground truth?
- `faithfulness` — does the generated answer stay grounded in retrieved context?
- `answer_relevancy` — does the answer address the question?

Runner structure:

```python
# evals/run_rag_eval.py
# Usage: uv run python -m evals.run_rag_eval --mode baseline|corrective
```

- Loads `evals/datasets/rag_qa/dataset.jsonl`
- For each question: calls `rag.retriever.retrieve()` (baseline) or the LangGraph graph (corrective)
- Sends to Ragas with a Gemini LLM judge
- Writes `results_baseline.json` (or `results_corrective.json`) with timestamp, mode, per-question scores, and aggregate p50/p95

**Never overwrite an existing results file.** Append a timestamp suffix if the file already exists.

---

### 3.3 — LangGraph corrective-RAG graph

**File:** `rag/graph.py`

A `StateGraph` with four nodes and one conditional edge. Hard rules:
- Max 2 rewrite iterations — the loop cannot run indefinitely
- If total graph time > 2500ms, bail out and return best chunks seen so far
- This graph is invoked once per tool call, never per turn — it is not in the audio path

```python
# State schema
class RAGState(TypedDict):
    query: str
    original_query: str
    chunks: list[Chunk]
    passed_chunks: list[Chunk]
    rewrites: int           # how many times we've rewritten the query
    bailed_out: bool
    start_time: float       # time.perf_counter() at graph entry
```

Graph flow:
```
retrieve_node → grade_node → conditional:
                                ├── any passed? → generate_node → END
                                └── rewrites < 2? → rewrite_node → retrieve_node
                                └── rewrites >= 2 → generate_node → END (with whatever chunks exist)
```

Compile with `MemorySaver` checkpointer for state within one lookup (not cross-turn).

```python
# rag/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_graph() -> CompiledGraph:
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {...})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)

    return g.compile(checkpointer=MemorySaver())
```

---

### 3.4 — Node: retrieve

**File:** `rag/nodes/retrieve.py`

Calls `rag.retriever.retrieve(state["query"], top_k=5)`. Respects the 2500ms bail-out — checks `time.perf_counter() - state["start_time"]` before executing; if already over 2400ms, skips the Qdrant call and sets `bailed_out: true`.

Returns updated state with `chunks` populated.

---

### 3.5 — Node: grade

**File:** `rag/nodes/grade.py`

For each chunk in `state["chunks"]`, ask an LLM (Gemini flash, temp=0) to score it relevant/not-relevant to the query.

```
Prompt: "Is this chunk relevant to the question: '{query}'?\nChunk: '{chunk.text}'\nAnswer YES or NO only."
```

- Parse YES/NO (case-insensitive), treat anything else as NO
- Set `passed_chunks` to chunks that scored YES
- One Gemini call per chunk — run them concurrently with `asyncio.gather`
- If all chunks fail: do not set `passed_chunks = []` silently — log the count so it's visible in Langfuse

---

### 3.6 — Node: rewrite

**File:** `rag/nodes/rewrite.py`

Reformulate the query to improve retrieval. One Gemini call.

```
Prompt: "The following search query returned no relevant results from an NCERT textbook database.
Rewrite it to be more specific and likely to match educational content.
Original question: '{original_query}'
Current query: '{query}'
Rewritten query:"
```

- Increments `state["rewrites"]`
- Updates `state["query"]` with the rewritten text
- Does NOT modify `original_query` — that stays fixed for the Ragas eval

---

### 3.7 — Node: generate

**File:** `rag/nodes/generate.py`

Composes a grounded answer from `passed_chunks` (or `chunks` if nothing passed grading).

This node is only used by the eval runner, not by the agent. The agent still speaks its own response — the generate node exists so Ragas has a string to evaluate faithfulness and answer relevancy against.

```python
async def generate_node(state: RAGState) -> RAGState:
    context_chunks = state["passed_chunks"] or state["chunks"]
    context = "\n\n".join(c.text for c in context_chunks)
    answer = await gemini_generate(
        f"Answer this question based only on the provided context.\n\nContext:\n{context}\n\nQuestion: {state['original_query']}"
    )
    return {**state, "answer": answer}
```

Add `answer: str` to `RAGState`.

---

### 3.8 — Bail-out logic

**Location:** `rag/nodes/retrieve.py` (checked before each retrieve call) + `rag/graph.py` (post-graph check)

Two-layer:
1. **Pre-call check** — in `retrieve_node`, if elapsed > 2400ms, skip Qdrant, set `bailed_out: True`, return current state
2. **Post-graph check** — in the public `run_graph()` wrapper in `graph.py`, wrap the compiled graph invoke in a `asyncio.wait_for(..., timeout=2.5)`. If it times out, catch `asyncio.TimeoutError`, log it, return best chunks seen

```python
async def run_graph(query: str) -> tuple[list[Chunk], dict]:
    """Public entry point. Returns (chunks, metadata)."""
    ...
    metadata = {
        "rag_rewrites": state["rewrites"],
        "rag_bailed_out": state["bailed_out"],
        "chunks_retrieved": len(state["chunks"]),
        "chunks_passed_grading": len(state["passed_chunks"]),
        "rag_total_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return best_chunks, metadata
```

---

### 3.9 — Wire graph into `retrieve` tool

**File:** `agent/tools/retrieve.py`

Replace the direct `rag.retriever.retrieve()` call with `rag.graph.run_graph()`. The tool signature stays identical — the agent sees no change.

```python
async def retrieve_tool(ctx: RunContext[dict], query: str) -> str:
    t0 = time.perf_counter()
    try:
        from rag.graph import run_graph
        chunks, meta = await run_graph(query)
        ctx.deps["langfuse_span"].update(output=meta)  # log RAG fields
        return "\n\n".join(c.text for c in chunks)
    except Exception as e:
        return f"[RAG unavailable: {type(e).__name__}]"
```

Langfuse span update logs: `rag_total_ms`, `rag_rewrites`, `rag_bailed_out`, `chunks_retrieved`, `chunks_passed_grading`.

---

### 3.10 — Corrective eval run

**File:** `evals/datasets/rag_qa/results_corrective.json`

Re-run `evals/run_rag_eval.py --mode corrective` against the live LangGraph graph. Same Ragas metrics, same dataset.

The delta between `results_baseline.json` and `results_corrective.json` is the deliverable. If the corrective graph doesn't beat the baseline, that is still a valid result — record it honestly in FINDINGS.md with the breakdown (e.g., "grade node was too strict, filtered good chunks").

---

### 3.11 — Langfuse RAG spans

**File:** `agent/tools/retrieve.py` (see 3.9), `agent/session.py`

Every live turn that fires `retrieve` must log these fields on the `tool:retrieve` Langfuse span:

| Field | Type | Notes |
|---|---|---|
| `rag_total_ms` | float | wall clock from graph entry to return |
| `rag_rewrites` | int | 0, 1, or 2 |
| `rag_bailed_out` | bool | true if 2500ms limit was hit |
| `chunks_retrieved` | int | total chunks from Qdrant |
| `chunks_passed_grading` | int | chunks that scored YES |

Node-level spans in Langfuse (retrieve / grade / rewrite / generate) are a nice-to-have — add them if `rag_total_ms` timing shows breakdown is needed. Not required for done.

---

## Done when

1. `evals/datasets/rag_qa/dataset.jsonl` exists with ~50 labelled Q&A pairs
2. `results_baseline.json` committed — Ragas scores for naive top-k
3. `results_corrective.json` committed — Ragas scores for corrective graph
4. Delta is documented (even if negative)
5. Graph bails at 2500ms and logs `rag_bailed_out: true` — verifiable in a Langfuse trace
6. `rag_rewrites` count visible in every `tool:retrieve` Langfuse span
7. Agent conversations still work end-to-end with the graph wired in

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
      generate_dataset.py       (optional helper)
      results_baseline.json     (committed after 3.2)
      results_corrective.json   (committed after 3.10)
  metrics/
    ragas_helpers.py            (thin wrapper if Ragas API needs patching for Gemini)

agent/
  tools/retrieve.py             (rewritten — replaces naive call with run_graph)
```

---

## Dependency notes

New packages needed:
- `langgraph` — corrective graph
- `ragas` — eval metrics
- `jiwer` — already planned for Phase 4 STT eval, can add now

All added to `pyproject.toml` under `[project.dependencies]`, pinned loosely (`>=x.y`).

Ragas requires an LLM judge and an embeddings model. Wire it to:
- LLM: Gemini flash (same as agent — no extra cost)
- Embeddings: Voyage AI (already available)

If Ragas doesn't support Voyage embeddings natively, wrap it with a LangChain-compatible adapter or use the `ragas.llms` / `ragas.embeddings` override hooks. Note the workaround in a comment.
