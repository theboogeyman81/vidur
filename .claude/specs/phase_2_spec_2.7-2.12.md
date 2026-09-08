# Phase 2 Spec — Steps 2.7 through 2.12

## Scope

**Covered:** 2.7 NCERT ingestion · 2.8 hybrid retriever · 2.10 Langfuse tracing · 2.11 sessions/turns tables · 2.12 FastAPI token endpoint

**Explicitly skipped:** 2.9 (LangGraph corrective-RAG graph) — Phase 3 only. Phase 2 is naive top-k retrieval. That baseline is what Phase 3 has to beat.

**Already done:** 2.1–2.6 (registries, Pydantic AI agent, three tools, progress table)

---

## New dependencies

Add to `pyproject.toml`:

```
"qdrant-client[fastembed]>=1.9",
"voyageai>=0.3",
"pypdf>=4.0",
"tiktoken>=0.7",
"langfuse>=2.0",
"fastapi>=0.111",
"uvicorn>=0.30",
```

`livekit-api` is already pulled in transitively by `livekit-agents` — no need to add.

Run `uv sync` after.

---

## 2.7 — NCERT content ingestion

**What:** Load NCERT chapter PDFs, split into ~400-token chunks, write to disk as JSON for the embedder to pick up.

**Why:** The retriever needs something to retrieve. Phase 3's corrective-RAG graph needs a baseline to beat. This is that baseline content.

### Content

Drop 3–5 NCERT chapter PDFs into `rag/data/ncert/`. Suggested chapters (all freely available at ncert.nic.in):
- Class 11 Political Science Ch. 2 — Rights in the Indian Constitution
- Class 11 Political Science Ch. 3 — Election and Representation
- Class 11 Political Science Ch. 6 — Judiciary
- Class 12 Political Science Ch. 1 — The Cold War Era (for diverse topics)

### File: `rag/loader.py`

```python
import pathlib, json, re
import tiktoken
from pypdf import PdfReader

DATA_DIR = pathlib.Path("rag/data/ncert")
CHUNKS_PATH = pathlib.Path("rag/data/chunks.json")
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50

enc = tiktoken.get_encoding("cl100k_base")

def _pdf_to_text(path: pathlib.Path) -> str:
    reader = PdfReader(path)
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def _chunk(text: str, source: str) -> list[dict]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        chunks.append({"text": chunk_text, "source": source, "tokens": end - start})
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks

def load_chunks() -> list[dict]:
    all_chunks = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        text = _pdf_to_text(pdf)
        all_chunks.extend(_chunk(text, pdf.name))
    return all_chunks

if __name__ == "__main__":
    chunks = load_chunks()
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
    print(f"Wrote {len(chunks)} chunks to {CHUNKS_PATH}")
```

Run: `uv run python -m rag.loader`

### File: `rag/embedder.py`

Reads `chunks.json`, embeds with Voyage AI, upserts to Qdrant.

```python
import asyncio, json, os, pathlib, uuid
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, SparseVector, SparseVectorParams,
    NamedVector, NamedSparseVector,
)

CHUNKS_PATH = pathlib.Path("rag/data/chunks.json")
COLLECTION = "ncert"
DENSE_DIM = 512          # voyage-3-lite output dimension
QDRANT_URL = "http://localhost:6333"

vc = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
qc = QdrantClient(url=QDRANT_URL)


def _ensure_collection():
    existing = [c.name for c in qc.get_collections().collections]
    if COLLECTION not in existing:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )


def _embed_batch(texts: list[str]) -> list[list[float]]:
    result = vc.embed(texts, model="voyage-3-lite", input_type="document")
    return result.embeddings


def _sparse_batch(texts: list[str]) -> list[SparseVector]:
    # Use qdrant's built-in fastembed BM25 sparse encoder
    from fastembed import SparseTextEmbedding
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    embeddings = list(model.embed(texts))
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in embeddings
    ]


def upsert_chunks(chunks: list[dict], batch_size: int = 50) -> None:
    _ensure_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        dense_vecs = _embed_batch(texts)
        sparse_vecs = _sparse_batch(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vecs[j],
                    "sparse": sparse_vecs[j],
                },
                payload={"text": batch[j]["text"], "source": batch[j]["source"]},
            )
            for j in range(len(batch))
        ]
        qc.upsert(collection_name=COLLECTION, points=points)
        print(f"Upserted {i + len(batch)}/{len(chunks)}")


if __name__ == "__main__":
    chunks = json.loads(CHUNKS_PATH.read_text())
    upsert_chunks(chunks)
    print("Done.")
```

Run: `uv run python -m rag.embedder`

**Qdrant local setup (one-time):**
```
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

**New env vars needed:**
```
VOYAGE_API_KEY=
QDRANT_URL=http://localhost:6333
```

**Acceptance:** `uv run python -m rag.embedder` exits without error. Qdrant UI at `http://localhost:6333/dashboard` shows a `ncert` collection with N points.

---

## 2.8 — Hybrid retriever

**What:** Query Qdrant with both dense (semantic) and sparse (keyword) vectors, merge results, return top-k chunks.

**Why:** NCERT content has specific terminology ("Article 21", "DPSP") that pure semantic search misses. BM25 catches the keywords; dense catches the meaning. Hybrid beats either alone.

### File: `rag/retriever.py`

```python
import os
from pydantic import BaseModel
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, SearchRequest

COLLECTION = "ncert"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

_vc = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
_qc = QdrantClient(url=QDRANT_URL)


class Chunk(BaseModel):
    text: str
    source: str
    score: float


async def retrieve(query: str, top_k: int = 5) -> list[Chunk]:
    # Dense embedding for the query
    dense_vec = _vc.embed([query], model="voyage-3-lite", input_type="query").embeddings[0]

    # Sparse embedding (BM25)
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse_emb = next(sparse_model.embed([query]))
    sparse_vec = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )

    results = _qc.query_points(
        collection_name=COLLECTION,
        prefetch=[
            {"using": "dense", "query": dense_vec, "limit": top_k * 2},
            {"using": "sparse", "query": sparse_vec, "limit": top_k * 2},
        ],
        query={"fusion": "rrf"},   # Reciprocal Rank Fusion merges the two lists
        limit=top_k,
    )

    return [
        Chunk(
            text=r.payload["text"],
            source=r.payload["source"],
            score=r.score,
        )
        for r in results.points
    ]
```

**Note:** `retrieve()` is called by `agent/tools/retrieve.py` which already has the `ImportError` fallback. Once this file exists, the fallback is bypassed automatically — no changes to `retrieve.py` needed.

**Acceptance:**
```python
import asyncio, os
from dotenv import load_dotenv; load_dotenv()
from rag.retriever import retrieve
chunks = asyncio.run(retrieve("What are Fundamental Rights?"))
for c in chunks: print(c.source, c.score, c.text[:80])
```
Returns 5 chunks with non-zero scores.

---

## 2.10 — Langfuse tracing

**What:** Wrap every turn in a Langfuse trace. Each span (stt, llm, tool, tts) is a nested child. All required metrics logged.

**Why:** This is how we measure latency and prove the pipeline works. Without it, Phase 3's before/after comparison has no numbers.

### New env vars:
```
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

Sign up free at cloud.langfuse.com, create a project, copy the keys.

### File: `agent/session.py` (add Langfuse spans)

Langfuse SDK wraps at the turn level. Each turn = one `langfuse.trace()`. Inside it, nested `trace.span()` calls for stt, llm, tts.

```python
from langfuse import Langfuse

_lf = Langfuse()   # reads LANGFUSE_PUBLIC_KEY, SECRET_KEY, HOST from env

# In run_session(), before session.start():
_active_trace: dict = {}   # holds the current trace per turn

@session.on("user_input_transcribed")
def _on_transcribed(ev):
    if not ev.is_final: return
    _turn_start["t"] = time.perf_counter()
    trace = _lf.trace(name="turn")
    _active_trace["ref"] = trace
    trace.span(name="stt", output={
        "transcript": ev.transcript,
        "latency_ms": round(stt.last_latency_ms, 1),
        "engine": stt.name,
    })

@session.on("agent_state_changed")
def _on_state_changed(ev):
    if ev.old_state == "speaking" and ev.new_state in ("listening", "idle"):
        elapsed = (time.perf_counter() - _turn_start["t"]) * 1000
        trace = _active_trace.get("ref")
        if trace:
            trace.span(name="tts", output={
                "ttfb_ms": round(tts.last_ttfb_ms, 1),
                "engine": tts.name,
            })
            trace.update(output={
                "e2e_ms": round(elapsed + stt.last_latency_ms, 1),
                "interrupted": False,
            })
```

Tool spans: in `_PydanticAIStream._run()`, wrap the `vidur_agent.run()` call and log which tools fired:

```python
# after result = await vidur_agent.run(...)
trace = _active_trace.get("ref")
if trace:
    tools_called = [m.tool_name for m in result.new_messages()
                    if hasattr(m, "tool_name")]
    trace.span(name="llm", output={"tools_called": tools_called})
```

The `_active_trace` dict is shared between the session event handlers and the LLM stream. Pass it into `_PydanticAIStream` via the `PydanticAILLM` constructor.

**Required fields checklist:**
- `vad_end_ms` — approximate via `_turn_start["t"]` timestamp
- `stt_ms`, `stt_engine`, `transcript` — from stt span
- `llm_first_token_ms`, `llm_total_ms` — time `vidur_agent.run()` call
- `tools_called[]` — from result messages
- `tts_ttfb_ms`, `tts_engine` — from tts span
- `e2e_ms` — from state_changed handler
- `interrupted` — False for now, wire properly in a later pass

**Acceptance:** After a session turn, Langfuse dashboard shows a trace with nested stt / llm / tts spans and the correct timing values.

---

## 2.11 — SQLite sessions + turns tables

**What:** Add two more tables to `api/db.py` — `sessions` and `turns`. Every session start writes a row; every turn writes its full trace JSON.

**Why:** Persistent turn history lets the dashboard (Phase 7) replay any session and lets the eval harness join turn traces to sessions.

### Update `api/db.py`

Add to existing `init_db()`:

```python
_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,   -- LiveKit room name
    started_at  DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    trace       TEXT    NOT NULL,   -- JSON blob: all turn metrics
    ts          DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

async def log_session(session_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO sessions (id) VALUES (?)", (session_id,)
        )
        await db.commit()

async def log_turn(session_id: str, trace: dict) -> None:
    import json
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO turns (session_id, trace) VALUES (?, ?)",
            (session_id, json.dumps(trace)),
        )
        await db.commit()
```

### Wire in `session.py`

- Call `await log_session(ctx.room.name)` once at the start of `run_session()`
- Call `await log_turn(session_id, trace_dict)` in `_on_state_changed` with the full metrics dict

**Acceptance:** After a session, `SELECT * FROM turns` shows one row per completed turn with a valid JSON trace blob.

---

## 2.12 — FastAPI token endpoint

**What:** `GET /token?room=<name>&identity=<name>` returns a short-lived LiveKit access token. Runs as a separate process alongside the agent.

**Why:** The React frontend (Phase 7) needs to join a LiveKit room. It can't hold the API secret — the backend mints the token and the frontend uses it.

### File: `api/routes/token.py`

```python
from fastapi import APIRouter, HTTPException
from livekit.api import AccessToken, VideoGrants
import os

router = APIRouter()

@router.get("/token")
async def get_token(room: str, identity: str):
    if not room or not identity:
        raise HTTPException(400, "room and identity required")

    token = (
        AccessToken(
            api_key=os.environ["LIVEKIT_API_KEY"],
            api_secret=os.environ["LIVEKIT_API_SECRET"],
        )
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return {"token": token, "url": os.environ["LIVEKIT_URL"]}
```

### File: `api/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.token import router as token_router

app = FastAPI(title="Vidur API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(token_router)
```

### File: `api/routes/__init__.py` — empty package marker

Run: `uv run uvicorn api.main:app --reload --port 8000`

**Acceptance:** `curl "http://localhost:8000/token?room=test&identity=student"` returns `{"token": "...", "url": "wss://..."}`.

---

## Build order

```
1. uv sync                          (new deps)
2. Docker: start Qdrant             (needed for 2.7/2.8)
3. rag/loader.py                    (2.7 — chunk PDFs)
4. rag/embedder.py                  (2.7 — embed + upsert)
5. rag/retriever.py                 (2.8 — wire retrieve_tool live)
6. api/db.py                        (2.11 — add sessions/turns tables)
7. agent/session.py                 (2.10 — Langfuse spans + turn logging)
8. api/routes/__init__.py           (2.12)
9. api/routes/token.py              (2.12)
10. api/main.py                     (2.12)
```

Steps 3–5 must run in order (load → embed → retrieve).
Steps 6–10 are independent of each other and of 3–5.

---

## New env vars (add to `.env` and `.env.example`)

```
VOYAGE_API_KEY=
QDRANT_URL=http://localhost:6333
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Files created this scope

```
rag/
  __init__.py
  loader.py
  embedder.py
  retriever.py
  data/
    ncert/          (PDFs go here — not committed)
    chunks.json     (not committed)
api/
  routes/
    __init__.py
    token.py
  main.py
  db.py             (EDIT — add sessions/turns tables + helpers)
agent/
  session.py        (EDIT — add Langfuse spans, log_session, log_turn)
pyproject.toml      (EDIT — new deps)
.env.example        (EDIT — new keys)
```

---

## Done when (2.7–2.12)

- [ ] `uv run python -m rag.embedder` runs without error, Qdrant shows chunks
- [ ] Asking the agent a UPSC question returns an answer citing NCERT content (not the placeholder)
- [ ] Langfuse dashboard shows a trace for each turn with stt/llm/tts spans
- [ ] `SELECT * FROM turns` shows one row per completed turn
- [ ] `curl localhost:8000/token?room=test&identity=me` returns a valid JWT
- [ ] `uv run ruff check .` passes
