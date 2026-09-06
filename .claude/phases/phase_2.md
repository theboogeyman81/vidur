# Phase 2 — `feat/tools-and-rag`

**Goal:** Give the agent a brain. Provider adapter layer locked in, three tools wired, NCERT content in Qdrant, every turn traced in Langfuse. Phase 1 must be merged and talking before this starts.

---

## Features

### 2.1 — Provider adapter layer (STT)
- Files: `agent/providers/stt/__init__.py`, `agent/providers/stt/sarvam.py`, `agent/providers/stt/deepgram.py`, `agent/providers/stt/whisper.py`, `agent/providers/stt/google.py`
- Each file implements `STTProvider` from `base.py`
- Registry dict in `agent/providers/stt/__init__.py`: `{"sarvam": SarvamSTT, "deepgram": DeepgramSTT, ...}`
- Active engine selected via `STT_ENGINE` env var, defaults to `sarvam`
- Deepgram and Whisper stubs are enough here — full impl lands in Phase 3

### 2.2 — Provider adapter layer (TTS)
- Files: `agent/providers/tts/__init__.py`, `agent/providers/tts/sarvam.py`, `agent/providers/tts/cartesia.py`, `agent/providers/tts/elevenlabs.py`, `agent/providers/tts/piper.py`
- Same pattern: registry + env var `TTS_ENGINE`
- Cartesia, ElevenLabs, Piper stubs — full impl in Phase 4

### 2.3 — Pydantic AI agent
- File: `agent/session.py` (replace the raw Gemini call from Phase 1)
- Wrap LLM call in a Pydantic AI agent with typed tool definitions
- Agent receives the transcript, returns a typed response
- Tool calls extracted from the response and dispatched before TTS

### 2.4 — Tool: RAG lookup (`retrieve`)
- File: `agent/tools/retrieve.py`
- Calls into `rag/retriever.py` to do a hybrid Qdrant search
- Returns top-k chunks as a string the LLM can cite
- Logs `rag_total_ms` to the turn trace

### 2.5 — Tool: Quiz generator (`quiz`)
- File: `agent/tools/quiz.py`
- Takes a topic string, calls Gemini to generate a Socratic question
- Returns the question text — agent speaks it as the next turn
- Input validated with Pydantic

### 2.6 — Tool: Progress logger (`progress`)
- File: `agent/tools/progress.py`
- Writes a row to SQLite: `(session_id, topic, timestamp)`
- No return value needed — side effect only
- Schema defined in `api/db.py`

### 2.7 — NCERT content ingestion
- Files: `rag/loader.py`, `rag/embedder.py`
- `loader.py`: load a small NCERT chapter subset (PDF or text), chunk to ~400 tokens with overlap
- `embedder.py`: embed each chunk with Voyage AI, upsert to Qdrant with both dense and sparse vectors
- Run once as a script: `uv run python -m rag.embedder`
- Qdrant runs locally via Docker

### 2.8 — Hybrid retriever
- File: `rag/retriever.py`
- Qdrant hybrid search: dense (Voyage) + sparse (BM25)
- Returns top-k `Chunk(text, source, score)` Pydantic objects
- No LangGraph corrective loop yet — straight retrieval only

### 2.9 — LangGraph corrective-RAG graph
- Files: `rag/graph.py`, `rag/nodes/retrieve.py`, `rag/nodes/grade.py`, `rag/nodes/rewrite.py`, `rag/nodes/generate.py`
- `StateGraph`: retrieve → grade → branch (generate | rewrite → retrieve, max 2 retries)
- Grader: LLM prompt that scores each chunk relevant/not
- Hard cap: if graph exceeds 2500ms, return best chunks and set `rag_bailed_out: true`
- Invoked once per `retrieve` tool call, never per turn

### 2.10 — Langfuse tracing
- File: `agent/session.py` (add spans)
- Wrap each turn in a Langfuse trace with nested spans: `stt`, `llm`, `tool:<name>`, `tts`
- Log all required fields: `vad_end_ms`, `stt_ms`, `stt_engine`, `transcript`, `llm_first_token_ms`, `llm_total_ms`, `tools_called[]`, `tts_ttfb_ms`, `tts_engine`, `e2e_ms`, `interrupted`
- RAG spans log: `rag_total_ms`, `rag_rewrites`, `rag_bailed_out`, `chunks_retrieved`, `chunks_passed_grading`
- Use `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` from `.env`

### 2.11 — SQLite schema
- File: `api/db.py`
- Tables: `sessions(id, started_at)`, `turns(id, session_id, trace JSON)`, `progress(id, session_id, topic, ts)`
- Migrations run on startup, no migration tool needed at this scale

### 2.12 — FastAPI token endpoint
- Files: `api/main.py`, `api/routes/token.py`
- `GET /token` — returns a short-lived LiveKit room token
- Required for the frontend to join a room without exposing secrets

---

## Done when

- Agent uses Pydantic AI, not a raw Gemini call
- All three tools fire correctly mid-conversation and appear in Langfuse spans
- NCERT content is in Qdrant; RAG lookup returns relevant chunks
- Corrective-RAG loop retries on bad chunks, bails at 2500ms
- Every turn has a Langfuse trace with all required fields
- Provider registry lets you swap STT/TTS engine via env var with no code change

---

## Files created this phase

```
agent/
  session.py          (rewritten)
  tools/
    retrieve.py
    quiz.py
    progress.py
  providers/
    stt/__init__.py
    stt/deepgram.py
    stt/whisper.py
    stt/google.py
    tts/__init__.py
    tts/cartesia.py
    tts/elevenlabs.py
    tts/piper.py
rag/
  loader.py
  embedder.py
  retriever.py
  graph.py
  nodes/
    retrieve.py
    grade.py
    rewrite.py
    generate.py
api/
  main.py
  db.py
  routes/token.py
```
