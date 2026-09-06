# Phase 2 Build Spec — Steps 2.1 through 2.6

## Scope

**Covered:** 2.1 STT registry · 2.2 TTS registry · 2.3 Pydantic AI agent · 2.4 retrieve tool · 2.5 quiz tool · 2.6 progress tool + SQLite  
**Deferred:** 2.7 NCERT ingestion · 2.8 hybrid retriever · 2.9 corrective-RAG graph · 2.10 Langfuse · 2.11 sessions/turns tables · 2.12 FastAPI token

---

## What Phase 1 left

| File | State |
|---|---|
| `agent/providers/base.py` | `STTProvider`, `TTSProvider` protocols, `STTResult`, `TTSResult` models — done |
| `agent/providers/stt/sarvam.py` | `SarvamSTT` dual-mode (LiveKit plugin + eval harness) — done |
| `agent/providers/tts/sarvam.py` | `SarvamTTS` (eval) + `SarvamLKTTS` (LiveKit) — done |
| `agent/providers/stt/__init__.py` | empty |
| `agent/providers/tts/__init__.py` | empty |
| `agent/session.py` | bare `AgentSession` with raw Gemini — to be rewritten |
| `agent/worker.py` | LiveKit entrypoint — minor additions only |
| `agent/prompts/tutor_v1.md` | system prompt — unchanged |

---

## New dependencies

Add to `pyproject.toml` `dependencies`:

```
"pydantic-ai[google]>=0.0.14",
"aiosqlite>=0.20",
```

Run `uv sync` after.

---

## 2.1 — STT provider registry

**Goal:** Select STT engine via `STT_ENGINE` env var. Stubs for Deepgram, Whisper, Google so the registry is complete without breaking anything.

### `agent/providers/stt/__init__.py`

```python
import os
from agent.providers.stt.sarvam import SarvamSTT
from agent.providers.stt.deepgram import DeepgramSTT
from agent.providers.stt.whisper import WhisperSTT
from agent.providers.stt.google import GoogleSTT

_REGISTRY: dict[str, type] = {
    "sarvam": SarvamSTT,
    "deepgram": DeepgramSTT,
    "whisper": WhisperSTT,
    "google": GoogleSTT,
}

def get_stt_provider():
    key = os.getenv("STT_ENGINE", "sarvam")
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown STT engine: {key!r}. Options: {list(_REGISTRY)}")
    return cls()
```

### `agent/providers/stt/deepgram.py` (stub)

```python
from agent.providers.base import STTResult

class DeepgramSTT:
    name = "deepgram-nova"

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        raise NotImplementedError("DeepgramSTT full impl lands in Phase 4")
```

Same stub pattern for:
- `agent/providers/stt/whisper.py` — `name = "whisper-large-v3"`
- `agent/providers/stt/google.py` — `name = "google-stt"`

**Acceptance:** `from agent.providers.stt import get_stt_provider; get_stt_provider()` imports cleanly with any `STT_ENGINE` value. Only raises `NotImplementedError` when `transcribe()` is actually called on a stub.

---

## 2.2 — TTS provider registry

**Goal:** Select TTS engine via `TTS_ENGINE` env var. Stubs for Cartesia, ElevenLabs, Piper.

### `agent/providers/tts/__init__.py`

Same structure as STT registry:

```python
_REGISTRY: dict[str, type] = {
    "sarvam": SarvamTTS,
    "cartesia": CartesiaTTS,
    "elevenlabs": ElevenLabsTTS,
    "piper": PiperTTS,
}

def get_tts_provider(): ...
```

### Stubs

- `agent/providers/tts/cartesia.py` — `name = "cartesia-sonic"`
- `agent/providers/tts/elevenlabs.py` — `name = "elevenlabs-flash"`
- `agent/providers/tts/piper.py` — `name = "piper-local"`

Each stub: `raise NotImplementedError("full impl in Phase 5")` in `synthesize()`.

**Note:** The registry exposes eval-harness adapters (`SarvamTTS`, not `SarvamLKTTS`). The LiveKit plugin classes stay separate and are wired directly in `session.py`. This is by design — the registry is the eval harness surface; LiveKit wiring is the voice surface.

---

## 2.3 — Pydantic AI agent

**Goal:** Replace the raw Gemini call in the LiveKit session with a Pydantic AI `Agent` that has typed tool definitions. Tool calls fire and resolve inside Pydantic AI's run loop before TTS speaks.

### Architecture

LiveKit `AgentSession` controls audio (VAD, STT routing, TTS playback, barge-in). We keep it. For the LLM slot, we write a thin `PydanticAILLM` class that implements LiveKit's `llm.LLM` interface. When LiveKit calls `chat()`, we run the Pydantic AI agent, tools fire, and we return the final text reply as a streaming response.

This means:
- LiveKit still handles the pipeline shape
- Pydantic AI handles tool dispatch and typed outputs
- Barge-in detection remains LiveKit's job — no change there

### New file: `agent/brain.py`

Defines the Pydantic AI `Agent` singleton. Loads system prompt from `prompts/tutor_v1.md`. Registers the three tools.

```python
import pathlib
from pydantic_ai import Agent
from agent.tools.retrieve import retrieve_tool
from agent.tools.quiz import quiz_tool
from agent.tools.progress import progress_tool

_SYSTEM_PROMPT = (pathlib.Path(__file__).parent / "prompts" / "tutor_v1.md").read_text()

vidur_agent = Agent(
    "google-gla:gemini-2.5-flash",
    system_prompt=_SYSTEM_PROMPT,
    tools=[retrieve_tool, quiz_tool, progress_tool],
)
```

### Updated `agent/session.py`

Write a `PydanticAILLM` class that wraps `vidur_agent.run()`. Wire it into `AgentSession(llm=PydanticAILLM(...))`.

Key details:
- `PydanticAILLM.chat()` accepts LiveKit `ChatContext`, extracts the user message text, runs the Pydantic AI agent with `deps={"session_id": ctx.room.name}`, yields the result text as a single `LLMStream` chunk
- Tools called by the agent during `run()` execute before `chat()` returns — LiveKit never sees the tool calls, only the final reply
- The existing turn-trace logging (stt, tts timing) stays in the session event handlers unchanged

**Acceptance:** A conversation turn completes without error. Asking "Quiz me on Fundamental Rights" causes `quiz_tool` to be called. The agent speaks a question.

---

## 2.4 — Tool: retrieve

**File:** `agent/tools/retrieve.py`

```python
import time
from pydantic_ai import RunContext

async def retrieve_tool(ctx: RunContext[dict], query: str) -> str:
    """Look up relevant NCERT content for the student's question."""
    t0 = time.perf_counter()
    try:
        from rag.retriever import retrieve
        chunks = await retrieve(query, top_k=5)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        # rag_total_ms logged here; Langfuse span added in 2.10
        return "\n\n".join(c.text for c in chunks)
    except ImportError:
        return "[RAG retriever not yet built — stub response]"
```

The `ImportError` fallback is intentional: `rag/retriever.py` doesn't exist until step 2.8. This lets the tool register and dispatch cleanly now without crashing the agent.

**Acceptance:** Tool invocation returns a string. No crash. Langfuse logging is a `# TODO: 2.10` comment for now.

---

## 2.5 — Tool: quiz

**File:** `agent/tools/quiz.py`

```python
import os
from pydantic_ai import RunContext

async def quiz_tool(ctx: RunContext[dict], topic: str) -> str:
    """Generate one Socratic UPSC question on the given topic."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.5-flash")
    resp = model.generate_content(
        f"Generate one Socratic question for UPSC preparation on: {topic}. "
        "Output only the question, no preamble."
    )
    return resp.text.strip()
```

Input `topic` is a plain string. Pydantic AI extracts it from the LLM's tool call JSON automatically.

**Acceptance:** `await quiz_tool(ctx, "Preamble of the Constitution")` returns a non-empty question string.

---

## 2.6 — Tool: progress + SQLite schema

### `api/db.py` (new)

Only the `progress` table is created here. `sessions` and `turns` tables come in step 2.11.

```python
import aiosqlite
import pathlib

DB_PATH = pathlib.Path("vidur.db")

_CREATE_PROGRESS = """
CREATE TABLE IF NOT EXISTS progress (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    topic       TEXT    NOT NULL,
    ts          DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_CREATE_PROGRESS)
        await db.commit()

async def log_progress(session_id: str, topic: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO progress (session_id, topic) VALUES (?, ?)",
            (session_id, topic),
        )
        await db.commit()
```

### `agent/tools/progress.py`

```python
from pydantic_ai import RunContext
from api.db import log_progress

async def progress_tool(ctx: RunContext[dict], topic: str) -> None:
    """Record that the student studied this topic in the current session."""
    session_id = ctx.deps.get("session_id", "unknown")
    await log_progress(session_id, topic)
```

Return type is `None` — side effect only. Pydantic AI handles `None`-returning tools correctly.

### Startup wiring

In `agent/worker.py`, call `asyncio.run(init_db())` (or `await init_db()` if already in async context) before the LiveKit worker starts. This creates the table if it doesn't exist and is idempotent.

**Acceptance:** After a session, `sqlite3 vidur.db "SELECT * FROM progress"` shows a row for each topic the agent logged.

---

## Files created/modified this scope

```
agent/
  brain.py                     NEW — Pydantic AI agent + tool registration
  session.py                   REWRITE — PydanticAILLM wrapper, tool dispatch
  worker.py                    MINOR — add init_db() call at startup
  tools/
    retrieve.py                NEW
    quiz.py                    NEW
    progress.py                NEW
  providers/
    stt/__init__.py            FILL — registry + get_stt_provider()
    stt/deepgram.py            NEW — stub
    stt/whisper.py             NEW — stub
    stt/google.py              NEW — stub
    tts/__init__.py            FILL — registry + get_tts_provider()
    tts/cartesia.py            NEW — stub
    tts/elevenlabs.py          NEW — stub
    tts/piper.py               NEW — stub
api/
  db.py                        NEW — progress table + init_db / log_progress
pyproject.toml                 EDIT — add pydantic-ai[google], aiosqlite
```

---

## Done when (2.1–2.6)

- [ ] `STT_ENGINE=sarvam` (default) still works end-to-end in a live session
- [ ] `STT_ENGINE=deepgram` imports without error, raises `NotImplementedError` only when called
- [ ] Same for TTS registry
- [ ] Agent responds using Pydantic AI, not raw Gemini
- [ ] Asking for a quiz causes `quiz_tool` to fire and the agent speaks a question
- [ ] `progress_tool` writes a row to `vidur.db`
- [ ] `retrieve_tool` falls back cleanly with a placeholder string (no crash)
- [ ] `uv run ruff check .` passes

---

## Scope boundary — do NOT build in this pass

- Any file under `rag/` (comes in 2.7–2.9)
- Langfuse span wiring (2.10)
- `sessions` or `turns` tables (2.11)
- FastAPI or any HTTP endpoints (2.12)
- Full implementations for Deepgram, Whisper, Google, Cartesia, ElevenLabs, Piper
