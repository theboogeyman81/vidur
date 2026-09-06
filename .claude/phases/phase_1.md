# Phase 1 — `feat/voice-loop`

**Goal:** A working end-to-end voice loop. User speaks → STT → LLM → TTS → user hears a response. Nothing else. Must be deployed and talking before Phase 2 starts.

---

## Features

### 1.1 — LiveKit agent worker
- File: `agent/worker.py`
- Register a LiveKit Agents worker that connects to a room
- Handle participant join/leave events
- Wire VAD (Voice Activity Detection) — use LiveKit's built-in silero VAD
- No custom logic yet; just prove the worker connects and stays alive

### 1.2 — Turn loop + interruption handling
- File: `agent/session.py`
- Listen for VAD end event → trigger STT
- Pipe STT result to LLM → stream response
- Pipe LLM tokens to TTS → stream audio back into the room
- Handle barge-in: if user speaks while TTS is playing, cancel the TTS stream immediately and restart the turn loop
- Log `interrupted: bool` and `interruption_handled_ms` per turn

### 1.3 — Sarvam STT adapter (Saarika)
- File: `agent/providers/stt/sarvam.py`
- Implement `STTProvider` protocol (defined in `agent/providers/base.py`)
- POST raw PCM audio to Sarvam `/speech-to-text` endpoint
- Return `STTResult(text, latency_ms, raw)`
- Use `SARVAM_API_KEY` from `.env`

### 1.4 — Provider base contracts
- File: `agent/providers/base.py`
- Define `STTProvider`, `STTResult`, `TTSProvider`, `TTSResult` as Pydantic models + Protocols
- This is the interface every future vendor must satisfy — get it right here

### 1.5 — Gemini LLM call
- File: `agent/session.py` (inline for now, extracted in Phase 2)
- Call Gemini 2.5 Flash with a hardcoded system prompt (file: `agent/prompts/tutor_v1.md`)
- Stream tokens back; feed first token time into `llm_first_token_ms`
- Use `GEMINI_API_KEY` from `.env`

### 1.6 — Hardcoded tutor prompt
- File: `agent/prompts/tutor_v1.md`
- Socratic voice tutor persona for UPSC prep
- Instructs: ask before telling, code-switch on request, keep turns short for voice
- No tool instructions yet

### 1.7 — Sarvam TTS adapter (Bulbul)
- File: `agent/providers/tts/sarvam.py`
- Implement `TTSProvider` protocol
- POST text to Sarvam `/text-to-speech`, stream audio bytes back
- Record `time_to_first_byte_ms` and `total_ms`
- Return `TTSResult`

### 1.8 — Per-turn trace logging (stub)
- File: `agent/session.py`
- Print a structured JSON line per turn to stdout with: `vad_end_ms`, `stt_ms`, `stt_engine`, `transcript`, `llm_first_token_ms`, `llm_total_ms`, `tts_ttfb_ms`, `tts_engine`, `e2e_ms`, `interrupted`
- No DB write yet — just stdout. Langfuse wiring comes in Phase 2.

### 1.9 — Project scaffolding
- `pyproject.toml` with `uv`, `ruff`, `python = ">=3.11"`
- `.env.example` with all required keys: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `SARVAM_API_KEY`, `GEMINI_API_KEY`
- `Makefile` targets: `make dev` (run agent locally), `make lint`

### 1.10 — Deploy to Fly.io
- `fly.toml` config
- Dockerfile for the agent worker
- Agent running live on Fly, connectable from a browser LiveKit room

---

## Done when

- A real person can open a LiveKit room, speak a sentence in Hindi or English, and hear a Socratic response within ~2 seconds
- Barge-in cancels the TTS stream visibly (no audio overlap)
- Stdout shows a JSON trace line for every turn
- Worker is deployed on Fly.io and stays alive

---

## Files created this phase

```
agent/
  worker.py
  session.py
  prompts/tutor_v1.md
  providers/
    base.py
    stt/sarvam.py
    tts/sarvam.py
pyproject.toml
.env.example
Makefile
fly.toml
Dockerfile
```
