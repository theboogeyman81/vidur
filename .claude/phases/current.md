# Vidur — Phase & Feature Tracker

Last updated: 2026-09-06
Current phase: Phase 1 — `feat/voice-loop`
Spec: `specs/phase_1_spec.md`

Status key: `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` skipped

---

## Phase 1 — `feat/voice-loop`
**Branch:** `feat/voice-loop`
**Status:** in progress
**Spec:** `specs/phase_1_spec.md`

- [x] 1.1 LiveKit agent worker
- [x] 1.2 Turn loop + interruption handling
- [x] 1.3 Sarvam STT adapter (Saarika)
- [x] 1.4 Provider base contracts
- [x] 1.5 Gemini LLM call
- [x] 1.6 Hardcoded tutor prompt
- [x] 1.7 Sarvam TTS adapter (Bulbul)
- [x] 1.8 Per-turn trace logging (stub)
- [x] 1.9 Project scaffolding
- [ ] 1.10 Deploy to Fly.io

**Done when:** Real person speaks Hinglish in a LiveKit room and hears a Socratic response. Barge-in works. JSON trace logged per turn. Worker live on Fly.io.

---

## Phase 2 — `feat/tools-and-rag`
**Branch:** `feat/tools-and-rag`
**Status:** not started
**Blocked by:** Phase 1 merged and talking

- [ ] 2.1 Provider adapter layer (STT)
- [ ] 2.2 Provider adapter layer (TTS)
- [ ] 2.3 Pydantic AI agent
- [ ] 2.4 Tool: RAG lookup (`retrieve`)
- [ ] 2.5 Tool: Quiz generator (`quiz`)
- [ ] 2.6 Tool: Progress logger (`progress`)
- [ ] 2.7 NCERT content ingestion
- [ ] 2.8 Hybrid retriever
- [ ] 2.9 LangGraph corrective-RAG graph
- [ ] 2.10 Langfuse tracing
- [ ] 2.11 SQLite schema
- [ ] 2.12 FastAPI token endpoint

**Done when:** All three tools fire mid-conversation. Every turn has a Langfuse trace. STT/TTS engine swappable via env var.

---

## Phase 3 — `feat/stt-eval`
**Branch:** `feat/stt-eval`
**Status:** not started
**Blocked by:** Phase 2 merged

- [ ] 3.1 STT eval dataset (50–100 clips)
- [ ] 3.2 WER / CER metrics (`jiwer`)
- [ ] 3.3 Deepgram Nova adapter (full impl)
- [ ] 3.4 Whisper large-v3 adapter (full impl)
- [ ] 3.5 Google STT adapter (full impl)
- [ ] 3.6 STT eval runner
- [ ] 3.7 Latency metrics helper
- [ ] 3.8 Results summary printer

**Done when:** `run_stt_eval.py` completes across 4 engines. WER/CER/p50/p95 table committed to FINDINGS.md.

---

## Phase 4 — `feat/tts-eval`
**Branch:** `feat/tts-eval`
**Status:** not started
**Blocked by:** Phase 3 merged

- [ ] 4.1 TTS eval dataset (50 sentences)
- [ ] 4.2 Cartesia Sonic adapter (full impl)
- [ ] 4.3 ElevenLabs Flash adapter (full impl)
- [ ] 4.4 Piper adapter (full impl)
- [ ] 4.5 TTS eval runner
- [ ] 4.6 Audio artifact storage
- [ ] 4.7 Barge-in success measurement (live agent)
- [ ] 4.8 Results summary printer

**Done when:** `run_tts_eval.py` completes across 4 engines. Audio files saved. TTFB p50/p95 table committed. Barge-in latency logged per turn.

---

## Phase 5 — `feat/piper-voice`
**Branch:** `feat/piper-voice`
**Status:** not started
**Blocked by:** Phase 4 merged

- [ ] 5.1 Training data prep (Common Voice en-IN)
- [ ] 5.2 Piper fine-tune (Colab, ~2000 steps)
- [ ] 5.3 Fallback: F5-TTS zero-shot cloning (if fine-tune stalls)
- [ ] 5.4 ONNX model packaging
- [ ] 5.5 Piper TTSProvider update
- [ ] 5.6 Add fine-tuned voice to TTS leaderboard
- [ ] 5.7 Eval: code-mixing degradation on fine-tuned voice

**Done when:** Fine-tuned (or F5 fallback) Piper model runs via TTSProvider. New row in leaderboard. Code-mix degradation documented honestly.

---

## Phase 6 — `feat/dashboard`
**Branch:** `feat/dashboard`
**Status:** not started
**Blocked by:** Phase 5 merged

- [ ] 6.1 FastAPI evals endpoint
- [ ] 6.2 FastAPI sessions/traces endpoint
- [ ] 6.3 Vite + React + Tailwind setup
- [ ] 6.4 Session page (live voice UI)
- [ ] 6.5 Evals page — STT leaderboard
- [ ] 6.6 Evals page — TTS leaderboard
- [ ] 6.7 Evals page — A/B audio player
- [ ] 6.8 Evals page — latency waterfall
- [ ] 6.9 CORS + API wiring

**Done when:** Session page shows live turns with latency badges. Evals page shows real leaderboard numbers. A/B audio player works. Latency waterfall renders from a real session.

---

## Phase 7 — `chore/ship`
**Branch:** `chore/ship`
**Status:** not started
**Blocked by:** Phase 6 merged

- [ ] 7.1 FINDINGS.md writeup (real numbers, no placeholders)
- [ ] 7.2 README with architecture diagram
- [ ] 7.3 Makefile targets (final)
- [ ] 7.4 Deploy: Fly.io (agent + API)
- [ ] 7.5 Deploy: Vercel (frontend)
- [ ] 7.6 .env.example (final, all keys)
- [ ] 7.7 Loom script
- [ ] 7.8 Final smoke test checklist

**Done when:** All 5 Loom demo moments are demoable. Live deploy stable. FINDINGS.md has real numbers. README is self-sufficient.

---

## Progress summary

| Phase | Features | Done | Status |
|-------|----------|------|--------|
| 1 — voice-loop | 10 | 9 | in progress (deploy pending) |
| 2 — tools-and-rag | 12 | 0 | not started |
| 3 — stt-eval | 8 | 0 | not started |
| 4 — tts-eval | 8 | 0 | not started |
| 5 — piper-voice | 7 | 0 | not started |
| 6 — dashboard | 9 | 0 | not started |
| 7 — ship | 8 | 0 | not started |
| **Total** | **62** | **0** | |
