# Vidur — Phase & Feature Tracker

Last updated: 2026-09-23
Current phase: Phase 4 — `feat/stt-eval` (in progress; Phase 3 parked with eval numbers outstanding)
Spec: `.claude/specs/phase_4_spec.md`

Status key: `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` skipped/deferred

---

## Phase 1 — `feat/voice-loop`
**Branch:** `feat/voice-loop`
**Status:** done ✅
**Spec:** `specs/phase_1_spec.md`

- [x] 1.1 LiveKit agent worker
- [x] 1.2 Turn loop + interruption handling
- [x] 1.3 Sarvam STT adapter (Saarika)
- [x] 1.4 Provider base contracts
- [x] 1.5 Gemini LLM call (gemini-3.6-flash)
- [x] 1.6 Hardcoded tutor prompt
- [x] 1.7 Sarvam TTS adapter (Bulbul)
- [x] 1.8 Per-turn trace logging (stub)
- [x] 1.9 Project scaffolding
- [-] 1.10 Deploy to Fly.io (deferred to Phase 8 / chore/ship)

**Done when:** Real person speaks Hinglish in a LiveKit room and hears a Socratic response. Barge-in works. JSON trace logged per turn.

---

## Phase 2 — `feat/tools-and-baseline-rag`
**Branch:** `feat/tools-and-baseline-rag`
**Status:** done ✅
**Spec:** `specs/phase_2_spec_2.1-2.6.md`, `specs/phase_2_spec_2.7-2.12.md`

- [x] 2.1 Provider adapter layer (STT) — registry + sarvam/deepgram/whisper/google stubs
- [x] 2.2 Provider adapter layer (TTS) — registry + sarvam/cartesia/elevenlabs/piper stubs
- [x] 2.3 Pydantic AI agent (`agent/brain.py`) — GoogleModel + three tools registered
- [x] 2.4 Tool: RAG lookup (`retrieve`) — graceful fallback if Qdrant/Voyage unavailable
- [x] 2.5 Tool: Quiz generator (`quiz`) — Gemini generates Socratic question
- [x] 2.6 Tool: Progress logger (`progress`) — writes to SQLite
- [x] 2.7 NCERT content ingestion — `rag/loader.py` (PDF→chunks) + `rag/embedder.py` (Voyage+BM25→Qdrant)
- [x] 2.8 Hybrid retriever — `rag/retriever.py`, dense+sparse via RRF, returns `list[Chunk]`
- [-] 2.9 LangGraph corrective-RAG — deferred to Phase 3 (by design; baseline must exist first)
- [x] 2.10 Langfuse tracing — `_lf.trace()` per turn, stt/llm/tts spans, graceful no-op if keys missing
- [x] 2.11 SQLite schema — sessions + turns + progress tables, `log_session` / `log_turn` helpers
- [x] 2.12 FastAPI token endpoint — `GET /token` mints LiveKit JWT, CORS wired

**Done when:** All three tools fire mid-conversation. Every turn has a Langfuse trace. STT/TTS engine swappable via env var. Naive top-k retrieval in Qdrant working as baseline for Phase 3.

---

## Phase 3 — `feat/corrective-rag`
**Branch:** `feat/corrective-rag` (merged to main via PR #3 — code only, eval numbers still outstanding)
**Status:** [~] in progress
**Blocked by:** Phase 2 merged

- [ ] 3.1 RAG eval dataset — ~50 question + ground-truth-chunk pairs (`evals/datasets/rag_qa/`) — only `generate_dataset.py` exists, no `dataset.jsonl` yet
- [ ] 3.2 `run_rag_eval.py` baseline run — Ragas against naive top-k, commit `results_baseline.json` — blocked on 3.1
- [x] 3.3 LangGraph corrective-RAG graph (`rag/graph.py`) — `StateGraph` w/ conditional edge, `MemorySaver` checkpointer
- [x] 3.4 Node: retrieve (`rag/nodes/retrieve.py`)
- [x] 3.5 Node: grade (`rag/nodes/grade.py`) — LLM scores each chunk relevant/not
- [x] 3.6 Node: rewrite (`rag/nodes/rewrite.py`) — reformulate query, loop back (max 2)
- [x] 3.7 Node: generate (`rag/nodes/generate.py`) — compose grounded answer
- [x] 3.8 Bail-out logic — two-layer: pre-call check at 2400ms in `retrieve_node`, post-graph `asyncio.wait_for(timeout=2.5)` in `run_graph()`, sets `bailed_out: true`
- [x] 3.9 Wire graph into `retrieve` tool (replaces naive call) — `agent/tools/retrieve.py` calls `run_graph()`
- [ ] 3.10 `run_rag_eval.py` corrective run — Ragas against graph, commit `results_corrective.json` — blocked on 3.1/3.2
- [x] 3.11 Langfuse RAG spans — single `rag` span per tool call logs `rag_total_ms`, `rag_rewrites`, `rag_bailed_out`, `chunks_retrieved`, `chunks_passed_grading` (node-level spans not added — spec marks these optional)

**Done when:** Two `results.json` files exist — baseline vs corrective. Ragas numbers show the delta (positive or honest negative). Graph bails at 2500ms and logs it.

**Remaining blockers (as of 2026-09-23):**
- Qdrant now running locally via Docker, but only 1 NCERT chapter ingested (`jess402.pdf`, 24 chunks) — too thin for a ~50-question eval set, need 2–3 more chapters
- Voyage AI account has no payment method → 3 RPM / 10K TPM rate limit; either add a card (free tier still applies) or switch embeddings to local `fastembed` (deferred, discussed, not yet done)
- 3.1/3.2/3.10 need the above resolved before the eval can run

**Decision:** parking this here and moving to Phase 4 (`feat/stt-eval`) next; will come back and finish 3.1/3.2/3.10 after Phase 5.

---

## Phase 4 — `feat/stt-eval`
**Branch:** `feat/stt-eval`
**Status:** [~] in progress — code done, dataset + full run outstanding
**Spec:** `specs/phase_4_spec.md`

- [~] 4.1 STT eval dataset — tooling done (`record.py`, `import_cv.py`, `validate.py`, 108 prompts); clips not recorded yet
- [x] 4.2 WER / CER metrics (`evals/metrics/wer.py`) — mixed-script normaliser, corpus WER, `cer_roman` diagnostic, tested
- [x] 4.3 Deepgram Nova-3 adapter (`language=multi`) — untested live, needs `DEEPGRAM_API_KEY`
- [x] 4.4 Whisper large-v3 adapter (faster-whisper, CPU int8, forced `hi`) — smoke-tested with `tiny`
- [x] 4.5 Google STT adapter — Chirp 3 on Speech v2; untested live, needs GCP project + service account
- [x] 4.6 STT eval runner (`run_stt_eval.py`) — smoke-tested on 3 synthetic clips; + `sarvam-codemix` engine
- [x] 4.7 Latency metrics helper
- [x] 4.8 Results summary (table + worst-5 clips per engine)
- [ ] 4.9 Full run committed + FINDINGS.md `## STT Results`

**Done when:** `run_stt_eval.py` completes across 4 engines. WER/CER/p50/p95 table committed to `results.json` and FINDINGS.md.

---

## Phase 5 — `feat/tts-eval`
**Branch:** `feat/tts-eval` (stacked on `feat/stt-eval` — rebase onto `main` once Phase 4 merges)
**Status:** [~] in progress — code done, keys + full run + live sessions outstanding
**Spec:** `specs/phase_5_spec.md`

- [~] 5.1 TTS eval dataset — 50 sentences + 15 roman, `validate.py` passes; needs a hand-edit pass
- [x] 5.2 Cartesia Sonic adapter — sonic-3.6, chunked HTTP; untested live, needs `CARTESIA_API_KEY` + voice id
- [x] 5.3 ElevenLabs Flash adapter — flash v2.5, pcm_24000; untested live, needs key + voice id
- [x] 5.4 Piper adapter — `piper-tts`, `hi_IN-pratham-medium`; smoke-tested (~80ms TTFB)
- [x] 5.5 TTS eval runner — synth pass + round-trip judge pass; smoke-tested with sarvam + piper
- [x] 5.6 Audio artifact storage — `results/audio/{ts}/` gitignored, `export_tts_demo.py` → `audio_demo/`
- [~] 5.7 Barge-in measurement — `BargeInTracker` wired + tested, `report_barge_in.py`; D12 sessions not run
- [ ] 5.8 Results summary — printer done; full run, listening pass, FINDINGS numbers outstanding
- [x] 5.9 Sarvam live-path fix — audio pushed per chunk; first frame now matches logged TTFB

**Done when:** `run_tts_eval.py` completes across 4 engines. Audio files saved. TTFB p50/p95 + round-trip WER committed. Barge-in latency logged per turn, D12 tradeoff in FINDINGS.

---

## Phase 6 — `feat/piper-voice`
**Branch:** `feat/piper-voice`
**Status:** not started
**Blocked by:** Phase 5 merged

- [ ] 6.1 Training data prep (Common Voice en-IN)
- [ ] 6.2 Piper fine-tune (Colab, ~2000 steps)
- [ ] 6.3 Fallback: F5-TTS zero-shot cloning (if fine-tune stalls past 6h)
- [ ] 6.4 ONNX model packaging
- [ ] 6.5 Piper TTSProvider update
- [ ] 6.6 Add fine-tuned voice to TTS leaderboard
- [ ] 6.7 Eval: code-mixing degradation on fine-tuned voice

**Done when:** Fine-tuned (or F5 fallback) Piper model runs via TTSProvider. New row in leaderboard. Code-mix degradation documented honestly.

---

## Phase 7 — `feat/dashboard`
**Branch:** `feat/dashboard`
**Status:** not started
**Blocked by:** Phase 6 merged

- [ ] 7.1 FastAPI evals endpoint
- [ ] 7.2 FastAPI sessions/traces endpoint
- [ ] 7.3 Vite + React + Tailwind setup
- [ ] 7.4 Session page (live voice UI)
- [ ] 7.5 Evals page — STT leaderboard
- [ ] 7.6 Evals page — TTS leaderboard + A/B audio player
- [ ] 7.7 Evals page — RAG before/after numbers
- [ ] 7.8 Evals page — latency waterfall
- [ ] 7.9 CORS + API wiring

**Done when:** Session page shows live turns with latency badges. Evals page shows real leaderboard numbers. A/B audio player works. Latency waterfall renders from a real session.

---

## Phase 8 — `chore/ship`
**Branch:** `chore/ship`
**Status:** not started
**Blocked by:** Phase 7 merged

- [ ] 8.1 FINDINGS.md writeup (real numbers, no placeholders)
- [ ] 8.2 README with architecture diagram
- [ ] 8.3 Deploy: Fly.io (agent + API)
- [ ] 8.4 Deploy: Vercel (frontend)
- [ ] 8.5 `.env.example` final (all keys)
- [ ] 8.6 Loom script
- [ ] 8.7 Final smoke test checklist

**Done when:** All 6 Loom demo moments demoable. Live deploy stable. FINDINGS.md has real numbers. README is self-sufficient.

---

## Progress summary

| Phase | Features | Done | Skipped | Status |
|-------|----------|------|---------|--------|
| 1 — voice-loop | 10 | 9 | 1 | ✅ done |
| 2 — tools-and-baseline-rag | 12 | 11 | 1 (2.9→Ph3) | ✅ done |
| 3 — corrective-rag | 11 | 8 | 0 | in progress ⏸ (eval numbers pending) |
| 4 — stt-eval | 9 | 6 | 0 | in progress |
| 5 — tts-eval | 9 | 6 | 0 | in progress |
| 6 — piper-voice | 7 | 0 | 0 | not started |
| 7 — dashboard | 9 | 0 | 0 | not started |
| 8 — ship | 7 | 0 | 0 | not started |
| **Total** | **73** | **34** | **2** | |
