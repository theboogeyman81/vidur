# Phase 7 — `chore/ship`

**Goal:** Everything demoable, documented, and deployed. FINDINGS.md with real numbers. README with an architecture diagram. Loom script written. Ship it.

---

## Features

### 7.1 — FINDINGS.md writeup
- File: `docs/FINDINGS.md`
- Sections:
  - **STT Leaderboard** — final WER/CER table, which engine won and why
  - **TTS Leaderboard** — final TTFB table, which engine won and why
  - **Latency Breakdown** — where the e2e budget goes (p50 and p95 per segment), did we hit the 1200ms p95 target?
  - **Surprises** — things that didn't behave as expected (e.g. Whisper quality vs latency tradeoff, Sarvam on pure English, code-mix degradation on fine-tuned Piper)
  - **What I'd do with more time** — honest, brief
- No fluff. Numbers + one-sentence interpretation per finding.

### 7.2 — README
- File: `README.md`
- Architecture diagram (Mermaid or ASCII — either works, must render on GitHub)
- What the project is (two sentences)
- How to run it locally: `cp .env.example .env`, fill keys, `make dev`
- How to run the evals: `make eval-stt`, `make eval-tts`
- Link to FINDINGS.md
- Link to the live demo (Fly.io + Vercel URLs)
- No tutorial prose, no wall of text

### 7.3 — Makefile targets (final)
- File: `Makefile`
- `make dev` — start agent + API + web in parallel (use `concurrently` or `foreman`)
- `make dev-agent` — agent worker only
- `make dev-api` — FastAPI only
- `make dev-web` — Vite only
- `make eval-stt` — run STT eval across all engines
- `make eval-tts` — run TTS eval across all engines
- `make lint` — ruff check
- `make deploy` — Fly.io deploy (agent + API) + Vercel deploy (web)

### 7.4 — Deploy: Fly.io (agent + API)
- `fly.toml` already exists from Phase 1 — update if needed
- Confirm: agent worker and FastAPI run in the same Fly app (two processes) or separate apps
- Set all secrets via `fly secrets set KEY=value` — document which secrets in README
- Health check: `GET /health` returns 200

### 7.5 — Deploy: Vercel (frontend)
- `vercel.json` config
- Set `VITE_API_URL` env var in Vercel dashboard pointing to Fly.io FastAPI URL
- Confirm the A/B audio player and leaderboard load against the deployed API

### 7.6 — .env.example (final)
- Update `.env.example` with every key used across all phases:
  - `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
  - `SARVAM_API_KEY`
  - `GEMINI_API_KEY`
  - `DEEPGRAM_API_KEY`
  - `GOOGLE_APPLICATION_CREDENTIALS`
  - `CARTESIA_API_KEY`
  - `ELEVENLABS_API_KEY`
  - `VOYAGE_API_KEY`
  - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
  - `STT_ENGINE`, `TTS_ENGINE` (with defaults noted)

### 7.7 — Loom script
- File: `docs/loom_script.md`
- 6-minute script, timed per section:
  - **0:00–1:00** — live Hinglish conversation with the tutor, including one interruption
  - **1:00–2:00** — trigger a tool call mid-conversation (quiz or retrieve), show it in the Langfuse trace
  - **2:00–3:30** — STT leaderboard: walk through the WER table, point out the winner and the surprise
  - **3:30–5:00** — A/B TTS: play the same sentence through two engines, the one that's wrong is audible
  - **5:00–6:00** — latency waterfall for a real turn, show where the time goes, note if 1200ms was hit
- Write the script as if talking, not reading

### 7.8 — Final smoke test checklist
- [ ] Agent connects to LiveKit room on Fly.io
- [ ] Barge-in cancels TTS within 300ms
- [ ] Tool call fires and appears in Langfuse
- [ ] RAG returns relevant NCERT content for a curriculum question
- [ ] STT eval runs end-to-end with no crashes
- [ ] TTS eval produces audio files for all four engines
- [ ] Dashboard loads on Vercel, leaderboard shows real numbers
- [ ] A/B audio player plays two engines for the same sentence
- [ ] Latency waterfall renders for a real session

---

## Done when

- All five Loom demo moments are demoable without rehearsal failures
- `README.md` is clear enough for someone who wasn't there to run the project
- `docs/FINDINGS.md` has real numbers, not placeholders
- Live deploy is stable (not just working locally)
- Every secret is documented in `.env.example`

---

## Files created this phase

```
README.md
docs/
  FINDINGS.md       (completed with real numbers)
  loom_script.md
Makefile             (finalized)
vercel.json
fly.toml             (updated if needed)
.env.example         (finalized)
```
