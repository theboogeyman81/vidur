# Phase 6 — `feat/dashboard`

**Goal:** A React frontend with two pages — a live session view and an evals leaderboard. Must show real data from the API, not mocks. The evals page is the centrepiece.

---

## Features

### 6.1 — FastAPI evals endpoint
- File: `api/routes/evals.py`
- `GET /evals/stt` — returns latest STT results.json parsed into a response model
- `GET /evals/tts` — returns latest TTS results.json
- `GET /evals/tts/audio/{engine}/{sentence_id}` — serves the saved WAV file
- Response models: Pydantic, typed. No raw dicts.
- Register routes in `api/main.py`

### 6.2 — FastAPI sessions/traces endpoint
- File: `api/routes/traces.py`
- `GET /sessions` — list recent sessions from SQLite
- `GET /sessions/{id}/turns` — return all turn traces for a session
- Turn trace includes all logged fields: `stt_ms`, `llm_first_token_ms`, `tts_ttfb_ms`, `e2e_ms`, `tools_called[]`, etc.

### 6.3 — Vite + React + Tailwind project setup
- Directory: `web/`
- `vite.config.ts`, `tsconfig.json`, `tailwind.config.ts`, `postcss.config.js`
- Proxy `/api` to FastAPI in dev (`vite.config.ts` server.proxy)
- Two routes: `/session` and `/evals`
- No auth, no router guards — this is a demo app

### 6.4 — Session page (live voice UI)
- File: `web/src/pages/Session.tsx`
- "Join Session" button → calls `GET /token` → connects to LiveKit room via LiveKit JS SDK
- Shows live transcript as turns complete (polling `GET /sessions/{id}/turns` every 2s)
- Shows tool calls that fired: small badge per turn (e.g. "retrieve", "quiz")
- Shows per-turn latency: `e2e_ms` displayed next to each turn
- Barge-in indicator: turn row highlighted if `interrupted: true`
- No streaming partial transcripts — wait for full turn

### 6.5 — Evals page — STT leaderboard
- File: `web/src/pages/Evals.tsx`
- Table: Engine | WER | CER | p50 latency | p95 latency
- Sorted by WER ascending by default
- Each row expandable: shows per-clip WER breakdown
- Data from `GET /evals/stt`

### 6.6 — Evals page — TTS leaderboard
- Same page (`Evals.tsx`), tabbed or sectioned below STT
- Table: Engine | p50 TTFB | p95 TTFB | p50 total | p95 total
- Data from `GET /evals/tts`

### 6.7 — Evals page — A/B audio player
- Same page, below the TTS table
- Sentence selector: pick from `sentences.jsonl` list
- Two audio players side by side: Engine A vs Engine B
- Both load from `GET /evals/tts/audio/{engine}/{sentence_id}`
- Play buttons are independent; user listens and compares
- This is the centrepiece of the demo — make it obvious and easy to use

### 6.8 — Evals page — latency waterfall
- Same page, separate section
- Pick a session from a dropdown
- Render a horizontal stacked bar per turn: `stt_ms` | `llm_ms` | `tts_ttfb_ms`
- Each segment a different colour, labelled
- Shows where the latency budget goes visually
- Data from `GET /sessions/{id}/turns`

### 6.9 — CORS + API wiring
- File: `api/main.py`
- Add `CORSMiddleware` allowing `localhost:5173` and the Vercel production URL
- All frontend fetches go through `/api` proxy in dev, direct URL in prod

---

## Done when

- `make dev-web` starts Vite at `localhost:5173`, `make dev-api` starts FastAPI at `localhost:8000`
- Session page connects to a LiveKit room and shows turns with latency badges
- Evals page shows STT and TTS leaderboard tables with real numbers from results.json
- A/B audio player works: pick a sentence, hear two engines, compare
- Latency waterfall renders for at least one real session

---

## Files created this phase

```
web/
  vite.config.ts
  tsconfig.json
  tailwind.config.ts
  postcss.config.js
  index.html
  src/
    main.tsx
    App.tsx
    pages/
      Session.tsx
      Evals.tsx
    components/
      AudioPlayer.tsx
      LatencyWaterfall.tsx
      LeaderboardTable.tsx
api/
  routes/
    evals.py
    traces.py
  main.py       (updated: CORS, new routes)
```
