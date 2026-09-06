# CLAUDE.md — Vidur

Hinglish voice tutor + the eval harness that proves which STT/TTS stack actually works for Indian-accented code-mixed speech.

Built as proof-of-work for an AI voice agent internship. Timeline: 7 days. Optimize for shippable and measurable, not complete.

Named after Vidur — the counsellor who asked hard questions instead of handing over answers. Use `vidur` (lowercase) for package names, module paths and the CLI; "Vidur" in prose and UI copy.

---

## The point of this project

Anyone can wire up a LiveKit voice agent. The differentiator here is **half 2** — an offline eval harness that benchmarks four STT engines and four TTS engines on Hinglish code-mixed speech, with real WER numbers and side-by-side audio.

If you have to cut scope, cut from the agent, not the harness.

---

## Two halves

**Half 1 — the agent.** Real-time voice tutor for exam prep (UPSC-flavoured). Socratic: asks before it tells. Handles interruption. Code-switches between English, Hindi and Hinglish on request. Calls tools.

**Half 2 — the harness.** Offline test suite that scores STT and TTS engines on a curated Hinglish eval set, plus per-turn latency traces from live sessions. Renders as a dashboard.

---

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Voice orchestration | LiveKit Agents (Python) | handles VAD, turn detection, barge-in, WebRTC |
| STT (default) | Sarvam Saarika | Indian-language native; the others are for the leaderboard |
| STT (benchmarked) | Deepgram Nova, Whisper large-v3, Google STT | |
| LLM | Gemini 2.5 Flash | picked for first-token latency, not quality |
| TTS (default) | Sarvam Bulbul | |
| TTS (benchmarked) | Cartesia Sonic, ElevenLabs Flash, self-hosted Piper | |
| Agent framework (voice path) | Pydantic AI | tool calling + typed outputs; kept thin for latency |
| RAG orchestration | LangGraph | corrective-RAG graph, runs *outside* the hot path |
| Vector DB | Qdrant | local Docker for dev; hybrid dense + sparse |
| Embeddings | Voyage AI | |
| Tracing | Langfuse | live turn traces, cost, spans |
| Eval (retrieval) | Ragas | context precision/recall, faithfulness, answer relevancy |
| Eval (STT) | jiwer | WER / CER |
| Backend | FastAPI | |
| Frontend | React + TypeScript + Vite + Tailwind | |
| Deploy | Fly.io (backend), Vercel (frontend) | |

Budget ceiling: ~$10 for the whole build. Prefer free tiers and local models. If a choice costs money and a local option is within 20% on quality, take the local option and note the tradeoff.

---

## Repo layout

```
vidur/
├── agent/                  # LiveKit voice agent
│   ├── worker.py           # entrypoint — LiveKit agent worker
│   ├── session.py          # turn loop, interruption handling
│   ├── prompts/            # system prompts, versioned as files not strings
│   ├── tools/              # tool-calling functions
│   │   ├── retrieve.py     # RAG lookup
│   │   ├── quiz.py         # generate a question on a topic
│   │   └── progress.py     # log what the student covered
│   └── providers/          # thin adapters — one interface per STT/TTS vendor
│       ├── stt/
│       └── tts/
├── rag/                    # ingestion + LangGraph corrective-RAG
│   ├── loader.py           # NCERT chapter subset → chunks
│   ├── embedder.py
│   ├── retriever.py        # hybrid dense + sparse search over Qdrant
│   ├── graph.py            # LangGraph StateGraph — the corrective loop
│   └── nodes/
│       ├── retrieve.py     # pull top-k
│       ├── grade.py        # LLM grades each chunk relevant / not
│       ├── rewrite.py      # reformulate the query, loop back
│       └── generate.py     # compose the grounded answer
├── evals/                  # THE IMPORTANT PART
│   ├── datasets/
│   │   ├── stt_hinglish/   # ~100 audio clips + ground-truth transcripts
│   │   ├── tts_codemix/    # ~50 code-mixed sentences
│   │   └── rag_qa/         # ~50 question + ground-truth-chunk pairs
│   ├── run_stt_eval.py     # WER/CER across engines → results.json
│   ├── run_tts_eval.py     # synthesize + latency + store audio → results.json
│   ├── run_rag_eval.py     # Ragas, baseline vs corrective graph → results.json
│   ├── run_judge_eval.py   # LLM-as-judge on pedagogy rubric
│   └── metrics/            # wer.py, latency.py, rubric.py
├── api/                    # FastAPI
│   ├── main.py
│   ├── routes/             # /token, /sessions, /evals, /traces
│   └── db.py               # SQLite dev, Postgres prod
├── web/                    # React + TS
│   └── src/
│       ├── pages/Session.tsx     # live voice UI
│       └── pages/Evals.tsx       # leaderboard, waterfall, A/B audio
├── training/
│   └── piper_finetune.ipynb      # Colab notebook, Indian English voice
└── docs/
    └── FINDINGS.md         # the writeup — numbers, tradeoffs, what surprised me
```

---

## Provider adapter contract

Every STT and TTS vendor sits behind one interface. This is non-negotiable — the whole eval harness depends on swapping engines without touching agent code.

```python
class STTProvider(Protocol):
    name: str

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult: ...


class STTResult(BaseModel):
    text: str
    latency_ms: float
    raw: dict  # vendor response, kept for debugging


class TTSProvider(Protocol):
    name: str

    async def synthesize(self, text: str, lang: str) -> TTSResult: ...


class TTSResult(BaseModel):
    audio: bytes
    time_to_first_byte_ms: float
    total_ms: float
```

Adding a vendor = one new file in `providers/`, registered in a dict. Nothing else changes.

---

## Two orchestrators, on purpose

This project deliberately uses **two** agent frameworks. Do not consolidate them — the split is the design.

**Pydantic AI owns the voice hot path.** Hear → decide → maybe call a tool → speak. Single-turn ReAct loop, no cycles, no checkpointing. It stays thin because every abstraction here eats into a 1200ms p95 budget, and when a turn runs slow you want a stack trace, not a graph execution to trace.

**LangGraph owns the RAG pipeline.** When the `retrieve` tool fires, it invokes a compiled LangGraph `StateGraph` that runs a corrective loop:

```
retrieve → grade chunks → any relevant?
                          ├── yes → generate
                          └── no  → rewrite query → retrieve (max 2 retries)
```

That's a genuine cycle with conditional edges — the thing LangGraph is actually built for. Naive top-k retrieval on NCERT content returns confidently wrong chunks for ambiguous student questions ("explain articles" — which articles?); the grade-and-rewrite loop is what fixes it.

Rules:
- The graph is invoked **once per tool call**, never per turn. It is not in the audio path.
- Hard-cap the rewrite loop at 2 iterations. An unbounded cycle inside a voice conversation is a hang.
- If the graph exceeds 2500ms, bail out and return the best chunks you have. Log the bailout. A slightly worse answer beats dead air.
- Use LangGraph's checkpointer for RAG state within one lookup only. Cross-turn conversation memory stays in SQLite, owned by the agent.

---

## Metrics that must be logged

Every live turn writes a trace row:

- `vad_end_ms` → when the user stopped talking
- `stt_ms`, `stt_engine`, `transcript`
- `llm_first_token_ms`, `llm_total_ms`, `tools_called[]`
- `tts_ttfb_ms`, `tts_engine`
- `e2e_ms` — user stopped speaking → first audio byte out
- `interrupted: bool`, `interruption_handled_ms`

When the RAG graph runs, log additionally:

- `rag_total_ms`, `rag_rewrites` (0, 1 or 2), `rag_bailed_out: bool`
- `chunks_retrieved`, `chunks_passed_grading`
- node-level spans in Langfuse so the retrieve/grade/rewrite/generate breakdown is visible

Report p50 and p95, never just the mean. Voice UX lives and dies in the tail.

Target: e2e p95 under 1200ms. If you can't hit it, say so in FINDINGS.md with the breakdown showing where it goes.

---

## Build order — one branch per phase

Follow spec-driven, branch-per-feature. Do not start a phase until the previous one is merged and demoable.

**Phase 1 — `feat/voice-loop`**
LiveKit agent worker. Sarvam STT → Gemini → Sarvam TTS. Hardcoded prompt. No RAG, no tools. Deployed and talking. This must work end-to-end before anything else exists.

**Phase 2 — `feat/tools-and-baseline-rag`**
Provider adapter layer. Pydantic AI agent with the three tools. NCERT subset ingested to Qdrant. **Naive top-k retrieval only** — no graph yet. Langfuse tracing wired to every turn. This baseline exists so Phase 3 has something to beat.

**Phase 3 — `feat/corrective-rag`**
Build `rag_qa/` eval set first (~50 question + ground-truth-chunk pairs). Run `run_rag_eval.py` against the Phase 2 baseline and record the numbers. *Then* build the LangGraph corrective loop and re-run. Two `results.json` files, before and after. The delta is the deliverable — if the graph doesn't beat the baseline, say so in FINDINGS.md and keep the honest result.

**Phase 4 — `feat/stt-eval`**
Build the eval dataset first (record clips, write ground truth). Then `run_stt_eval.py`. Output a WER/CER table across four engines. Commit results.json.

**Phase 5 — `feat/tts-eval`**
Code-mixed sentence set. Synthesize across engines, save audio artifacts, measure TTFB. Add barge-in success measurement to the live agent.

**Phase 6 — `feat/piper-voice`**
Colab fine-tune, export ONNX, wrap as a TTSProvider, add to the leaderboard. If it stalls past 6 hours, fall back to F5-TTS zero-shot cloning and label it honestly as cloned, not trained.

**Phase 7 — `feat/dashboard`**
React app. Session page (live transcript + tool trace + RAG graph path taken) and Evals page (STT leaderboard, RAG before/after, latency waterfall, A/B audio player).

**Phase 8 — `chore/ship`**
FINDINGS.md, README with a real architecture diagram, deploy, Loom script.

---

## Conventions

- Python 3.11+, `uv` for deps, `ruff` for lint, type hints everywhere
- Pydantic models for every boundary — no raw dicts crossing module lines
- Prompts live in `prompts/*.md` as versioned files, never inline strings
- All secrets via `.env`, never committed; ship a `.env.example`
- `async` throughout the agent path — one blocking call ruins the latency budget
- Structured logging (`structlog`), JSON in prod
- Commit messages: `phase(scope): what changed`
- Every eval run writes a timestamped `results.json`. Never overwrite. The history is the story.

---

## Non-goals — do not build these

- Auth, user accounts, multi-tenancy
- Mobile app
- Knowledge graphs
- More than one subject of content
- Streaming partial transcripts to the UI
- Any Kubernetes, any microservices
- Retry/queue infrastructure beyond a simple exponential backoff
- LangGraph anywhere in the voice hot path — it stays inside the RAG tool only
- Multi-agent / supervisor graphs, subagents, human-in-the-loop interrupts
- Consolidating the two orchestrators into one "for consistency"

If a request would add one of these, push back and say why.

---

## Working with me

- I learn by building. Explain concepts in context of the code, not as lectures.
- Short answers. No preamble, no summary of what you just did unless I ask.
- Confirm the plan before writing a large chunk of code.
- If a design choice has a real tradeoff, name it in one line and pick one. Don't give me three options and ask.
- Tell me when I'm wrong. If a phase is scoped badly or a number looks off, say so.

---

## Definition of done

Someone can watch a 6-minute Loom that shows:

1. A live Hinglish conversation with the tutor, including an interruption
2. A tool call firing mid-conversation
3. The corrective-RAG graph rewriting a bad query and recovering, with the Ragas before/after numbers next to it
4. The STT leaderboard with real WER numbers
5. Two TTS engines saying the same code-mixed sentence, back to back, where one is audibly wrong
6. The latency waterfall for a real turn

If all five are demoable, ship it. Everything else is optional.
