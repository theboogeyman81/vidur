# Phase 5 — `feat/tts-eval`

**Branch:** `feat/tts-eval` (cut from `main` after Phase 4 merges)
**Status:** not started
**Blocked by:** Phase 4 merged. Its code is done, but the clips and the full run are still outstanding. Phase 5 also *depends* on Phase 4's result, because the round-trip judge (D6) is the Phase 4 STT winner.
**Goal:** Four working TTS adapters, a code-mixed sentence set, and a table giving TTFB p50/p95, total p50/p95 and a round-trip intelligibility score for each engine, with audio saved for every engine × sentence pair. The live agent also has to log barge-in latency and success on every interrupted turn. This covers Loom demo moment #5 ("one is audibly wrong") and the interruption half of moment #1.

> TTFB alone can't show that an engine is "audibly wrong". A fast engine that says "Article" as "अर्टिकल-ए" wins a latency table and still loses the demo. This phase measures what the audio *says* as well as how quickly it arrives.

---

## Context: what earlier phases left you

- `agent/providers/base.py` defines the `TTSProvider` Protocol and `TTSResult` (`audio`, `time_to_first_byte_ms`, `total_ms`)
- `agent/providers/tts/__init__.py` has the registry `{sarvam, cartesia, elevenlabs, piper}`, and `get_tts_provider()` reads `TTS_ENGINE`
- `sarvam.py` contains two classes:
  - `SarvamTTS` is the harness adapter. It is non-streaming REST (`bulbul:v3`, speaker `kavya`), so **TTFB == total**. It opens a new `httpx.AsyncClient` on every call
  - `SarvamLKTTS` is the LiveKit plugin used live. It **pushes no audio until every sentence chunk has been synthesized**, but it records `last_ttfb_ms` at the first chunk. The logged `tts_ttfb_ms` therefore understates when audio actually starts (see 5.9)
- `cartesia.py`, `elevenlabs.py` and `piper.py` are stubs that raise `NotImplementedError`
- `agent/session.py` hardcodes `"interrupted": False` on every turn. Nothing listens for interruptions
- Phase 4 left `evals/metrics/wer.py` (`normalize`, `corpus_wer`, `romanize`) and `evals/metrics/latency.py` (`percentile`). `run_stt_eval.py` is the pattern to copy: Pydantic result models, one warm-up call, sequential calls, results opened in `"x"` mode, and a summary printer
- LiveKit Agents is **1.8.0**. Interruption is configured through `TurnHandlingOptions` → `InterruptionOptions`, where the **default `min_duration` is 0.5 s** (see D11)
- `.gitignore` already ignores `*.wav`

---

## Decisions locked for this phase

These fix gaps in `phases/phase_5.md`. Each has one tradeoff, and each has a pick.

| # | Decision | Tradeoff |
|---|---|---|
| D1 | **Features are numbered 5.x.** phase_5.md numbers them 4.x, which is a copy-paste slip. | None. Fix phase_5.md in the same PR |
| D2 | **Sentences are stored in mixed native script** (`आज हम Article 370 पढ़ेंगे`), the same convention as Phase 4 D1. **A 15-sentence subset also has `text_roman`** (`Aaj hum Article 370 padhenge`), and that version is synthesized as a separate `script: "roman"` bucket. | The roman bucket costs ~30% more characters. It exists because the tutor prompt says "code-switch freely" without pinning a script, so Gemini may emit either one. How TTS handles romanized Hindi decides whether the prompt should pin a script, which makes it a finding that changes code |
| D3 | **Sentences are tutor-side utterances**, meaning Socratic questions, short explanations and feedback. They are not student questions. TTS speaks the tutor's lines, so the eval set has to look like those lines. | They have to be hand-written or LLM-drafted and hand-edited, so they can't reuse the Phase 4 prompts |
| D4 | **All adapters return a complete WAV** (PCM16 mono) at the engine's native sample rate. Adapters that receive raw PCM wrap it in a WAV header. **Nothing is resampled.** | The files have different sample rates (22.05 or 24 kHz), and the A/B player handles that fine. Resampling would alter the audio being judged |
| D5 | **TTFB is the time from sending the request to receiving the first audio byte from the client**, measured with `perf_counter`. Streaming engines (Cartesia, ElevenLabs) use chunked HTTP streaming via `httpx.stream()`, and Piper reports its first yielded chunk. For non-streaming Sarvam, TTFB equals total, and it's marked `streaming: false` in `config` and footnoted in the table. | Sarvam looks worse on TTFB than it would with a streaming endpoint. That is still the honest result, because the agent runs this code today. If Sarvam's streaming (WebSocket) API supports `bulbul:v3`, add `sarvam-stream` as a **5th registry entry** (as `sarvam-codemix` was added in Phase 4) rather than swapping silently |
| D6 | **Intelligibility is measured by round-trip:** synthesize, transcribe with one fixed **judge STT**, then compute corpus WER against `text` using the Phase 4 normaliser. The judge is the Phase 4 winner. **If that winner is Sarvam, use the runner-up instead**, so that Sarvam STT isn't grading Sarvam TTS. | The judge's own errors add noise equally to every engine, so the ranking holds even if the absolute numbers inflate. Report the judge's clean-speech WER from Phase 4 next to it as the noise floor |
| D7 | **Entity hit rate.** Each sentence lists the English terms embedded in it (`["Article 370", "Preamble"]`). A hit means the romanized form of the entity appears in the romanized round-trip transcript. Report it per engine. | Substring matching is crude, and numbers are left out because "370" and "तीन सौ सत्तर" are both correct. It is still the column that names the "audibly wrong" engine |
| D8 | **Calls are sequential, with one discarded warm-up per engine. Each adapter holds one persistent `httpx.AsyncClient`** that is created lazily and reused. | This is slower, but concurrency or per-call TLS handshakes would contaminate TTFB. The existing `SarvamTTS` has to change to match |
| D9 | **A failed call is counted in `failures` and scores round-trip WER 1.0 and entity-miss for that sentence.** It doesn't crash the run. | Same as Phase 4 D8 |
| D10 | **Cartesia and ElevenLabs use raw `httpx` streaming, not their SDKs.** This follows the Phase 4 D6 style and means every hosted engine is timed the same way. | A warm WebSocket (the SDK path) could be ~50–100 ms faster on TTFB. The table compares like with like, and FINDINGS notes it |
| D11 | **Barge-in metrics use CLAUDE.md's field names, `interrupted` and `interruption_handled_ms`**, plus `barge_in_success` and `false_interruption`. phase_5.md's `barge_in_latency_ms` duplicates `interruption_handled_ms`, so drop it. | None. CLAUDE.md is the metric contract |
| D12 | **Barge-in is measured twice: once with LiveKit defaults (`min_duration=0.5`), then with `min_duration=0.3`.** The 300 ms target in phase_5.md **can't be reached with the defaults**, because LiveKit won't even register an interruption until 500 ms of speech. Lowering the threshold raises false interruptions from Hindi backchannels (`haan`, `hmm`, `achha`), so report the tradeoff instead of picking a number blind. | It takes two recording sessions instead of one. The latency-vs-false-interruption tradeoff is a better finding than a single number |
| D13 | **The audio artifacts are gitignored, with one exception: a curated demo set** (~5 sentences × 4 engines, ≈4 MB) is committed under `evals/results/audio_demo/`. | The repo grows a little, but Phase 7's A/B player and the Vercel deploy need files that exist in a clean clone |

---

## Features

### 5.1 — TTS eval dataset

**Dir:** `evals/datasets/tts_codemix/`

This is **50 sentences**, and every one is something the tutor could plausibly say.

**Buckets:**

| `mix` | Count | Example |
|---|---|---|
| `heavy` (Hinglish, English terms throughout) | 20 | `अच्छा, तो Fundamental Rights और Directive Principles में main difference क्या है?` |
| `light` (Hindi with 1–2 English terms) | 10 | `बहुत अच्छा, अब Preamble के बारे में सोचो` |
| `pure_hi` | 8 | `क्या तुम बता सकते हो कि संविधान सभा का गठन कब हुआ था?` |
| `pure_en` (Indian English register) | 7 | `Good. Now tell me, why do you think the framers chose a parliamentary system?` |
| `numeric` (text-normalisation stress) | 5 | `42nd Amendment 1976 में आया था, और इसने Preamble में तीन शब्द जोड़े` |

Spread the lengths across all buckets, with about a third in each: `short` (≤8 words), `medium` (9–20) and `long` (21–35). Put UPSC domain terms in on purpose (Lok Sabha, Article 21, RBI, GDP, Directive Principles), because these are where engines break.

`text_roman` is filled in for **15 sentences**: 10 `heavy` and 5 `light` (D2).

**`sentences.jsonl`**, one row per sentence:

```json
{"id": "s001", "text": "आज हम Article 370 के बारे में पढ़ेंगे", "text_roman": "Aaj hum Article 370 ke baare mein padhenge", "lang": "hi-en", "mix": "heavy", "length": "short", "entities": ["Article"]}
```

- `lang` is one of `hi-en`, `hi` or `en-IN`. The runner maps it to the BCP-47 code passed to `synthesize()`: `hi-en → hi-IN`, `hi → hi-IN` and `en-IN → en-IN`
- `entities` holds non-numeric English terms only (D7), and it can be empty
- Use a Pydantic `SentenceRow` model in `evals/datasets/tts_codemix/models.py`, which mirrors the STT `ManifestRow`

**Drafting.** Draft ~70 sentences with Gemini, using the tutor prompt as context and asking for bucketed output. Then hand-edit them down to 50. The hand-edit matters: fix Devanagari spelling, keep the English terms in Latin, and make sure the sentences sound like the tutor. Tag each one `"source": "llm-edited"` or `"source": "hand"`.

**`validate.py`** checks for unique ids, non-empty `text` and valid enum values. It also checks that every entity appears in `text` (and in `text_roman` when present). It prints the bucket counts.

---

### 5.2 — Cartesia Sonic adapter

**File:** `agent/providers/tts/cartesia.py`

- Sends `POST https://api.cartesia.ai/tts/bytes` through `client.stream(...)` (D10)
- Headers are `X-API-Key: $CARTESIA_API_KEY` and `Cartesia-Version: <current>`. **Verify the current version string and model id (`sonic-2` vs `sonic-3`) at impl time**
- Body: `model_id`, `transcript`, `voice: {mode: "id", id: $CARTESIA_VOICE_ID}`, `language` (`hi-IN → "hi"`, `en-IN → "en"`) and `output_format: {container: "raw", encoding: "pcm_s16le", sample_rate: 24000}`
- TTFB is taken at the first non-empty chunk from `aiter_bytes()`, and total at stream end. The PCM is wrapped as WAV (D4)
- Pick an Indian-accented Hindi voice with the same gender as Sarvam's `kavya`, and put its id and name in `config`
- `name = "cartesia-sonic"`. Use a 15 s timeout, and on a non-2xx status log the body and raise

### 5.3 — ElevenLabs Flash adapter

**File:** `agent/providers/tts/elevenlabs.py`

- Sends `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_24000` with `xi-api-key`
- Body: `model_id: "eleven_flash_v2_5"` and `text`. Pass `language_code` (`"hi"` / `"en"`) **only if Flash v2.5 honours it. Verify this.** If it doesn't, drop it and note that in `config`
- TTFB, WAV wrapping, voice choice and error handling work the same as in 5.2
- `name = "elevenlabs-flash"`

### 5.4 — Piper adapter

**File:** `agent/providers/tts/piper.py`

- Use the `piper-tts` Python package (`PiperVoice`), **not a subprocess**. It has fewer moving parts, and model load is easy to keep out of the timed region. The package is GPL-3, which is fine for this project; note it in the README
- The voice is a pre-trained **`hi_IN` medium voice from `rhasspy/piper-voices`**. Check which ones exist at impl time. The ONNX file and its JSON go in `models/piper/` (gitignored), and the path comes from `PIPER_MODEL_PATH`
- **Load the model lazily, once, and outside the timed region**
- `synthesize()` runs in `asyncio.to_thread`. Inside the thread, record `perf_counter` at the first yielded audio chunk (TTFB) and at the end (total). Concatenate the chunks and wrap them as WAV
- `name = "piper-hi"`. Put the voice name and `device: "cpu"` in `config`
- **Expect Latin-script English to come out badly**, since espeak's Hindi phonemizer has to guess at the English words. That is probably the "audibly wrong" engine for demo #5. Phase 6 exists to fix it, and this run is Phase 6's baseline

### 5.5 — TTS eval runner

**File:** `evals/run_tts_eval.py`

```
uv run python -m evals.run_tts_eval --engines sarvam,cartesia,elevenlabs,piper \
    [--limit N] [--mix heavy] [--script native|roman|both] [--judge deepgram] [--no-roundtrip]
```

The default is `--script both`, which synthesizes `text` for all 50 sentences and `text_roman` for the 15 that have it.

Flow:
1. Validate `sentences.jsonl`, and abort if it's invalid. Instantiate the judge STT from the STT registry and abort if that fails, unless `--no-roundtrip` is set
2. For each engine, sequentially:
   1. Instantiate it from the registry. If that fails, mark it `skipped` with the reason (the same as STT)
   2. Make one warm-up call and discard it (D8)
   3. For each sentence × script:
      1. `await provider.synthesize(text, bcp47)`
      2. Save the WAV (5.6)
      3. Read `duration_s` and `sample_rate` from the header
      4. Catch any exception as a failure (D9)
3. **Round-trip pass, after all synthesis is done.** Feed every saved WAV through the judge STT sequentially. This is separate so that judge latency never overlaps with TTS timing
4. Per sentence, store `id`, `script`, `mix`, `length`, `text`, `audio_path`, `ttfb_ms`, `total_ms`, `rtf` (total ÷ audio duration), `duration_s`, `sample_rate`, `roundtrip_hyp`, `roundtrip_wer`, `entities_hit` / `entities_total` and `error`
5. Aggregate per engine:
   - `p50_ttfb_ms`, `p95_ttfb_ms`, `p50_total_ms` and `p95_total_ms` (all on `native` only, so the numbers are comparable)
   - `roundtrip_wer` (corpus-level) and `entity_hit_rate`
   - `failures`
   - `by_mix`: round-trip WER per bucket
   - `by_script`: round-trip WER for `native` vs `roman` on the **same 15 sentences**, for a paired comparison
6. Write `evals/results/tts_{YYYYMMDDTHHMMSSZ}.json`, opened in `"x"` mode

**The results schema:**

```json
{
  "run_at": "2026-09-27T14:02:11+05:30",
  "git_sha": "abc1234",
  "dataset": {"n_sentences": 50, "n_roman": 15, "sentences_sha256": "…", "by_mix": {"heavy": 20, "…": 0}},
  "judge": {"engine": "deepgram-nova3", "phase4_clean_wer": 0.11},
  "engines": [
    {
      "engine": "cartesia-sonic",
      "status": "ok",
      "config": {"model_id": "sonic-2", "voice_id": "…", "voice_name": "…", "streaming": true},
      "p50_ttfb_ms": 140, "p95_ttfb_ms": 280, "p50_total_ms": 820, "p95_total_ms": 1400,
      "roundtrip_wer": 0.18, "entity_hit_rate": 0.86, "failures": 0,
      "by_mix": {"heavy": 0.21, "light": 0.15, "pure_hi": 0.12, "pure_en": 0.09, "numeric": 0.31},
      "by_script": {"native": 0.19, "roman": 0.34},
      "per_sentence": []
    }
  ]
}
```

Each adapter gets a `config: dict` class attribute, the same as the STT adapters. Use Pydantic models for every row.

**Shared helpers** go in `evals/metrics/audio.py`: `pcm16_to_wav(pcm, sr, channels=1) -> bytes` and `wav_info(wav) -> (sample_rate, duration_s)`. The adapters import `pcm16_to_wav` from `agent/providers/tts/_wav.py` instead, so that agent code never imports from `evals/`. Keep a single implementation in `_wav.py` and re-export it from `evals/metrics/audio.py`.

### 5.6 — Audio artifact storage

- The path is `evals/results/audio/{run_ts}/{engine}/{sentence_id}_{script}.wav`. A per-run directory is cleaner than a timestamp in every filename, and the results JSON stores `audio_path` relative to `evals/results/`
- The existing `*.wav` rule already gitignores these
- **Demo export (D13):** `evals/export_tts_demo.py <results.json> s003,s017,…` copies the listed sentences for every engine into `evals/results/audio_demo/{engine}/{id}_{script}.wav` and writes `audio_demo/index.json`, which records the sentence text, engine, TTFB and round-trip WER per file. Phase 7 reads that index
- Add `!evals/results/audio_demo/**/*.wav` to `.gitignore`
- Choose the demo sentences *after* the run. Use the ones where round-trip WER diverges most between engines, and include at least one `numeric` and one `roman` sentence

### 5.7 — Barge-in measurement (live agent)

**New file:** `agent/barge_in.py`. This is a pure, synchronous `BargeInTracker` that receives event timestamps and returns metrics. It has no LiveKit imports, so it can be tested without a room.

```python
class BargeInTracker:
    def on_agent_state(self, old: str, new: str, at: float) -> None: ...
    def on_user_state(self, old: str, new: str, at: float) -> None: ...
    def on_false_interruption(self, resumed: bool, at: float) -> None: ...
    def pop_turn_metrics(self) -> BargeInMetrics: ...   # resets for the next turn


class BargeInMetrics(BaseModel):
    interrupted: bool
    interruption_handled_ms: float | None   # cancel_at - detected_at
    barge_in_success: bool | None           # TTS stopped before the user stopped speaking
    false_interruption: bool
    false_interruption_resumed: bool | None
```

**Event mapping** in `session.py` (LiveKit 1.8). Use `ev.created_at` (wall clock) for every timestamp, so that all of them come from one clock:

| Moment | Event |
|---|---|
| `detected_at` | `user_state_changed` → `new_state == "speaking"` **while the agent state is `"speaking"`** |
| `cancel_at` | `agent_state_changed` → `old_state == "speaking"` and `new_state != "speaking"`, after a detection |
| `user_done_at` | `user_state_changed` → `speaking → listening` after a detection |
| false interruption | `agent_false_interruption` (with `resumed`) |

- `barge_in_success = cancel_at < user_done_at`
- If the user stops before any cancel, the result is `false` and `interruption_handled_ms` is `None`
- Replace the hardcoded `"interrupted": False` with the tracker's metrics, merged into `turn_data`. They then flow into SQLite and the Langfuse trace, where you add a `barge_in` span
- Put the active interruption config (`mode` and `min_duration`) in the trace as well. You can't compare the two runs in D12 without it
- Make `min_duration` configurable through the env var `VIDUR_MIN_INTERRUPTION_S` (default: unset, meaning LiveKit's default), passed via `TurnHandlingOptions`

**Caveats for FINDINGS:**
- `detected_at` is the time VAD fired, which lags the true onset of speech, so `interruption_handled_ms` is a lower bound on what the user experiences
- `cancel_at` is when the agent stopped playout. Buffering on the client side adds more

**Measurement protocol** (D12), run once per config, on **defaults** and on **`VIDUR_MIN_INTERRUPTION_S=0.3`**:
- 10 real interruptions mid-answer (`ruko`, `ek second`, `wait, that's wrong`)
- 10 backchannels mid-answer (`haan`, `hmm`, `achha`, `ok`). The correct behaviour here is **no** interruption

**Report script:** `evals/report_barge_in.py --sessions <room1>,<room2>` reads the turns from SQLite and prints the following per session:
- n interruptions
- p50/p95 `interruption_handled_ms`
- success rate
- false-interruption count

It also writes `evals/results/bargein_{ts}.json` (in `"x"` mode).

### 5.8 — Results summary

This is inline in `run_tts_eval.py`. After writing the JSON, print this table:

```
| Engine | p50 TTFB | p95 TTFB | p50 total | p95 total | RT-WER | RT-WER roman | entity hit | fails |
|--------|----------|----------|-----------|-----------|--------|--------------|------------|-------|
```

Then print the **5 worst sentences per engine by round-trip WER**, showing the text, the round-trip hypothesis and the audio path. Those are the files to listen to first.

**Listening pass.** Listen to the worst 5 for each engine, plus the demo candidates. Tag each file in `evals/datasets/tts_codemix/listening_notes.jsonl` with one of:
- `ok`
- `mispronounced_en`
- `mispronounced_hi`
- `read_roman_as_english`
- `number_misread`
- `robotic`

Add a one-line note. It takes about 30 minutes, and it's what keeps the FINDINGS bullets from being guesses.

Paste the tables into `docs/FINDINGS.md` under `## TTS Results`, together with the following:
- dataset composition
- the judge and its noise floor (D6)
- 3–5 findings: which engine mangles English terms, whether romanized input hurts (and so **whether the tutor prompt should pin a script**), and how numbers get read
- the caveats: Sarvam is non-streaming (D5), Piper is local CPU, the network path is from India, and HTTP streaming is used rather than WebSocket (D10)
- a `### Barge-in` subsection with the D12 table: config, p50/p95 `interruption_handled_ms`, success rate and false interruptions out of 10 backchannels

### 5.9 — Sarvam live-path fix (small, measurement correctness)

**File:** `agent/providers/tts/sarvam.py`

- In `_SarvamChunkedStream._run`, `push()` each chunk's PCM **as soon as it arrives** instead of after the loop. Today the logged `tts_ttfb_ms` is the time of the first chunk, but playback only starts after the *last* one, so the e2e numbers going into Phase 7's waterfall are wrong
- Reuse one `httpx.AsyncClient` per `SarvamLKTTS` instance rather than one per chunk
- This is about 10 lines. Measure the e2e p50 over ~10 turns before and after, and put one line about it in FINDINGS

---

## Build order within the phase

| Day | Work |
|---|---|
| 1 | Draft `sentences.jsonl` and hand-edit it, then write `validate.py`. Build the `_wav.py` helper with tests. Get API keys and pick voices |
| 2 | Build the Cartesia, ElevenLabs and Piper adapters and make the `SarvamTTS` client persistent. Smoke-test each with `--limit 3 --no-roundtrip`. Then build the runner and the round-trip pass |
| 3 | Build the `BargeInTracker` with tests, then wire up session.py and do the 5.9 fix. Run the two D12 recording sessions and `report_barge_in.py` |
| 4 | Full run, listening pass, demo export, FINDINGS, current.md and phase_5.md fixes, then the PR |

The long pole here is the listening pass and the barge-in sessions, not the code. If you're behind, cut the `roman` bucket before cutting the round-trip.

---

## Done when

1. `validate.py` passes on `sentences.jsonl` with ≥50 sentences covering all five `mix` buckets and 15 `text_roman` rows
2. `uv run python -m evals.run_tts_eval --engines sarvam,cartesia,elevenlabs,piper` completes, with failures counted rather than crashing the run
3. `evals/results/tts_{ts}.json` is committed with TTFB p50/p95, total p50/p95, round-trip WER, `by_script` and entity hit rate for all 4 engines
4. Audio exists for every engine × sentence × script, and `audio_demo/` plus `index.json` are committed
5. Interrupted turns in SQLite and Langfuse carry `interrupted: true`, `interruption_handled_ms`, `barge_in_success` and `false_interruption`
6. `bargein_{ts}.json` exists for both D12 configs, and the tradeoff table is in FINDINGS
7. `docs/FINDINGS.md` has `## TTS Results` and `### Barge-in`
8. The tests pass: `tests/evals/test_tts_audio.py` (WAV wrap/parse round-trip, lang mapping, entity matching) and `tests/agent/test_barge_in.py` (at least 5 cases: a clean interruption, cancel after the user finishes → failure, a backchannel with a false interruption, no interruption → `interrupted: false`, and state reset between turns)
9. `TTS_ENGINE=cartesia` works in the live agent. **This is optional:** only Sarvam has a LiveKit wrapper, and the others would need `livekit-plugins-cartesia` / `-elevenlabs`. Try it once and note the result. Don't build wrappers

---

## Files created or changed this phase

```
evals/
  datasets/tts_codemix/
    sentences.jsonl
    models.py
    validate.py
    listening_notes.jsonl
  metrics/audio.py              (re-exports agent/providers/tts/_wav.py)
  results/
    tts_{ts}.json
    bargein_{ts}.json
    audio/                      (gitignored)
    audio_demo/                 (committed, + index.json)
  run_tts_eval.py
  export_tts_demo.py
  report_barge_in.py
agent/
  barge_in.py                   (new)
  session.py                    (barge-in wiring, interruption config)
  providers/tts/
    _wav.py                     (new)
    cartesia.py                 (completed)
    elevenlabs.py               (completed)
    piper.py                    (completed)
    sarvam.py                   (persistent client, config attr, 5.9 fix)
tests/
  evals/test_tts_audio.py
  agent/test_barge_in.py
docs/FINDINGS.md                (TTS Results + Barge-in)
.env.example                    (CARTESIA_API_KEY, CARTESIA_VOICE_ID, ELEVENLABS_API_KEY,
                                 ELEVENLABS_VOICE_ID, PIPER_MODEL_PATH, VIDUR_MIN_INTERRUPTION_S)
.gitignore                      (audio_demo exception, models/piper/)
.claude/phases/phase_5.md       (renumber 4.x → 5.x)
```

---

## Dependency notes

Add `piper-tts` to `pyproject.toml` (GPL-3, local only). No other dependency is needed: Cartesia and ElevenLabs use the `httpx` that's already installed (D10), and `soundfile` came in with Phase 4.

## Budget

At an average of ~80 chars per sentence, one full run is about 50 × 80 + 15 × 80 ≈ **5.2k characters per engine**. The round-trip judge adds about 65 clips × ~4 s ≈ 4–5 min of STT audio per engine, which is about 18 minutes in total.

| Engine | Constraint |
|---|---|
| ElevenLabs | The free tier is ~10k chars/month, which is **one full run plus smoke tests**. Iterate with `--limit 3` and do the full run once. **Verify the current limit** |
| Cartesia | The free tier is ~20k credits/month, which covers about 3 full runs. Verify it |
| Sarvam | free credits |
| Piper | $0 (local) |
| Judge STT | Deepgram costs $0 against the signup credit. Google's 60 min/month would run out in about 3 runs, so **don't make Google the judge** |

The phase costs ~$0 if you stay inside the free tiers.

## Known risks

- **ElevenLabs quota.** A single mistaken full run eats the month. Consider making the runner print the total character count and ask for confirmation before any run over 2k chars per engine
- **Whether Piper has a `hi_IN` voice.** If no usable one exists, use an `en_US`/`en_GB` voice, label it honestly, and expect the Hindi to be unintelligible. That is still a valid baseline for Phase 6
- **The Phase 4 winner isn't known yet**, so the judge isn't either. Default to Deepgram if Phase 4 hasn't finished its full run, and record the choice in `judge`
- **Barge-in sessions are manual and noisy.** Twenty trials per config is a small sample, so report counts ("7/10 backchannels triggered a false interruption") rather than rates with fake precision
- **The `adaptive` interruption mode** may be auto-selected when available, and it behaves differently from pure VAD. Log whichever mode is active; if the two D12 runs end up in different modes, the comparison is invalid
