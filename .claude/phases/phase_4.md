# Phase 4 — `feat/tts-eval`

**Goal:** Benchmark four TTS engines on code-mixed sentences. Save audio artifacts. Measure TTFB. Add barge-in success rate to the live agent. Real numbers committed to results.json.

---

## Features

### 4.1 — TTS eval dataset
- Directory: `evals/datasets/tts_codemix/`
- ~50 sentences, varying in:
  - Code-mix ratio (pure Hindi → pure English → heavy Hinglish)
  - Length (short 5-word, medium 15-word, long 30-word)
  - Named entities: English technical terms embedded in Hindi sentences
- File: `sentences.jsonl`:
  `{"id": "s001", "text": "Aaj hum Article 370 ke baare mein padh rahe hain", "lang": "hi-en"}`
- No audio input needed — TTS is synthesis only

### 4.2 — Cartesia Sonic adapter (full impl)
- File: `agent/providers/tts/cartesia.py`
- Complete the stub from Phase 2
- Use Cartesia's streaming Python client
- Capture first audio chunk timestamp for `time_to_first_byte_ms`
- Return `TTSResult(audio, time_to_first_byte_ms, total_ms)`

### 4.3 — ElevenLabs Flash adapter (full impl)
- File: `agent/providers/tts/elevenlabs.py`
- Use ElevenLabs Flash model (lowest latency tier)
- Streaming: capture TTFB from first chunk
- Return `TTSResult`

### 4.4 — Piper adapter (full impl)
- File: `agent/providers/tts/piper.py`
- Run Piper locally via subprocess (ONNX model)
- Use a pre-trained English or Hindi voice — fine-tuned voice comes in Phase 5
- TTFB = time to first byte from subprocess stdout
- Return `TTSResult`

### 4.5 — TTS eval runner
- File: `evals/run_tts_eval.py`
- Load `sentences.jsonl`, iterate sentences
- For each sentence × each engine: call `provider.synthesize(text, lang)`
- Save audio artifact: `evals/results/audio/tts_{engine}_{sentence_id}.wav`
- Write timestamped `evals/results/tts_{timestamp}.json`:
  ```json
  {
    "engine": "sarvam",
    "avg_ttfb_ms": 210,
    "p95_ttfb_ms": 380,
    "avg_total_ms": 1100,
    "p95_total_ms": 1800,
    "per_sentence": [...]
  }
  ```
- CLI: `uv run python -m evals.run_tts_eval --engines sarvam,cartesia,elevenlabs,piper`

### 4.6 — Audio artifact storage
- Save WAV files under `evals/results/audio/`
- Filename pattern: `tts_{engine}_{sentence_id}_{timestamp}.wav`
- These are the files the dashboard's A/B audio player will serve
- Do not commit audio files to git — add `evals/results/audio/` to `.gitignore`

### 4.7 — Barge-in success measurement (live agent)
- File: `agent/session.py`
- When a barge-in occurs, record: `barge_in_detected_ms` (when VAD fired during TTS), `tts_cancelled_ms` (when audio stream stopped)
- `barge_in_latency_ms = tts_cancelled_ms - barge_in_detected_ms`
- Target: under 300ms. Log to Langfuse span.
- Add `barge_in_success: bool` — true if TTS stopped before the user finished their next utterance

### 4.8 — Results summary printer
- After eval completes, print markdown table to stdout:
  ```
  | Engine     | p50 TTFB | p95 TTFB | p50 total | p95 total |
  |------------|----------|----------|-----------|-----------|
  | sarvam     | 190      | 360      | 980       | 1600      |
  | cartesia   | 140      | 280      | 820       | 1400      |
  | elevenlabs | 220      | 410      | 1100      | 1900      |
  | piper      | 80       | 150      | 600       | 1100      |
  ```
- Commit table to `docs/FINDINGS.md` under "TTS Results"

---

## Done when

- `sentences.jsonl` exists with at least 50 code-mixed sentences
- `uv run python -m evals.run_tts_eval --engines sarvam,cartesia,elevenlabs,piper` completes
- Audio artifacts saved for every engine × sentence pair
- `results/tts_{timestamp}.json` committed with TTFB p50/p95 for all four engines
- Barge-in latency logged per turn in live agent

---

## Files created this phase

```
evals/
  datasets/tts_codemix/
    sentences.jsonl
  results/
    tts_{timestamp}.json
    audio/          (.gitignored)
  run_tts_eval.py
agent/providers/tts/
  cartesia.py     (completed)
  elevenlabs.py   (completed)
  piper.py        (completed)
```
