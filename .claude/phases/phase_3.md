# Phase 3 — `feat/stt-eval`

**Goal:** Build the STT eval dataset, run all four engines against it, produce a WER/CER table with real numbers. This is the first half of the eval harness — the thing that makes the project worth showing.

---

## Features

### 3.1 — STT eval dataset
- Directory: `evals/datasets/stt_hinglish/`
- ~100 audio clips, 16kHz mono WAV
- Mix of: pure Hindi, pure English, Hinglish code-mixed, fast speech, accented speech
- Each clip paired with a ground-truth transcript in `manifest.jsonl`:
  `{"file": "clip_001.wav", "transcript": "aaj hum polity padh rahe hain", "lang": "hi-en"}`
- Source: record yourself, use Common Voice Hindi subset, or synthesize with Sarvam TTS then manually correct
- Minimum viable: 50 clips. Target: 100.

### 3.2 — WER / CER metrics
- File: `evals/metrics/wer.py`
- Wrap `jiwer` library
- `compute_wer(hypothesis: str, reference: str) -> float`
- `compute_cer(hypothesis: str, reference: str) -> float`
- Normalise before scoring: lowercase, strip punctuation, collapse whitespace
- Handle Hindi Devanagari and Latin script in the same batch

### 3.3 — Deepgram Nova adapter (full impl)
- File: `agent/providers/stt/deepgram.py`
- Complete the stub from Phase 2
- Use Deepgram's async Python SDK
- Enable `language: hi` and `code_switching: true` params
- Return `STTResult` with latency

### 3.4 — Whisper large-v3 adapter (full impl)
- File: `agent/providers/stt/whisper.py`
- Run locally via `faster-whisper` (CPU is fine for eval, not live)
- Model: `large-v3`, language: `None` (auto-detect)
- Return `STTResult`; latency will be high — that's expected and important to record

### 3.5 — Google STT adapter (full impl)
- File: `agent/providers/stt/google.py`
- Use `google-cloud-speech` async client
- Config: `BCP-47 hi-IN`, enhanced model, automatic punctuation off
- Return `STTResult`

### 3.6 — STT eval runner
- File: `evals/run_stt_eval.py`
- Load `manifest.jsonl`, iterate clips
- For each clip × each engine: call `provider.transcribe(audio, 16000)`
- Compute WER and CER against ground truth
- Write timestamped `evals/results/stt_{timestamp}.json`:
  ```json
  {
    "engine": "sarvam",
    "wer": 0.12,
    "cer": 0.08,
    "avg_latency_ms": 340,
    "p95_latency_ms": 610,
    "per_clip": [...]
  }
  ```
- Never overwrite previous results — append timestamp to filename
- CLI: `uv run python -m evals.run_stt_eval --engines sarvam,deepgram,whisper,google`

### 3.7 — Latency metrics helper
- File: `evals/metrics/latency.py`
- `percentile(values: list[float], p: int) -> float`
- Used by all eval runners — p50 and p95 reported, never just mean

### 3.8 — Results summary printer
- File: `evals/run_stt_eval.py` (inline)
- After eval completes, print a markdown table to stdout:
  ```
  | Engine   | WER  | CER  | p50 ms | p95 ms |
  |----------|------|------|--------|--------|
  | sarvam   | 0.12 | 0.08 | 310    | 590    |
  | deepgram | 0.18 | 0.11 | 280    | 480    |
  | whisper  | 0.09 | 0.06 | 4200   | 6100   |
  | google   | 0.21 | 0.14 | 390    | 720    |
  ```
- Commit the printed table + `results.json` together

---

## Done when

- `manifest.jsonl` exists with at least 50 clips
- `uv run python -m evals.run_stt_eval --engines sarvam,deepgram,whisper,google` runs to completion
- A `results/stt_{timestamp}.json` exists with WER, CER, p50, p95 for all four engines
- The markdown table is committed to `docs/FINDINGS.md` under "STT Results"

---

## Files created this phase

```
evals/
  datasets/stt_hinglish/
    manifest.jsonl
    clip_001.wav ... clip_NNN.wav
  results/
    stt_{timestamp}.json
  run_stt_eval.py
  metrics/
    wer.py
    latency.py
agent/providers/stt/
  deepgram.py     (completed)
  whisper.py      (completed)
  google.py       (completed)
```
