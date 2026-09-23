# Phase 4 — `feat/stt-eval`

**Branch:** `feat/stt-eval` (cut from `main`)
**Status:** not started
**Blocked by:** nothing. Phase 3 is parked by decision (see `phases/current.md`): graph code is merged, eval numbers still outstanding. Phase 4 touches no RAG code, so the two don't conflict.
**Goal:** A Hinglish STT eval set, four working STT adapters, and a WER/CER/p50/p95 table with real numbers. This is Loom demo moment #4 and the first half of the harness.

> Dataset first. Adapters that no clip has tested prove nothing.

---

## Context: what Phase 2 left you

- `agent/providers/base.py` has the `STTProvider` Protocol and the `STTResult` model (`text`, `latency_ms`, `raw`)
- `agent/providers/stt/__init__.py` has the registry `{sarvam, deepgram, whisper, google}` and `get_stt_provider()` reads `STT_ENGINE`
- `sarvam.py` works. It is a LiveKit `stt.STT` subclass that also exposes `transcribe(wav_bytes, sr)`, using model `saaras:v3` and `hi-IN`
- `deepgram.py`, `whisper.py` and `google.py` are stubs that raise `NotImplementedError`
- `evals/metrics/` and `evals/results/` don't exist yet
- `jiwer` isn't in `pyproject.toml` yet

---

## Decisions locked for this phase

These fix gaps in `phases/phase_4.md`. Each has one tradeoff, and each has a pick.

| # | Decision | Tradeoff |
|---|---|---|
| D1 | **The reference transcripts use mixed native script.** Hindi words are in Devanagari and English words in Latin: `आज हम polity पढ़ रहे हैं`. The Latin example in phase_4.md (`aaj hum polity...`) is wrong for scoring, because Sarvam, Google and Deepgram all emit Devanagari for Hindi, so a romanised reference would score every engine at ~100% WER. | An engine that writes English loanwords in Devanagari (`पॉलिटी`) is charged an error. That is a real downstream cost, since the LLM then sees an odd token. It is reported, not hidden (see D2). |
| D2 | **Primary metric is WER/CER on the native-script text. A secondary `cer_roman` is computed after both sides go through the same Devanagari→Latin transliterator.** | `cer_roman` is lossy (schwa, loanword spelling) and is a diagnostic only. It exposes script-choice errors: when `wer` is high and `cer_roman` is low, the engine heard it right but wrote it in the "wrong" script. |
| D3 | **Corpus-level WER, not the mean of per-clip WER.** Run `jiwer` over all refs and hyps of an engine at once. | Short clips stop dominating the number. Per-clip WER is still stored for drill-down. |
| D4 | **Whisper is forced to `language="hi"`, not auto-detect.** | Auto-detect on Hinglish flips per clip between Hindi, Urdu script and translating to English, which produces noise rather than a comparison. Forced `hi` is the fair best case. Note it in FINDINGS. |
| D5 | **Deepgram runs Nova-3 with `language=multi`.** phase_4.md says `language: hi` + `code_switching: true`, but `code_switching` isn't a Deepgram param as far as I know. Nova-3's `multi` mode is what does Hindi-English code-switching. **Verified 2026-09-23:** Hindi is in Nova-3's `multi` set. | If `multi` underperforms, add `deepgram-hi` (`language=hi`) as a 5th registry entry rather than swapping silently. |
| D6 | **Deepgram uses raw `httpx` against `/v1/listen`, not the SDK. Google uses the `google-cloud-speech` SDK.** | This matches the Sarvam adapter style and saves a dependency. Google's REST needs OAuth token plumbing, so there the SDK is the smaller surface. |
| D7 | **Calls are sequential, one clip at a time, with one discarded warm-up call per engine.** | The run is slower (~10 min total), but concurrency would contaminate latency, and latency is half the table. |
| D8 | **A failed call counts as an empty hypothesis (WER 1.0 for that clip) and is also counted in `failures`.** | This is honest. Dropping failures would flatter flaky engines. |
| D9 | **TTS-synthesised clips are excluded from the headline number.** phase_4.md lists "synthesize with Sarvam TTS" as a source. That is Sarvam grading its own homework on studio-clean audio. | If used at all, tag them `source: "synth"` and report them as a separate row. |
| D10 | **WAVs are committed to git** (~100 × 5s × 32 KB/s ≈ 16 MB). | The repo grows, but the eval is reproducible from a clean clone. Get consent from anyone whose voice is in there. |

---

## Build notes (2026-09-23)

- **Google → Chirp 3** (Speech v2, `us` multi-region, `hi-IN`) instead of v1 `latest_short`. It needs `GOOGLE_CLOUD_PROJECT` in addition to the credentials.
- **Added `sarvam-codemix`** (Saaras v3 `mode=codemix`). The default `transcribe` mode writes English in Devanagari. A 3-clip smoke test gave WER 0.48 for transcribe and 0.04 for codemix. The live agent still uses `transcribe`; switch it only if the full run agrees.
- **Runner takes `--manifest`**, so smoke tests can use a scratch dataset. `validate()` resolves WAVs relative to the manifest.

---

## Features

### 4.1 — STT eval dataset

**Dir:** `evals/datasets/stt_hinglish/`

Floor is 50 clips and the target is 100. Every clip is 16 kHz mono PCM16 WAV, 2–12 s long.

**Buckets** (target counts for 100; halve them for the 50 floor):

| `lang` | `condition` | Count | Example |
|---|---|---|---|
| `hi-en` | `clean` | 45 | "Article 370 ke baare mein thoda explain karo" |
| `hi` | `clean` | 15 | "मौलिक अधिकार कितने प्रकार के होते हैं" |
| `en-IN` | `clean` | 15 | "What is the difference between a bill and an act?" |
| `hi-en` | `fast` | 10 | the same kind of sentence, spoken quickly |
| `hi-en` | `noisy` | 15 | fan or traffic in the background, phone mic at arm's length |

Stay in domain: the utterances are what a UPSC student would actually say to the tutor, covering polity, history and "ek aur question do". Put domain terms in on purpose (Lok Sabha, Directive Principles, Preamble, 42nd Amendment), because these are exactly where engines break.

**Sources, in priority order:**
1. **You plus 2–3 other speakers** reading from `prompts.txt`. This is the main source and gives accent spread. Tag `speaker` with a short id.
2. **Common Voice Hindi** (CC0) for part of the `hi` bucket. It is read speech with verified transcripts. Tag `source: "cv"`. Resample to 16 kHz.
3. **A handful of spontaneous clips.** Speak freely, then transcribe by hand. Tag `condition: "spontaneous"`. It's optional, but it is the most honest bucket.

**`manifest.jsonl`**, one row per clip:

```json
{"file": "clip_001.wav", "transcript": "आज हम polity पढ़ रहे हैं", "lang": "hi-en", "condition": "clean", "speaker": "p1", "source": "self", "duration_s": 3.4}
```

**Transcript conventions.** Write these at the top of `evals/datasets/stt_hinglish/README.md` and apply them strictly:
- Hindi words are written in Devanagari and English words in Latin. Proper nouns follow the language of the word ("Lok Sabha" → `लोक सभा`, "Parliament" → `Parliament`)
- Numbers are written as digits (`370`, `42nd`). The normaliser handles the rest
- No punctuation is needed, because it gets stripped anyway
- Transcribe what was *said*, not what the prompt said. If the reading slipped, fix the transcript, not the audio

**Helper scripts:**
- `evals/datasets/stt_hinglish/prompts.txt` has one prompt per line, prefixed with a bucket tag: `hi-en|clean|Article 370 ke baare mein...`
- `evals/datasets/stt_hinglish/record.py` walks the prompts. For each one it shows the text, records with `sounddevice` until Enter, writes `clip_NNN.wav` at 16 kHz mono, and appends a manifest row. It supports `--speaker p2` and `--start-at N`. This script is the time-saver, since it makes 100 clips about an hour of work rather than an afternoon
- `evals/datasets/stt_hinglish/validate.py` checks that every manifest file exists, is 16 kHz mono, is 0.3–15 s long and has a non-empty transcript. It prints the bucket counts. Run it before every eval

---

### 4.2 — WER / CER metrics

**File:** `evals/metrics/wer.py`

```python
def normalize(text: str) -> str: ...
def compute_wer(hypothesis: str, reference: str) -> float: ...
def compute_cer(hypothesis: str, reference: str) -> float: ...
def corpus_wer(hyps: list[str], refs: list[str]) -> float: ...   # D3
def corpus_cer(hyps: list[str], refs: list[str]) -> float: ...
def romanize(text: str) -> str: ...                              # D2
```

**The normaliser** applies the same steps to both sides, in this order:
1. Unicode NFC
2. Lowercase the Latin characters (Devanagari has no case)
3. Fold nuktas (`क़→क`, `ज़→ज`, `फ़→फ`) and fold chandrabindu to anusvara (`ँ→ं`). Engines disagree on these constantly, and they aren't recognition errors
4. Strip punctuation, including the Devanagari danda `।` and `॥`, and zero-width joiners (`‌`, `‍`)
5. Collapse whitespace

`romanize()` runs `indic_transliteration.sanscript` Devanagari→ITRANS, then lowercases and strips anything that isn't alphanumeric. It is only used for `cer_roman`.

Edge case: an empty reference after normalisation is a manifest bug, so raise. An empty hypothesis is valid and gives WER 1.0.

**Tests:** `tests/evals/test_wer.py` needs at least 6 cases:
- identical strings give 0
- nukta variants give 0
- the danda is ignored
- Latin case is ignored
- one substituted word in 4 gives 0.25
- an empty hypothesis gives 1.0

---

### 4.3 — Deepgram Nova adapter

**File:** `agent/providers/stt/deepgram.py`

- Uses `POST https://api.deepgram.com/v1/listen?model=nova-3&language=multi&smart_format=false&punctuate=false` (D5, D6)
- Headers are `Authorization: Token $DEEPGRAM_API_KEY` and `Content-Type: audio/wav`, and the body is the raw WAV bytes
- `text` comes from `results.channels[0].alternatives[0].transcript`
- `latency_ms` is measured with `perf_counter` around the HTTP call only
- `raw` holds the full JSON
- `name = "deepgram-nova3"`
- Use a 15 s timeout. On a non-2xx status, log the body and raise; the runner catches it (D8)

### 4.4 — Whisper large-v3 adapter

**File:** `agent/providers/stt/whisper.py`

- Uses `faster-whisper` with `WhisperModel("large-v3", device="cpu", compute_type="int8")`
- **Load the model lazily, once, and outside the timed region.** The first load downloads ~3 GB
- Decode the WAV bytes to float32 numpy with `soundfile` and call `model.transcribe(arr, language="hi", beam_size=5, vad_filter=False)` (D4)
- `transcribe()` is sync CPU work, so wrap it in `asyncio.to_thread` to keep the adapter async-correct, even though it won't be used live
- `text` is the joined segment texts, `raw` is `{"segments": [...], "info": {...}}`, and `name = "whisper-large-v3"`
- Latency will be seconds, not milliseconds. That is the finding, so record it. Also note in FINDINGS that the run was local CPU on an M-series Mac, not a GPU

### 4.5 — Google STT adapter

**File:** `agent/providers/stt/google.py`

- Uses `google.cloud.speech_v1.SpeechAsyncClient` with `recognize()` (synchronous recognition is fine for clips under 60 s)
- Config: `encoding=LINEAR16`, `sample_rate_hertz=16000`, `language_code="hi-IN"`, `alternative_language_codes=["en-IN"]`, `model="latest_short"` and `enable_automatic_punctuation=False`
- "Enhanced model" in phase_4.md: `use_enhanced` only applies to specific en-US models. Use `latest_short` and note it. **Verify at impl time.** If Chirp (the v2 API) is available on the free tier for `hi-IN`, it is the stronger comparison. Pick one, and record which one in `name` (`google-latest-short` or `google-chirp`)
- Credentials come from `GOOGLE_APPLICATION_CREDENTIALS` (a service-account JSON path). This needs a GCP project with billing enabled, even though usage stays inside the free 60 min/month
- `raw` is `MessageToDict(response._pb)`

### 4.6 — STT eval runner

**File:** `evals/run_stt_eval.py`

```
uv run python -m evals.run_stt_eval --engines sarvam,deepgram,whisper,google [--limit N] [--lang hi-en]
```

Flow:
1. Run `validate.py` logic. If the manifest is bad, abort before spending any API calls
2. For each engine, sequentially:
   1. Instantiate it from the registry
   2. Make one warm-up call on clip 0 and discard it (D7)
   3. Loop through the clips: read the WAV bytes, `await provider.transcribe(wav, 16000)`, and catch exceptions as failures (D8)
3. Per clip, store `file`, `lang`, `condition`, `ref`, `hyp`, `wer`, `cer`, `cer_roman`, `latency_ms`, `rtf` (latency divided by audio duration) and `error`
4. Aggregate per engine:
   - `wer`, `cer` and `cer_roman` (corpus-level)
   - `p50_ms` and `p95_ms`
   - `failures`
   - `by_lang` and `by_condition`, each holding corpus WER per bucket. **This breakdown is the Hinglish story**, because the question is whether engines degrade on `hi-en` compared with `hi` or `en-IN`
5. Write `evals/results/stt_{YYYYMMDD_HHMMSS}.json`. The filename has seconds, and the runner refuses to open an existing path

**The results schema** (a wrapper around phase_4.md's per-engine object):

```json
{
  "run_at": "2026-09-24T14:02:11+05:30",
  "git_sha": "abc1234",
  "dataset": {"n_clips": 100, "manifest_sha256": "…", "by_lang": {"hi-en": 70, "hi": 15, "en-IN": 15}},
  "engines": [
    {
      "engine": "sarvam-saaras",
      "config": {"model": "saaras:v3", "language_code": "hi-IN"},
      "wer": 0.12, "cer": 0.08, "cer_roman": 0.05,
      "p50_ms": 310, "p95_ms": 590, "failures": 0,
      "by_lang": {"hi-en": 0.14, "hi": 0.09, "en-IN": 0.11},
      "by_condition": {"clean": 0.10, "fast": 0.17, "noisy": 0.21},
      "per_clip": []
    }
  ]
}
```

Each adapter exposes a `config: dict` class attribute, which the runner copies in. This addition to the adapters doesn't change the Protocol.

Use Pydantic models for the result rows (convention: no raw dicts crossing module lines). Dump them with `model_dump_json(indent=2)`.

### 4.7 — Latency helper

**File:** `evals/metrics/latency.py`

- `percentile(values: list[float], p: int) -> float` uses linear interpolation, the same as `numpy.percentile` defaults
- It raises on an empty list
- Add `tests/evals/test_latency.py` with 3 cases

### 4.8 — Results summary

This is inline in `run_stt_eval.py`. After writing the JSON, print these to stdout:

```
| Engine            | WER  | CER  | CER (roman) | hi-en WER | p50 ms | p95 ms | fails |
|-------------------|------|------|-------------|-----------|--------|--------|-------|
```

Then print the **5 worst clips per engine** (ref and hyp side by side). These feed the "what surprised me" part of FINDINGS.

Paste the table into `docs/FINDINGS.md` under `## STT Results`, together with the following:
- the dataset composition (buckets, speakers, recording conditions)
- 3–5 bullet findings drawn from the worst-clips output, such as script errors, domain-term misses or the noisy-bucket drop
- the caveats: Whisper is forced to `hi` on CPU, where the Google model runs, and that latency was measured from India over home broadband

---

## Build order within the phase

| Day | Work |
|---|---|
| 1 | `prompts.txt` (~100 lines), `record.py` and `validate.py`, then record yourself (~60 clips) |
| 2 | Record 2–3 other speakers and pull the Common Voice clips, then `validate.py` passes. Build `wer.py` and `latency.py` with their tests |
| 3 | Build the Deepgram, Whisper and Google adapters. Smoke-test each on 3 clips with `--limit 3` |
| 4 | Full run, then FINDINGS section, current.md update and PR |

The long pole is recording, not code. If you're behind on day 2, ship with 50 clips and say so.

---

## Done when

1. `validate.py` passes on `manifest.jsonl` with ≥50 clips covering all three `lang` values
2. `uv run python -m evals.run_stt_eval --engines sarvam,deepgram,whisper,google` completes, with any failures counted rather than crashing the run
3. `evals/results/stt_{timestamp}.json` is committed with WER, CER, `cer_roman`, p50, p95 and the `by_lang` breakdown for all 4 engines
4. The table, the dataset composition and the findings are in `docs/FINDINGS.md` under `## STT Results`
5. `tests/evals/test_wer.py` and `tests/evals/test_latency.py` pass
6. `STT_ENGINE=deepgram` works in the live agent. This isn't required, but the adapter contract says it should come for free, so check it once. Sarvam is the only one that's a LiveKit `stt.STT` subclass, so the others may need a thin wrapper, which is out of scope. If they do, note it and move on

---

## Files created or changed this phase

```
evals/
  datasets/stt_hinglish/
    README.md               (transcript conventions)
    prompts.txt
    record.py
    validate.py
    manifest.jsonl
    clip_001.wav … clip_NNN.wav
  metrics/
    __init__.py
    wer.py
    latency.py
  results/
    stt_{timestamp}.json
  run_stt_eval.py
agent/providers/stt/
  deepgram.py               (completed)
  whisper.py                (completed)
  google.py                 (completed)
  sarvam.py                 (add `config` attr only)
tests/evals/
  test_wer.py
  test_latency.py
docs/FINDINGS.md            (STT Results section)
.env.example                (DEEPGRAM_API_KEY, GOOGLE_APPLICATION_CREDENTIALS)
```

---

## Dependency notes

Add to `pyproject.toml`:
- `jiwer>=3.0`
- `faster-whisper>=1.0`
- `google-cloud-speech>=2.26`
- `indic-transliteration>=2.3`
- `soundfile>=0.12`

Put `sounddevice>=0.4` (only `record.py` needs it) in the `[dependency-groups] dev` group.

## Budget

| Engine | Cost for ~10 min of audio × ~3 runs |
|---|---|
| Sarvam | free credits |
| Deepgram | $0, from the $200 signup credit |
| Google | $0 inside 60 min/month. **Cap full runs at 5 per month**, or use `--engines` to skip Google while iterating |
| Whisper | $0 (local) |

The phase costs ~$0.

## Known risks

- **Whisper latency on CPU** could be 5–15 s per clip, which makes a full run 15–25 minutes. That's acceptable, but do the Whisper run last and run it once
- **Google setup friction** (GCP project, billing, service account) is the most likely thing to eat an afternoon. If it blocks for more than 2 hours, ship with 3 engines, mark Google `[-]` with the reason, and add it later
- **Sarvam model naming:** CLAUDE.md says Saarika, but the adapter runs `saaras:v3`. Record whichever actually ran in `config`. Benchmarking both is a one-line registry addition if you want it
