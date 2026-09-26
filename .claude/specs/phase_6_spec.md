# Phase 6 — `feat/piper-voice`

**Branch:** `feat/piper-voice` (stacked on `feat/tts-eval`; rebase onto `main` once Phase 5 merges)
**Status:** not started
**Blocked by:** Phase 5 merged. The code is done, but the full run is still outstanding. Phase 6 does **not** need the Phase 5 full run to start, because it re-runs its own Piper baseline (D7). It does need the Phase 5 runner, the Piper adapter and the judge choice.
**Goal:** Fine-tune a Piper voice on Indian-accented English, wrap it as a registry entry next to the existing `piper-hi`, and put three local Piper rows on the TTS leaderboard: stock Hindi, stock US English and fine-tuned Indian English. The finding is where each one breaks on code-mixed tutor speech. If the fine-tune stalls past the 6 h time-box, fall back to F5-TTS zero-shot cloning and label it as cloned.

> Phase 5 already set the baseline: `piper-hi` (`hi_IN-pratham-medium`) reads Devanagari fine but has to guess at every Latin-script English term, because espeak's Hindi phonemizer handles them. An English base voice fixes the English terms and can't read Devanagari at all. This phase measures that trade and doesn't hide it.

---

## Context: what earlier phases left you

- `agent/providers/tts/piper.py` has a `PiperTTS` class (`name = "piper-hi"`). It uses the `piper-tts` package (≥1.8.0) with lazy load outside the timed region, `asyncio.to_thread`, TTFB taken at the first yielded chunk, and output wrapped with `pcm16_to_wav`. `PIPER_MODEL_PATH` sets the model, and the default is `models/piper/hi_IN-pratham-medium.onnx`
- The registry key `piper` maps to `PiperTTS`. `run_tts_eval.py` treats `"piper"` as local in `_confirm_budget` (so no quota prompt)
- `models/piper/` is gitignored. `training/` doesn't exist yet
- `run_tts_eval.py` writes `evals/results/tts_{ts}.json` in `"x"` mode, supports `--mix`, `--script native|roman|both` and `--judge`, and saves audio to `evals/results/audio/{run_ts}/{engine}/`
- The `tts_codemix` set has 50 sentences: heavy 20, light 10, pure_hi 8, pure_en 7 and numeric 5. 15 of them have `text_roman`
- `docs/FINDINGS.md` has `## TTS Results`, with an empty leaderboard table until Phase 5's full run
- Phase 4's `evals/datasets/stt_hinglish/record.py` already records clips from a prompt list. It can be reused if the data has to be self-recorded (D3)

---

## Decisions locked for this phase

These fix gaps in `phases/phase_6.md`. Each one names its tradeoff and picks a side.

| # | Decision | Tradeoff |
|---|---|---|
| D1 | **Features are numbered 6.x.** phase_6.md numbers them 5.x, which is the same copy-paste slip Phase 5 had. | None. Fix phase_6.md in the same PR |
| D2 | **The fine-tune is added as a new registry entry and doesn't replace `piper-hi`.** phase_6.md says to "update piper.py to load vidur_en.onnx instead", but that would delete the baseline the phase is supposed to beat. `PiperTTS` gets a class-level `name`, `_env_var` and `_default_model`, and each voice is a 3-line subclass in the same file. The keys are `piper` → `piper-hi` (unchanged), `piper-en-us` → stock `en_US` voice (control), and `piper-en-in` → fine-tuned (`piper-en-in-ft`). | Three registry entries share one class. That still fits the adapter contract, because each is one registration and there are no agent changes |
| D3 | **The training data comes from a single speaker.** phase_6.md suggests Common Voice `en-IN`. Common Voice has no `en-IN` locale, only English with an "India and South Asia" accent tag, and that data is multi-speaker, recorded on phone mics and heavy in reading-style noise. A single-speaker Piper fine-tune on that data blends voices and learns the noise floor. **First choice is a single-speaker Indian English studio corpus** (IndicTTS English from IIT Madras is the candidate; **verify the license allows non-commercial research use at impl time**). **The fallback is to self-record ~300 lines (~30–40 min)** with Phase 4's `record.py`, using sentences from the `pure_en` style plus UPSC terms. | Self-recording is ~2 h of manual work but gives a clean license and the author's own voice. The corpus route is faster but has a license to check. Either way, don't use Common Voice |
| D4 | **The base checkpoint is `en_US` medium (e.g. `lessac` or `ryan`) from `rhasspy/piper-checkpoints`**, at 22050 Hz, the same quality tier as `hi_IN-pratham-medium`. The stock ONNX of **that same voice** is the `piper-en-us` control row. | Without the control you can't tell whether a gain on English terms comes from the fine-tune or just from switching to the English phonemizer. It costs one extra local row at $0 |
| D5 | **Data prep is a local, tested script (`training/prepare_dataset.py`), not notebook cells.** The notebook only trains and exports. | There are two places to look instead of one, but the part that can silently corrupt data (resampling, filtering, the metadata format) is covered by tests |
| D6 | **The 6 h time-box is GPU wall-clock, counted from the first training step**, summed across Colab sessions. Data prep and setup don't count. At 6 h, apply the gate (6.3): pass → keep going to export; fail → fallback. Log each session's start, end and step in the notebook. | A hard clock can kill a run that was about to converge. The alternative, an unbounded Colab loop, is worse for a 7-day build |
| D7 | **All Piper rows (and the fallback row, if it's used) are produced in one run** with the same judge as Phase 5: `--engines piper,piper-en-us,piper-en-in`. The hosted engines are **not** re-run (quota). The FINDINGS leaderboard merges rows from two result files and cites both filenames. | Merging across runs is only valid if the judge and dataset hash match. The runner already records `judge` and `sentences_sha256`, so check that both match before merging and say so in FINDINGS |
| D8 | **The fallback is labelled `f5-cloned`, not `piper-f5-cloned`.** F5-TTS isn't Piper; it's a different architecture. The Piper name on it would be the dishonest label CLAUDE.md warns about. | None |
| D9 | **Latency for the F5 fallback is measured locally (MPS on the Mac, or CPU) and reported as-is, or marked `n/a — offline only` if it's slower than real time.** Colab GPU timings aren't comparable to anything else in the table, so they don't go in it. | F5 will almost certainly lose on latency. That's the honest result: a flow-matching model isn't a real-time voice-agent TTS on this hardware |
| D10 | **The ONNX files are never committed.** A medium Piper voice is ~60 MB, which is over the 50 MB line. They live in `models/piper/` (already gitignored; don't add `training/models/`) and are uploaded to a **Hugging Face model repo** with a `training/download_model.py` that pulls them. | There's one more external dependency, but Fly.io (Phase 8) needs a scripted download anyway. Google Drive links aren't scriptable reliably |
| D11 | **Harness only, not the live agent.** Piper has no LiveKit wrapper (the same situation as Phase 5's done-when #9). Setting `TTS_ENGINE=piper-en-in` in the agent is out of scope. | The fine-tuned voice is never heard in the live demo. That's fine: the demo moment is the A/B player (#5), not the live call |

---

## Features

### 6.1 — Training data prep

**Files:** `training/prepare_dataset.py`, `training/data/` (gitignored), `training/DATA.md`

```
uv run python -m training.prepare_dataset --src <raw_dir> --transcripts <file> --out training/data/vidur_en
```

- Input is the raw corpus (D3) or self-recorded clips plus a transcript file
- Output uses the **LJSpeech layout**: `wavs/{id}.wav` and `metadata.csv` (`id|transcript`, no header), in the format Piper's preprocessing expects. **Verify the exact column format for the current `piper1-gpl` training CLI at impl time**
- Resample to **22050 Hz, mono, PCM16** (to match the medium base, D4)
- Filter out clips shorter than 1 s or longer than 15 s, trim leading and trailing silence (keeping ~100 ms), and drop clips whose transcript contains digits (Piper doesn't normalise numbers; spell them out or drop them)
- Target **45–90 min** after filtering. Print total duration, clip count, and p50/p95 clip length
- Validate each transcript: non-empty, Latin script only (it's an English voice)
- Put a Pydantic `ClipRow` model in the script
- `training/DATA.md` records the source, license, speaker (gender, region), total duration, filters applied, and the zip's SHA-256
- Zip the output for upload to Drive/Colab

**Tests:** `tests/training/test_prepare_dataset.py` covers the resample output rate, the duration filter, digit rejection and the metadata format (using 3 synthetic sine-wave clips).

### 6.2 — Piper fine-tune (Colab)

**File:** `training/piper_finetune.ipynb`

The notebook's cells, in order:
1. Mount Drive, install the training deps (`piper1-gpl` with its training extra; **verify the package, CLI entry point and Lightning version at impl time**, because Piper's training moved repos in 2025), and check that the GPU is a T4 or better
2. Download the base checkpoint (D4) and the dataset zip
3. Preprocess: phonemize with `espeak-ng` voice `en-us`, single speaker
4. **Fine-tune from the checkpoint** with the batch size tuned to fit on a T4. Save a checkpoint to **Drive** every ~500 steps, so that a Colab disconnect costs at most one interval
5. **Listening checkpoints:** every ~500 steps, synthesize 5 fixed probe sentences and play them inline. Use `pure_en` sentences from the eval set plus two with UPSC terms (`Directive Principles`, `Lok Sabha`). Log the step, GPU minutes used so far (D6) and a one-line note for each checkpoint
6. Export the chosen checkpoint to ONNX: `vidur_en_in.onnx` and `vidur_en_in.onnx.json`. **Edit the JSON's dataset/name fields** so that the voice doesn't claim to be `lessac`
7. A session log table in markdown: session #, start, end, steps, cumulative GPU h

phase_6.md's "~2000 steps" is a starting guess. The real stopping rule is the gate in 6.3.

### 6.3 — Quality gate and F5-TTS fallback

**The gate** runs at each listening checkpoint after step 1000, and at the 6 h mark:
- **Accent:** a listener would call it Indian English on 4 of the 5 probes
- **Intelligibility:** run the 5 probes through the judge STT (Deepgram, via a notebook cell or locally), and require **WER ≤ the stock `en_US` voice's WER on the same probes + 0.05**. A fine-tune that picks up the accent but loses intelligibility fails

Pass → export (6.2 step 6). The fallback triggers at the 6 h mark if the gate still fails, or earlier on a hard failure: the loss diverges, the audio is noise, or Colab can't get a GPU for more than a day.

**Fallback (F5-TTS zero-shot):**
- Take a **10 s reference clip** of Indian-accented English from the same D3 source, with its exact transcript (F5 needs the reference text)
- Add the adapter `agent/providers/tts/f5.py` with the registry key `f5` and `name = "f5-cloned"` (D8). It uses the `f5-tts` package with lazy load, and the device is MPS if available, else CPU. Put `device` and `nfe_steps` in `config`. It returns WAV at the model's native rate (24 kHz)
- **Check the license:** F5-TTS's pretrained weights are CC-BY-NC. That's fine for this project but has to be noted in the README and FINDINGS
- Pick the base model at impl time. The official base is en+zh; a community Hindi or Indic checkpoint may exist. Whichever is used, name it in `config`
- Put the reference clip in `models/f5/reference_clip.wav` (gitignored) plus `reference_clip.txt`
- In `docs/FINDINGS.md`, add a `### Piper fine-tune → F5 fallback` note: what failed, at what step, after how many GPU hours, and with the gate numbers

### 6.4 — Model packaging

**Files:** `training/download_model.py`, `.env.example`

- Upload `vidur_en_in.onnx` and `.onnx.json` to an HF model repo (`<user>/vidur-piper-en-in`) with a model card that states the base checkpoint, data source and license, step count, and "fine-tuned, not trained from scratch"
- `uv run python -m training.download_model` pulls **all three** Piper voices into `models/piper/`: `hi_IN-pratham-medium` and the `en_US` control from `rhasspy/piper-voices`, and `vidur_en_in` from the HF repo. It's idempotent: skip a file if it's present and its SHA-256 matches
- `.env.example` gets `PIPER_EN_US_MODEL_PATH` and `PIPER_EN_IN_MODEL_PATH`, plus `F5_REF_CLIP` if the fallback is used
- The README gets a "Local voices" section with one command

### 6.5 — Piper provider: three voices, one class

**Files:** `agent/providers/tts/piper.py`, `agent/providers/tts/__init__.py`

```python
class PiperTTS:
    name = "piper-hi"
    _env_var = "PIPER_MODEL_PATH"
    _default_model = "models/piper/hi_IN-pratham-medium.onnx"
    ...

class PiperEnUSTTS(PiperTTS):
    name = "piper-en-us"
    _env_var = "PIPER_EN_US_MODEL_PATH"
    _default_model = "models/piper/en_US-lessac-medium.onnx"   # whichever base D4 picks

class PiperEnINTTS(PiperTTS):
    name = "piper-en-in-ft"
    _env_var = "PIPER_EN_IN_MODEL_PATH"
    _default_model = "models/piper/vidur_en_in.onnx"
```

- `__init__` reads `self._env_var` and `self._default_model`. The `FileNotFoundError` message points at `training.download_model`
- `config` gains a `base` field (`"stock"` or `"finetuned:<base voice>@<step>"`). Read it from the `.onnx.json` if you put it there in 6.2 step 6, and otherwise hardcode it
- Registry keys: `piper-en-us`, `piper-en-in` (and `f5` if the fallback is used)
- `run_tts_eval._confirm_budget` currently checks `e != "piper"`. Change that to a `_LOCAL = {"piper", "piper-en-us", "piper-en-in", "f5"}` set
- Nothing else changes. `synthesize()` and the timing path stay the same

**Tests:** `tests/agent/test_piper_voices.py` checks that each subclass resolves its env var over its default, that a missing file raises with the download hint, that the registry has the new keys, and that `name` values are unique across the TTS registry.

### 6.6 — Leaderboard run

```
uv run python -m evals.run_tts_eval --engines piper,piper-en-us,piper-en-in --judge <phase5 judge>
```

- It's one run with all Piper rows (D7). If Phase 5's full run hasn't happened yet, run with `--judge deepgram` and record that choice. Phase 5 has to use the same judge for its merge to be valid
- Add the three rows to the FINDINGS leaderboard with an **"Engine source"** footnote: hosted rows come from `tts_<phase5 ts>.json`, Piper rows from `tts_<phase6 ts>.json`, and the footnote confirms the judge and `sentences_sha256` match
- Also add a **Piper-only table** that breaks RT-WER down by `mix` bucket (heavy / light / pure_hi / pure_en / numeric) and by script (native vs roman, on the paired 15). This is where the phase's finding shows up
- **Demo export:** run `export_tts_demo.py` for 3 sentences (one `heavy`, one `pure_en`, one `roman`) across `piper-hi` / `piper-en-in-ft` / `sarvam`. The existing `audio_demo/` from Phase 5 stays as it is. Add to it; don't replace it

### 6.7 — Code-mix degradation finding

A new `docs/FINDINGS.md` subsection, `### Local voices: where Piper breaks`.

**Expected shape** (write down the real one, not this):
- `piper-hi`: good on `pure_hi`, bad on English entities (low entity hit rate)
- `piper-en-us` / `piper-en-in-ft`: good on `pure_en` and entities, with **Devanagari silently dropped or read as letter names**. Check what espeak `en-us` actually does with Devanagari input and describe it exactly
- **The roman bucket is the interesting one.** An English-phonemizer voice reading `Aaj hum Article 370 padhenge` may beat `piper-hi` reading the Devanagari. If it does, that's a concrete argument for pinning the tutor prompt to romanized Hinglish **when the TTS is English-based**. It ties back to Phase 5's "should the prompt pin a script" question
- The accent delta: `piper-en-in-ft` vs `piper-en-us` on `pure_en` RT-WER. Say what the fine-tune bought, if anything

Do a listening pass on the 5 worst sentences per Piper voice and add tags to `listening_notes.jsonl` using the Phase 5 tag set. Add a new tag, `dropped_devanagari`.

State the non-goal plainly: **script-routing** (split the utterance by script and send each span to a different voice) would fix this, but it's out of scope. Name it as the next step instead.

---

## Build order within the phase

| Day | Work |
|---|---|
| 1 | Check the D3 license and pick the source, or start self-recording. Write `prepare_dataset.py` with tests. Pick the base (D4). Refactor 6.5 and download the stock `en_US` voice. Smoke-test `--engines piper,piper-en-us --limit 3 --no-roundtrip` |
| 2 | Colab: set up the notebook and start the fine-tune. Listening checkpoints every ~500 steps. Keep a running clock (D6) |
| 3 | Gate → export and HF upload, or the fallback. Write `download_model.py`. Full Piper run, listening pass, FINDINGS, demo export, fixes to current.md and phase_6.md, then the PR |

The long pole is Colab time and GPU availability. Start training at the beginning of day 2, not the end. If D3 needs self-recording, day 1 is mostly recording.

---

## Done when

1. `training/prepare_dataset.py` produces a valid LJSpeech dataset, and `training/DATA.md` documents the source and license
2. Either `vidur_en_in.onnx` exists and passes the 6.3 gate, **or** the `f5-cloned` fallback exists with the reason documented in FINDINGS. Whichever it is, it's labelled honestly everywhere
3. `uv run python -m training.download_model` reproduces `models/piper/` from a clean clone
4. `uv run python -m evals.run_tts_eval --engines piper,piper-en-us,piper-en-in` completes, and `tts_{ts}.json` is committed
5. The FINDINGS leaderboard has the new rows with the source footnote, plus the Piper per-bucket table and `### Local voices: where Piper breaks`
6. Tests pass: `tests/training/test_prepare_dataset.py` and `tests/agent/test_piper_voices.py`
7. The notebook is committed **with outputs cleared except the session log table and gate results**, so Colab audio blobs don't bloat the repo

---

## Files created or changed this phase

```
training/
  prepare_dataset.py            (new)
  download_model.py             (new)
  piper_finetune.ipynb          (new)
  DATA.md                       (new)
  data/                         (gitignored)
agent/providers/tts/
  piper.py                      (class attrs + two subclasses)
  f5.py                         (fallback only)
  __init__.py                   (piper-en-us, piper-en-in, [f5])
evals/
  run_tts_eval.py               (_LOCAL engine set)
  results/tts_{ts}.json
  results/audio_demo/           (3 sentences × 3 engines added)
  datasets/tts_codemix/listening_notes.jsonl
tests/
  training/test_prepare_dataset.py
  agent/test_piper_voices.py
docs/FINDINGS.md                (leaderboard rows, per-bucket table, Local voices, [F5 note])
README.md                       (Local voices section, GPL-3 / CC-BY-NC notes)
.env.example                    (PIPER_EN_US_MODEL_PATH, PIPER_EN_IN_MODEL_PATH, [F5_REF_CLIP])
.gitignore                      (training/data/, models/f5/)
.claude/phases/phase_6.md       (renumber 5.x → 6.x, D2/D3/D8 corrections)
```

---

## Dependency notes

- Locally, only `huggingface_hub` is needed for `download_model.py` (check whether it's already a transitive dependency). `piper-tts` is already in place
- The training dependencies live **only in the Colab notebook** and never go into `pyproject.toml`: torch, lightning and the piper training extra
- For the fallback, add `f5-tts` as an **optional** dependency group (`uv add --optional f5 f5-tts`), because it pulls in torch, which the main install doesn't need

## Budget

| Item | Cost |
|---|---|
| Colab free T4 | $0. Session limits and GPU availability vary, and that's the real constraint. **Don't buy Colab Pro** ($10 is the whole-project ceiling) |
| HF model hosting | $0 (a public model repo) |
| Judge STT (Deepgram) | ~65 clips × 3 voices × ~4 s ≈ 13 min of audio, within the signup credit |
| Piper / F5 inference | $0, local |

The phase costs about $0.

## Known risks

- **Piper's training tooling moved** (rhasspy/piper → OHF-Voice/piper1-gpl), and Colab's default Python/torch versions drift. Pin versions in the notebook's first cell. If the environment itself eats more than 2 h, that time counts against the D6 clock
- **Colab disconnects.** Save a checkpoint to Drive every interval, and make resuming a single cell
- **The fine-tune shifts the accent but keeps the US prosody**, or overfits after ~45 min of data and gets raspy. The 500-step listening checkpoints are there to catch this, so keep the best checkpoint rather than the last one
- **espeak `en-us` on Devanagari** may emit nothing, spell out Unicode names, or crash the phonemizer. Test one heavy sentence on day 1 with the stock voice. If it crashes, the runner counts it as a failure (Phase 5 D9); record that as the finding, and don't pre-filter the text
- **IndicTTS license** may not permit this use. If it's unclear, go straight to self-recording rather than guessing
- **F5 on CPU/MPS** may take tens of seconds per sentence. Run the fallback row with `--limit` if it has to, and report the partial coverage honestly
