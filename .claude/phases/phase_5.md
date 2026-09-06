# Phase 5 — `feat/piper-voice`

**Goal:** Fine-tune a Piper TTS voice on Indian-accented English, export it as ONNX, wrap it as a TTSProvider, and add it to the leaderboard. If fine-tuning stalls past 6 hours, fall back to F5-TTS zero-shot cloning — label it honestly.

---

## Features

### 5.1 — Training data prep
- File: `training/piper_finetune.ipynb` (Colab notebook)
- Collect ~1–2 hours of Indian-accented English speech + transcripts
- Sources: Common Voice `en-IN`, curated YouTube clips (short segments), or self-recorded
- Format: LJSpeech structure — `wavs/` folder + `metadata.csv` (`filename|transcript`)
- Resample to 22050Hz mono
- Document source and license in the notebook

### 5.2 — Piper fine-tune (Colab)
- Start from a pre-trained Piper checkpoint (English base)
- Fine-tune for ~2000 steps — enough to shift accent without full training cost
- Monitor: listen to checkpoint audio every 500 steps; stop when accent is clearly Indian
- Export final checkpoint to ONNX: `vidur_en.onnx` + `vidur_en.onnx.json`
- Save to Google Drive, document Drive path in notebook

### 5.3 — Fallback: F5-TTS zero-shot cloning
- If fine-tuning stalls past 6 hours wall-clock or quality is clearly unusable:
  - Switch to F5-TTS with a 10-second reference clip of Indian-accented English
  - Zero-shot voice cloning — no training required
  - Label the result as `piper-f5-cloned`, not `piper-finetuned`, in all results
  - Document the switch and reason in `docs/FINDINGS.md`

### 5.4 — ONNX model packaging
- Store model files at: `training/models/vidur_en.onnx` and `vidur_en.onnx.json`
- Do not commit to git if >50MB — add to `.gitignore`, document download instructions in README
- If using F5-TTS fallback, store the reference clip at `training/models/reference_clip.wav`

### 5.5 — Piper TTSProvider update
- File: `agent/providers/tts/piper.py`
- Update to load `vidur_en.onnx` instead of the generic pre-trained voice
- Engine name in registry: `piper-finetuned` (or `piper-f5-cloned` if fallback)
- No other changes — the interface stays the same

### 5.6 — Add fine-tuned voice to TTS leaderboard
- Re-run `evals/run_tts_eval.py` with the updated Piper adapter
- Append results to a new timestamped `results/tts_{timestamp}.json`
- Update `docs/FINDINGS.md` table to include the fine-tuned row
- Side-by-side: same sentence through Sarvam Bulbul vs fine-tuned Piper — note which sounds more natural

### 5.7 — Eval: does the fine-tuned voice handle code-mixing?
- Run the full `tts_codemix` sentence set through the fine-tuned voice
- Note: Piper is English-only — Hindi words will be mispronounced. Document this.
- This is expected and worth recording: the finding is "fine-tuned English voice degrades on Devanagari tokens"
- Add this finding explicitly to `docs/FINDINGS.md`

---

## Done when

- A Piper ONNX model exists (fine-tuned OR F5-TTS fallback, clearly labelled)
- `piper.py` loads the new model; `uv run python -m evals.run_tts_eval --engines piper` runs
- New results row added to `docs/FINDINGS.md` leaderboard
- Code-mixing degradation documented honestly in FINDINGS.md

---

## Files created this phase

```
training/
  piper_finetune.ipynb    (updated with training run results)
  models/
    vidur_en.onnx         (or reference_clip.wav for F5 fallback)
    vidur_en.onnx.json
agent/providers/tts/
  piper.py                (updated to use fine-tuned model)
docs/
  FINDINGS.md             (updated with fine-tune results)
```
