# stt_hinglish

The Hinglish STT eval set. Every clip is 16 kHz mono PCM16 WAV, 0.3–15 s long. Each clip has one row in `manifest.jsonl`:

```json
{"file": "clip_001.wav", "transcript": "आज हम polity पढ़ रहे हैं", "lang": "hi-en", "condition": "clean", "speaker": "p1", "source": "self", "duration_s": 3.4}
```

## Transcript conventions

Apply these strictly, because the score depends on them.

- **Hindi words are written in Devanagari and English words in Latin**: `आज हम polity पढ़ रहे हैं`. Don't use romanised Hindi (`aaj hum`). Sarvam, Google and Deepgram all write Hindi in Devanagari, so a romanised reference would score every engine near 100% WER.
- **Proper nouns follow the language of the word.** `Lok Sabha` is written `लोक सभा`, and `Parliament` stays `Parliament`.
- **Numbers are written as digits**: `370`, `42nd`.
- **No punctuation is needed**, because the normaliser strips it.
- **Transcribe what was said, not what the prompt said.** If the reading slipped, fix the transcript, not the audio.

## Buckets

The targets below are for 100 clips; halve them for the 50-clip floor.

| lang | condition | target |
|---|---|---|
| hi-en | clean | 45 |
| hi | clean | 15 (part of these from Common Voice) |
| en-IN | clean | 15 |
| hi-en | fast | 10 |
| hi-en | noisy | 15 |

## Workflow

```bash
# record, one speaker at a time (resume with --start-at)
uv run python -m evals.datasets.stt_hinglish.record --speaker p1
uv run python -m evals.datasets.stt_hinglish.record --speaker p2 --source friend

# pull ~10 Common Voice Hindi clips into the `hi` bucket
uv run python -m evals.datasets.stt_hinglish.import_cv --tsv ~/Downloads/cv-hi/validated.tsv --n 10

# check the dataset before any eval run
uv run python -m evals.datasets.stt_hinglish.validate
```

Sarvam-TTS-synthesised clips are **not** part of the headline number (spec D9). If you add any, tag them `source: "synth"`.

Get consent from anyone whose voice goes in here, because the WAVs are committed.
