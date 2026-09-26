# Vidur — Findings

## TTS Results

_Pending the full 4-engine run (Phase 5). Filled from `evals/results/tts_{ts}.json`; no numbers until then._

- **Dataset:** 50 tutor-side sentences (`evals/datasets/tts_codemix/`) — heavy 20, light 10, pure_hi 8, pure_en 7, numeric 5; 15 with a romanized twin
- **Judge STT:** _tbd_ (Phase 4 winner, not Sarvam) — clean-speech WER _tbd_ is the noise floor
- **Caveats:** Sarvam is non-streaming (TTFB == total); Piper runs locally on CPU; hosted engines timed over chunked HTTP, not WebSocket; network path is India → vendor

| Engine | p50 TTFB | p95 TTFB | p50 total | p95 total | RT-WER | RT-WER roman | entity hit | fails |
|---|---|---|---|---|---|---|---|---|

### Barge-in

| Config | p50 handled ms | p95 handled ms | real interruptions caught | backchannels → false interruption |
|---|---|---|---|---|
| default (`min_duration=0.5`) | | | /10 | /10 |
| `min_duration=0.3` | | | /10 | /10 |

### Sarvam live-path fix (5.9)

Before the fix, the first audio frame arrived ~2.5 s after the logged `tts_ttfb_ms` on a three-sentence reply, because every chunk was synthesized before any was pushed. After: first frame within ~30 ms of logged TTFB. Sarvam's first-chunk latency itself varied 1.0–6.4 s across 6 local runs.
