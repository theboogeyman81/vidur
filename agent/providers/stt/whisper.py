from __future__ import annotations

import asyncio
import io
import os
import time

import soundfile as sf

from agent.providers.base import STTResult

_model = None  # faster_whisper.WhisperModel, loaded lazily — first load downloads ~3 GB


def _get_model(size: str):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # noqa: PLC0415 — heavy import, eval-only

        _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


class WhisperSTT:
    """Local faster-whisper on CPU. Eval-only — far too slow for the live path.

    Language is forced to Hindi (spec D4): auto-detect on Hinglish flips between Hindi,
    Urdu script and English translation per clip, which measures noise, not recognition.
    """

    name = "whisper-large-v3"

    def __init__(self) -> None:
        self._size = os.getenv("WHISPER_MODEL", "large-v3")
        self.config = {
            "model": self._size,
            "device": "cpu",
            "compute_type": "int8",
            "language": "hi",
            "beam_size": 5,
        }
        if self._size != "large-v3":
            self.name = f"whisper-{self._size}"

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        samples, sr = sf.read(io.BytesIO(audio), dtype="float32")
        if sr != 16000:
            raise ValueError(f"whisper adapter expects 16 kHz audio, got {sr}")
        model = _get_model(self._size)  # outside the timed region

        def _run() -> tuple[list[dict], dict]:
            segments, info = model.transcribe(samples, language="hi", beam_size=5, vad_filter=False)
            # segments is a lazy generator — decoding actually happens here
            segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
            return segs, {"language": info.language, "duration": info.duration}

        t0 = time.perf_counter()
        segs, info = await asyncio.to_thread(_run)
        latency_ms = (time.perf_counter() - t0) * 1000

        return STTResult(
            text=" ".join(s["text"].strip() for s in segs).strip(),
            latency_ms=latency_ms,
            raw={"segments": segs, "info": info},
        )
