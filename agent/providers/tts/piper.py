from __future__ import annotations

import asyncio
import os
import pathlib
import time

from agent.providers.base import TTSResult
from agent.providers.tts._wav import pcm16_to_wav

_DEFAULT_MODEL = "models/piper/hi_IN-pratham-medium.onnx"


class PiperTTS:
    """Local Piper (ONNX, CPU). Pre-trained hi_IN voice; the fine-tuned voice lands in Phase 6."""

    name = "piper-hi"

    def __init__(self) -> None:
        self._model_path = pathlib.Path(os.getenv("PIPER_MODEL_PATH", _DEFAULT_MODEL))
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"{self._model_path} missing — run: uv run python -m piper.download_voices "
                "--download-dir models/piper hi_IN-pratham-medium"
            )
        self.config = {"voice": self._model_path.stem, "device": "cpu", "streaming": True}
        self._voice = None

    def _load(self):
        # Lazy, once, outside the timed region
        if self._voice is None:
            from piper import PiperVoice

            self._voice = PiperVoice.load(self._model_path)
        return self._voice

    def _synthesize_sync(self, text: str) -> TTSResult:
        voice = self._load()
        t0 = time.perf_counter()
        ttfb_ms: float | None = None
        parts: list[bytes] = []
        sample_rate = voice.config.sample_rate
        for chunk in voice.synthesize(text):  # one chunk per sentence
            if ttfb_ms is None:
                ttfb_ms = (time.perf_counter() - t0) * 1000
            parts.append(chunk.audio_int16_bytes)
            sample_rate = chunk.sample_rate
        total_ms = (time.perf_counter() - t0) * 1000
        if not parts:
            raise RuntimeError("piper: no audio produced")
        return TTSResult(
            audio=pcm16_to_wav(b"".join(parts), sample_rate),
            time_to_first_byte_ms=ttfb_ms if ttfb_ms is not None else total_ms,
            total_ms=total_ms,
        )

    async def synthesize(self, text: str, lang: str = "hi-IN") -> TTSResult:
        # lang is fixed by the voice model; Piper has no per-call language switch
        return await asyncio.to_thread(self._synthesize_sync, text)
