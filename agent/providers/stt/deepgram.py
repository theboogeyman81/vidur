from __future__ import annotations

import logging
import os
import time

import httpx

from agent.providers.base import STTResult

_DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

log = logging.getLogger(__name__)


class DeepgramSTT:
    """Deepgram Nova-3, multilingual code-switching mode (Hindi is in the `multi` set)."""

    name = "deepgram-nova3"
    config = {"model": "nova-3", "language": "multi", "punctuate": False, "smart_format": False}

    def __init__(self) -> None:
        self._key = os.environ["DEEPGRAM_API_KEY"]

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        """`audio` is full WAV bytes; Deepgram reads the header, so sample_rate is informational."""
        params = {k: str(v).lower() for k, v in self.config.items()}
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                _DEEPGRAM_URL,
                params=params,
                headers={"Authorization": f"Token {self._key}", "Content-Type": "audio/wav"},
                content=audio,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        if r.is_error:
            log.error("stt: Deepgram %d — %s", r.status_code, r.text[:500])
            r.raise_for_status()

        payload = r.json()
        alts = payload["results"]["channels"][0]["alternatives"]
        return STTResult(
            text=alts[0]["transcript"] if alts else "",
            latency_ms=latency_ms,
            raw=payload,
        )
