from __future__ import annotations

import os

import httpx

from agent.providers.base import TTSResult
from agent.providers.tts._http import stream_pcm16

_CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
_CARTESIA_VERSION = "2026-08-14"
_SAMPLE_RATE = 24000
_LANG = {"hi-IN": "hi-IN", "en-IN": "en-IN"}


class CartesiaTTS:
    """Cartesia Sonic over chunked HTTP (spec D10) — raw PCM16, wrapped as WAV."""

    name = "cartesia-sonic"

    def __init__(self) -> None:
        self._key = os.environ["CARTESIA_API_KEY"]
        self._voice = os.environ["CARTESIA_VOICE_ID"]
        self.config = {
            "model_id": "sonic-3.6",
            "api_version": _CARTESIA_VERSION,
            "voice_id": self._voice,
            "voice_name": os.getenv("CARTESIA_VOICE_NAME", ""),
            "sample_rate": _SAMPLE_RATE,
            "streaming": True,
        }
        self._client: httpx.AsyncClient | None = None

    async def synthesize(self, text: str, lang: str = "hi-IN") -> TTSResult:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15)
        return await stream_pcm16(
            self._client,
            _CARTESIA_URL,
            vendor="cartesia",
            sample_rate=_SAMPLE_RATE,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Cartesia-Version": _CARTESIA_VERSION,
            },
            json={
                "model_id": self.config["model_id"],
                "transcript": text,
                "voice": self._voice,
                "language": _LANG.get(lang, lang),
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": _SAMPLE_RATE,
                },
            },
        )
