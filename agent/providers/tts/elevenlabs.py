from __future__ import annotations

import os

import httpx

from agent.providers.base import TTSResult
from agent.providers.tts._http import stream_pcm16

_ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
_SAMPLE_RATE = 24000
_LANG = {"hi-IN": "hi", "en-IN": "en"}


class ElevenLabsTTS:
    """ElevenLabs Flash v2.5 over chunked HTTP (spec D10) — raw PCM16, wrapped as WAV."""

    name = "elevenlabs-flash"

    def __init__(self) -> None:
        self._key = os.environ["ELEVENLABS_API_KEY"]
        self._voice = os.environ["ELEVENLABS_VOICE_ID"]
        # language_code isn't documented for Flash v2.5; set ELEVENLABS_LANGUAGE_CODE=0 if rejected
        self._send_lang = os.getenv("ELEVENLABS_LANGUAGE_CODE", "1") != "0"
        self.config = {
            "model_id": "eleven_flash_v2_5",
            "voice_id": self._voice,
            "voice_name": os.getenv("ELEVENLABS_VOICE_NAME", ""),
            "sample_rate": _SAMPLE_RATE,
            "language_code_sent": self._send_lang,
            "streaming": True,
        }
        self._client: httpx.AsyncClient | None = None

    async def synthesize(self, text: str, lang: str = "hi-IN") -> TTSResult:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15)
        body: dict = {"model_id": self.config["model_id"], "text": text}
        if self._send_lang:
            body["language_code"] = _LANG.get(lang, lang.split("-")[0])
        return await stream_pcm16(
            self._client,
            _ELEVENLABS_URL.format(voice_id=self._voice),
            vendor="elevenlabs",
            sample_rate=_SAMPLE_RATE,
            params={"output_format": f"pcm_{_SAMPLE_RATE}"},
            headers={"xi-api-key": self._key},
            json=body,
        )
