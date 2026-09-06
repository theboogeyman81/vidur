from __future__ import annotations

import base64
import io
import os
import time
import uuid
import wave

import httpx
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents import tts as lk_tts
from livekit.agents.tts.tts import AudioEmitter

from agent.providers.base import TTSResult

_SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
_SARVAM_SAMPLE_RATE = 22050


def _split_sentences(text: str) -> list[str]:
    """Split at sentence boundaries to stay within Sarvam's 500-char limit."""
    import re

    parts = re.split(r"(?<=[.?!।])\s+", text.strip())
    chunks: list[str] = []
    for part in parts:
        while len(part) > 500:
            chunks.append(part[:500])
            part = part[500:]
        if part:
            chunks.append(part)
    return chunks or [text[:500]]


class SarvamTTS:
    """Eval harness TTS adapter — implements TTSProvider protocol."""

    name = "sarvam-bulbul"

    def __init__(self) -> None:
        self._key = os.environ["SARVAM_API_KEY"]

    async def synthesize(self, text: str, lang: str = "hi-IN") -> TTSResult:
        t0 = time.perf_counter()
        wav_bytes = await self._http_synthesize(text, lang)
        elapsed = (time.perf_counter() - t0) * 1000
        return TTSResult(audio=wav_bytes, time_to_first_byte_ms=elapsed, total_ms=elapsed)

    async def _http_synthesize(self, text: str, lang: str) -> bytes:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                _SARVAM_TTS_URL,
                headers={
                    "api-subscription-key": self._key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [text[:500]],
                    "target_language_code": lang,
                    "speaker": "meera",
                    "model": "bulbul:v1",
                    "enable_preprocessing": True,
                },
            )
        r.raise_for_status()
        return base64.b64decode(r.json()["audios"][0])


class _SarvamChunkedStream(lk_tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: SarvamLKTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        api_key: str,
        lang: str,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._api_key = api_key
        self._lang = lang
        self.ttfb_ms: float = 0.0

    async def _run(self, output_emitter: AudioEmitter) -> None:
        chunks = _split_sentences(self._input_text)
        t0 = time.perf_counter()
        pcm_parts: list[tuple[int, int, bytes]] = []  # (sample_rate, channels, pcm)

        for chunk in chunks:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(
                    _SARVAM_TTS_URL,
                    headers={
                        "api-subscription-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "inputs": [chunk],
                        "target_language_code": self._lang,
                        "speaker": "meera",
                        "model": "bulbul:v1",
                        "enable_preprocessing": True,
                    },
                )
            r.raise_for_status()
            wav_bytes = base64.b64decode(r.json()["audios"][0])

            with wave.open(io.BytesIO(wav_bytes)) as wf:
                sr = wf.getframerate()
                nch = wf.getnchannels()
                pcm = wf.readframes(wf.getnframes())

            if not pcm_parts:
                # first chunk — record TTFB and initialize emitter
                self.ttfb_ms = (time.perf_counter() - t0) * 1000
                self._tts.last_ttfb_ms = self.ttfb_ms  # propagate to parent for trace logging
                output_emitter.initialize(
                    request_id=str(uuid.uuid4()),
                    sample_rate=sr,
                    num_channels=nch,
                    mime_type="audio/pcm",
                )

            pcm_parts.append((sr, nch, pcm))

        for _, _, pcm in pcm_parts:
            output_emitter.push(pcm)


class SarvamLKTTS(lk_tts.TTS):
    """LiveKit TTS plugin wrapping Sarvam Bulbul."""

    name = "sarvam-bulbul"

    def __init__(self, lang: str = "hi-IN") -> None:
        super().__init__(
            capabilities=lk_tts.TTSCapabilities(streaming=False),
            sample_rate=_SARVAM_SAMPLE_RATE,
            num_channels=1,
        )
        self._key = os.environ["SARVAM_API_KEY"]
        self._lang = lang
        self.last_ttfb_ms: float = 0.0

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> _SarvamChunkedStream:
        stream = _SarvamChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            api_key=self._key,
            lang=self._lang,
        )
        return stream
