from __future__ import annotations

import io
import logging
import os
import time
import uuid
import wave

import httpx
from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents import stt as lk_stt
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from agent.providers.base import STTResult

_SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
_MIN_AUDIO_DURATION_S = 0.3  # skip clips shorter than 300ms — Sarvam rejects silence/noise

log = logging.getLogger(__name__)


class SarvamSTT(lk_stt.STT):
    """Sarvam Saarika v2 STT — LiveKit plugin + eval harness adapter."""

    name = "sarvam-saaras"

    def __init__(self) -> None:
        super().__init__(
            capabilities=lk_stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._key = os.environ["SARVAM_API_KEY"]
        self.last_latency_ms: float = 0.0

    async def _recognize_impl(
        self,
        buffer: lk_stt.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> lk_stt.SpeechEvent:
        if isinstance(buffer, rtc.AudioFrame):
            frames: list[rtc.AudioFrame] = [buffer]
        else:
            frames = list(buffer)

        combined = rtc.combine_audio_frames(frames)

        if combined.duration < _MIN_AUDIO_DURATION_S:
            log.debug("stt: skipping short audio %.3fs", combined.duration)
            return _empty_event()

        wav_bytes = combined.to_wav_bytes()
        log.debug("stt: sending audio %.3fs (%d bytes)", combined.duration, len(wav_bytes))

        result = await self._http_transcribe(wav_bytes)
        self.last_latency_ms = result.latency_ms

        return lk_stt.SpeechEvent(
            type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[
                lk_stt.SpeechData(
                    language="hi-IN",
                    text=result.text,
                    confidence=1.0,
                )
            ],
        )

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        """Eval harness interface — caller passes pre-formed WAV bytes."""
        return await self._http_transcribe(audio)

    async def _http_transcribe(self, wav: bytes) -> STTResult:
        try:
            with wave.open(io.BytesIO(wav)) as wf:
                duration_s = wf.getnframes() / wf.getframerate()
        except Exception as exc:
            log.warning("stt: invalid WAV bytes: %s", exc)
            return STTResult(text="", latency_ms=0.0, raw={"error": str(exc)})

        if duration_s < _MIN_AUDIO_DURATION_S:
            return STTResult(text="", latency_ms=0.0, raw={"skipped": "too_short"})

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                _SARVAM_STT_URL,
                headers={"api-subscription-key": self._key},
                files={"file": ("audio.wav", wav, "audio/wav")},
                data={"model": "saaras:v3", "language_code": "hi-IN"},
            )

        if r.is_error:
            log.error("stt: Sarvam %d — %s", r.status_code, r.text[:500])
            r.raise_for_status()

        payload = r.json()
        return STTResult(
            text=payload.get("transcript", ""),
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=payload,
        )


def _empty_event() -> lk_stt.SpeechEvent:
    return lk_stt.SpeechEvent(
        type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
        request_id=str(uuid.uuid4()),
        alternatives=[lk_stt.SpeechData(language="hi-IN", text="", confidence=0.0)],
    )
