"""Chunked-HTTP streaming for the hosted TTS adapters — every vendor timed the same way."""

from __future__ import annotations

import logging
import time

import httpx

from agent.providers.base import TTSResult
from agent.providers.tts._wav import pcm16_to_wav

_log = logging.getLogger(__name__)


async def stream_pcm16(
    client: httpx.AsyncClient, url: str, *, vendor: str, sample_rate: int, **request_kwargs
) -> TTSResult:
    """POST and stream raw PCM16. TTFB = first non-empty chunk, total = stream end (spec D5)."""
    t0 = time.perf_counter()
    ttfb_ms: float | None = None
    parts: list[bytes] = []
    async with client.stream("POST", url, **request_kwargs) as r:
        if r.is_error:
            body = await r.aread()
            _log.error("tts: %s %d — %s", vendor, r.status_code, body[:500])
            r.raise_for_status()
        async for chunk in r.aiter_bytes():
            if not chunk:
                continue
            if ttfb_ms is None:
                ttfb_ms = (time.perf_counter() - t0) * 1000
            parts.append(chunk)
    total_ms = (time.perf_counter() - t0) * 1000
    pcm = b"".join(parts)
    if not pcm:
        raise RuntimeError(f"{vendor}: empty audio stream")
    if len(pcm) % 2:  # a truncated trailing sample would corrupt the WAV frame count
        pcm = pcm[:-1]
    return TTSResult(
        audio=pcm16_to_wav(pcm, sample_rate),
        time_to_first_byte_ms=ttfb_ms if ttfb_ms is not None else total_ms,
        total_ms=total_ms,
    )
