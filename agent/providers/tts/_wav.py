"""WAV helpers shared by the TTS adapters. Every adapter returns a complete PCM16 WAV (spec D4)."""

from __future__ import annotations

import io
import wave


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw little-endian PCM16 in a WAV header. No resampling."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_info(wav: bytes) -> tuple[int, float]:
    """Return (sample_rate, duration_s) from WAV bytes."""
    with wave.open(io.BytesIO(wav)) as wf:
        sr = wf.getframerate()
        return sr, wf.getnframes() / sr


def wav_to_pcm16(wav: bytes) -> tuple[bytes, int, int]:
    """Return (pcm, sample_rate, channels) from WAV bytes."""
    with wave.open(io.BytesIO(wav)) as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()
