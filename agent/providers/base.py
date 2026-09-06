from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class STTResult(BaseModel):
    text: str
    latency_ms: float
    raw: dict


class TTSResult(BaseModel):
    audio: bytes
    time_to_first_byte_ms: float
    total_ms: float


@runtime_checkable
class STTProvider(Protocol):
    name: str

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult: ...


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    async def synthesize(self, text: str, lang: str) -> TTSResult: ...
