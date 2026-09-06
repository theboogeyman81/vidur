from agent.providers.base import STTResult


class WhisperSTT:
    name = "whisper-large-v3"

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        raise NotImplementedError("WhisperSTT full impl lands in Phase 4")
