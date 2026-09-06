from agent.providers.base import STTResult


class GoogleSTT:
    name = "google-stt"

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        raise NotImplementedError("GoogleSTT full impl lands in Phase 4")
