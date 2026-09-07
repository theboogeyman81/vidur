from agent.providers.base import STTResult


class DeepgramSTT:
    name = "deepgram-nova"

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        raise NotImplementedError("DeepgramSTT full impl lands in Phase 4")
