from agent.providers.base import TTSResult


class PiperTTS:
    name = "piper-local"

    async def synthesize(self, text: str, lang: str) -> TTSResult:
        raise NotImplementedError("PiperTTS full impl lands in Phase 5")
