from agent.providers.base import TTSResult


class ElevenLabsTTS:
    name = "elevenlabs-flash"

    async def synthesize(self, text: str, lang: str) -> TTSResult:
        raise NotImplementedError("ElevenLabsTTS full impl lands in Phase 5")
