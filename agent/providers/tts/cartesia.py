from agent.providers.base import TTSResult


class CartesiaTTS:
    name = "cartesia-sonic"

    async def synthesize(self, text: str, lang: str) -> TTSResult:
        raise NotImplementedError("CartesiaTTS full impl lands in Phase 5")
