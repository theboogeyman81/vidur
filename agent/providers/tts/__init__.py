import os

from agent.providers.tts.cartesia import CartesiaTTS
from agent.providers.tts.elevenlabs import ElevenLabsTTS
from agent.providers.tts.piper import PiperTTS
from agent.providers.tts.sarvam import SarvamTTS

_REGISTRY: dict[str, type] = {
    "sarvam": SarvamTTS,
    "cartesia": CartesiaTTS,
    "elevenlabs": ElevenLabsTTS,
    "piper": PiperTTS,
}


def get_tts_provider():
    key = os.getenv("TTS_ENGINE", "sarvam")
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown TTS engine: {key!r}. Options: {list(_REGISTRY)}")
    return cls()
