import os

from agent.providers.stt.deepgram import DeepgramSTT
from agent.providers.stt.google import GoogleSTT
from agent.providers.stt.sarvam import SarvamSTT
from agent.providers.stt.whisper import WhisperSTT

_REGISTRY: dict[str, type] = {
    "sarvam": SarvamSTT,
    "deepgram": DeepgramSTT,
    "whisper": WhisperSTT,
    "google": GoogleSTT,
}


def get_stt_provider():
    key = os.getenv("STT_ENGINE", "sarvam")
    cls = _REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown STT engine: {key!r}. Options: {list(_REGISTRY)}")
    return cls()
