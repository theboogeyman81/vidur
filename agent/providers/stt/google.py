from __future__ import annotations

import io
import os
import time
import wave

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.protobuf.json_format import MessageToDict

from agent.providers.base import STTResult

_REGION = "us"  # Chirp 3 is GA in the `us` and `eu` multi-regions only


class GoogleSTT:
    """Google Chirp 3 via Speech-to-Text v2. Supports hi-IN.

    Auth: GOOGLE_APPLICATION_CREDENTIALS (service-account JSON) + GOOGLE_CLOUD_PROJECT.
    """

    name = "google-chirp3"
    config = {"model": "chirp_3", "language_codes": ["hi-IN"], "region": _REGION}

    def __init__(self) -> None:
        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        self._recognizer = f"projects/{project}/locations/{_REGION}/recognizers/_"
        self._client: SpeechAsyncClient | None = None

    def _get_client(self) -> SpeechAsyncClient:
        if self._client is None:
            self._client = SpeechAsyncClient(
                client_options=ClientOptions(api_endpoint=f"{_REGION}-speech.googleapis.com")
            )
        return self._client

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        with wave.open(io.BytesIO(audio)) as wf:
            pcm = wf.readframes(wf.getnframes())
            sr, channels = wf.getframerate(), wf.getnchannels()

        request = cloud_speech.RecognizeRequest(
            recognizer=self._recognizer,
            config=cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=sr,
                    audio_channel_count=channels,
                ),
                language_codes=self.config["language_codes"],
                model=self.config["model"],
                features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=False),
            ),
            content=pcm,
        )
        client = self._get_client()

        t0 = time.perf_counter()
        resp = await client.recognize(request=request)
        latency_ms = (time.perf_counter() - t0) * 1000

        text = " ".join(r.alternatives[0].transcript for r in resp.results if r.alternatives)
        return STTResult(text=text.strip(), latency_ms=latency_ms, raw=MessageToDict(resp._pb))
