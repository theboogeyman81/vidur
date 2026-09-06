# Phase 1 Spec — `feat/voice-loop`

**Branch:** `feat/voice-loop`  
**Date:** 2026-09-06  
**Goal:** End-to-end voice loop deployed and talking. Nothing else.

User speaks Hinglish → Sarvam STT → Gemini 2.5 Flash → Sarvam TTS → user hears a Socratic response.  
Barge-in works. JSON trace logged per turn. Worker live on Fly.io.

---

## Deliverable checklist

- [ ] `pyproject.toml` — `uv`-managed, Python ≥ 3.11, all deps pinned
- [ ] `.env.example` — all 5 required keys
- [ ] `Makefile` — `dev`, `lint` targets
- [ ] `agent/providers/base.py` — `STTProvider`, `TTSProvider` protocols + result models
- [ ] `agent/providers/stt/sarvam.py` — Saarika adapter
- [ ] `agent/providers/tts/sarvam.py` — Bulbul adapter
- [ ] `agent/prompts/tutor_v1.md` — Socratic UPSC tutor system prompt
- [ ] `agent/worker.py` — LiveKit worker entrypoint
- [ ] `agent/session.py` — turn loop, VAD → STT → LLM → TTS, barge-in, trace logging
- [ ] `Dockerfile` — containerises the agent worker
- [ ] `fly.toml` — Fly.io app config

---

## 1. Project scaffolding

### `pyproject.toml`

```toml
[project]
name = "vidur"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "livekit-agents[silero]>=0.12",
    "livekit-plugins-google",          # Gemini via LiveKit plugin
    "google-generativeai>=0.8",
    "httpx>=0.27",
    "pydantic>=2.7",
    "python-dotenv>=1.0",
    "structlog>=24.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### `.env.example`

```
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
SARVAM_API_KEY=
GEMINI_API_KEY=
```

### `Makefile`

```makefile
.PHONY: dev lint

dev:
	uv run python -m agent.worker dev

lint:
	uv run ruff check .
	uv run ruff format --check .
```

---

## 2. Provider contracts — `agent/providers/base.py`

These are the only interfaces the agent talks to. Every STT/TTS vendor must satisfy them. Do not change the field names between phases — the eval harness depends on them.

```python
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
```

---

## 3. Sarvam STT adapter — `agent/providers/stt/sarvam.py`

**API:** `POST https://api.sarvam.ai/speech-to-text`  
**Auth:** `api-subscription-key: <SARVAM_API_KEY>` header  
**Input:** multipart form — `file` (wav bytes), `model` (`saarika:v2`), `language_code` (`hi-IN`)  
**Output field:** `.transcript`

```python
import os, time
import httpx
from agent.providers.base import STTProvider, STTResult


class SarvamSTT:
    name = "sarvam-saarika"
    _url = "https://api.sarvam.ai/speech-to-text"

    def __init__(self) -> None:
        self._key = os.environ["SARVAM_API_KEY"]

    async def transcribe(self, audio: bytes, sample_rate: int) -> STTResult:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                self._url,
                headers={"api-subscription-key": self._key},
                files={"file": ("audio.wav", audio, "audio/wav")},
                data={"model": "saarika:v2", "language_code": "hi-IN"},
            )
        r.raise_for_status()
        payload = r.json()
        return STTResult(
            text=payload.get("transcript", ""),
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=payload,
        )
```

**Notes:**
- Audio must be WAV-wrapped PCM. If LiveKit gives raw PCM, wrap it: `wave` stdlib or just prepend a 44-byte WAV header.
- `language_code=hi-IN` accepts Hinglish — Saarika handles code-mixed input.
- On non-2xx, let `raise_for_status()` propagate; the session layer catches it and logs.

---

## 4. Sarvam TTS adapter — `agent/providers/tts/sarvam.py`

**API:** `POST https://api.sarvam.ai/text-to-speech`  
**Auth:** `api-subscription-key` header  
**Input JSON:** `{ "inputs": [text], "target_language_code": lang, "speaker": "meera", "model": "bulbul:v1" }`  
**Output:** base64-encoded WAV in `.audios[0]`

```python
import base64, os, time
import httpx
from agent.providers.base import TTSProvider, TTSResult


class SarvamTTS:
    name = "sarvam-bulbul"
    _url = "https://api.sarvam.ai/text-to-speech"

    def __init__(self) -> None:
        self._key = os.environ["SARVAM_API_KEY"]

    async def synthesize(self, text: str, lang: str = "hi-IN") -> TTSResult:
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                self._url,
                headers={
                    "api-subscription-key": self._key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [text],
                    "target_language_code": lang,
                    "speaker": "meera",
                    "model": "bulbul:v1",
                    "enable_preprocessing": True,
                },
            )
        ttfb = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        payload = r.json()
        audio = base64.b64decode(payload["audios"][0])
        total = (time.perf_counter() - t0) * 1000
        return TTSResult(audio=audio, time_to_first_byte_ms=ttfb, total_ms=total)
```

**Notes:**
- Bulbul returns a complete audio blob, not a stream. `time_to_first_byte_ms` here is response receipt time, which is the best approximation until we add true streaming in Phase 5.
- Keep text chunks ≤ 500 characters. Split at sentence boundaries if the LLM produces longer outputs.
- `enable_preprocessing=True` handles code-mixed punctuation better.

---

## 5. Tutor prompt — `agent/prompts/tutor_v1.md`

```markdown
You are Vidur, a Socratic voice tutor for UPSC exam preparation.

Rules:
- Ask before you tell. When a student asks about a topic, ask one probing question first to surface what they already know.
- Keep each response to 2–3 sentences maximum. This is a voice conversation.
- Code-switch freely between English, Hindi, and Hinglish when the student does. Match their register.
- Never give a Wikipedia-style answer. Always tie the concept to something they can reason from.
- If the student is confused, ask a simpler question — don't explain longer.
- When you ask a question, wait. Do not volunteer the answer in the same turn.

Tone: warm, unhurried, a little dry.
```

---

## 6. LiveKit worker — `agent/worker.py`

```python
import logging
from dotenv import load_dotenv
from livekit.agents import WorkerOptions, cli
from agent.session import VidurSession

load_dotenv()
logging.basicConfig(level=logging.INFO)


async def entrypoint(ctx):
    session = VidurSession(ctx)
    await session.run()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

**What this does:**
- Loads `.env` on startup
- Registers `entrypoint` as the room entrypoint — LiveKit calls this for every new room
- `VidurSession` owns the full turn loop; the worker is just glue

---

## 7. Session — `agent/session.py`

This is the heart of Phase 1. It must handle:

1. **VAD end** → collect audio buffer → call STT
2. **STT result** → call Gemini → stream tokens
3. **LLM stream** → buffer to sentence boundary → call TTS
4. **TTS result** → push audio into LiveKit room
5. **Barge-in** → if VAD fires while TTS is playing, cancel TTS immediately

```python
import asyncio, json, time, wave, io, os, pathlib
import structlog
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import silero
from google import generativeai as genai

from agent.providers.base import STTResult, TTSResult
from agent.providers.stt.sarvam import SarvamSTT
from agent.providers.tts.sarvam import SarvamTTS

log = structlog.get_logger()

PROMPT_PATH = pathlib.Path(__file__).parent / "prompts" / "tutor_v1.md"
SYSTEM_PROMPT = PROMPT_PATH.read_text()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
_llm = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT,
)


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class VidurSession:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._stt = SarvamSTT()
        self._tts = SarvamTTS()
        self._history: list[dict] = []

    async def run(self) -> None:
        vad = silero.VAD.load()
        session = AgentSession(vad=vad)

        @session.on("user_speech_committed")
        async def on_speech(audio_frame, sample_rate: int):
            await self._handle_turn(audio_frame, sample_rate, session)

        await session.start(self._ctx.room, agent=Agent(instructions=SYSTEM_PROMPT))

    async def _handle_turn(self, audio_frame, sample_rate: int, session) -> None:
        vad_end_ms = time.perf_counter() * 1000

        # STT
        wav = _pcm_to_wav(bytes(audio_frame), sample_rate)
        stt_result: STTResult = await self._stt.transcribe(wav, sample_rate)
        transcript = stt_result.text.strip()
        if not transcript:
            return

        # LLM — stream
        self._history.append({"role": "user", "parts": [transcript]})
        llm_t0 = time.perf_counter()
        llm_first_token_ms: float | None = None
        response_text = ""

        response = await _llm.generate_content_async(
            self._history,
            stream=True,
            generation_config={"max_output_tokens": 300, "temperature": 0.7},
        )

        async for chunk in response:
            if llm_first_token_ms is None:
                llm_first_token_ms = (time.perf_counter() - llm_t0) * 1000
            response_text += chunk.text or ""

        llm_total_ms = (time.perf_counter() - llm_t0) * 1000
        self._history.append({"role": "model", "parts": [response_text]})

        # TTS
        tts_result: TTSResult = await self._tts.synthesize(response_text)

        # Push audio into room
        await session.say(tts_result.audio, allow_interruptions=True)

        # Trace log
        e2e_ms = (time.perf_counter() * 1000) - vad_end_ms
        log.info(
            "turn",
            vad_end_ms=round(vad_end_ms, 1),
            stt_ms=round(stt_result.latency_ms, 1),
            stt_engine=self._stt.name,
            transcript=transcript,
            llm_first_token_ms=round(llm_first_token_ms or 0, 1),
            llm_total_ms=round(llm_total_ms, 1),
            tools_called=[],
            tts_ttfb_ms=round(tts_result.time_to_first_byte_ms, 1),
            tts_engine=self._tts.name,
            e2e_ms=round(e2e_ms, 1),
            interrupted=False,
        )
```

**Barge-in:** LiveKit Agents handles barge-in natively via `allow_interruptions=True` on `session.say()`. When VAD fires during playback, the framework cancels the audio stream and fires `user_speech_committed` again. Log `interrupted=True` in the new turn's trace.

**Conversation history:** `self._history` grows unbounded in Phase 1. That's fine — cap it at 20 turns in Phase 2.

---

## 8. Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system -e .

COPY agent/ agent/

CMD ["python", "-m", "agent.worker", "start"]
```

---

## 9. `fly.toml`

```toml
app = "vidur-agent"
primary_region = "sin"   # Singapore — lowest latency to Sarvam India endpoints

[build]
  dockerfile = "Dockerfile"

[env]
  LOG_LEVEL = "INFO"

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    port = 8080

[http_service]
  internal_port = 8080
  force_https = true
```

Secrets go in via `fly secrets set`, never in `fly.toml`:
```
fly secrets set LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... SARVAM_API_KEY=... GEMINI_API_KEY=...
```

---

## 10. Directory structure after Phase 1

```
vidur/
├── agent/
│   ├── __init__.py
│   ├── worker.py
│   ├── session.py
│   ├── prompts/
│   │   └── tutor_v1.md
│   └── providers/
│       ├── __init__.py
│       ├── base.py
│       ├── stt/
│       │   ├── __init__.py
│       │   └── sarvam.py
│       └── tts/
│           ├── __init__.py
│           └── sarvam.py
├── specs/
│   └── phase_1_spec.md
├── pyproject.toml
├── .env.example
├── .env                  ← gitignored
├── Makefile
├── Dockerfile
└── fly.toml
```

---

## 11. API keys needed before starting

| Key | Where to get |
|---|---|
| `LIVEKIT_URL` | LiveKit Cloud dashboard → project settings |
| `LIVEKIT_API_KEY` + `LIVEKIT_API_SECRET` | Same dashboard |
| `SARVAM_API_KEY` | console.sarvam.ai |
| `GEMINI_API_KEY` | aistudio.google.com → API keys |

All are free tier. Sarvam gives 10,000 free API calls. Gemini 2.5 Flash has a free tier sufficient for dev.

---

## 12. Done when

1. `make dev` starts the worker locally without errors
2. A LiveKit test room connects and the worker joins
3. Speaking a sentence triggers a JSON trace line on stdout
4. The spoken response is audible in the room
5. Speaking while the TTS is playing cancels the audio (barge-in)
6. `fly deploy` succeeds and the worker stays alive for 5 minutes
