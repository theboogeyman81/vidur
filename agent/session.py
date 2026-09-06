from __future__ import annotations

import os
import pathlib
import time

import structlog
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import google, silero

from agent.providers.stt.sarvam import SarvamSTT
from agent.providers.tts.sarvam import SarvamLKTTS

log = structlog.get_logger()

_PROMPT_PATH = pathlib.Path(__file__).parent / "prompts" / "tutor_v1.md"


async def run_session(ctx: JobContext) -> None:
    await ctx.connect()

    system_prompt = _PROMPT_PATH.read_text()
    stt = SarvamSTT()
    tts = SarvamLKTTS()

    session = AgentSession(
        stt=stt,
        llm=google.LLM(
            model="gemini-2.5-flash",
            api_key=os.environ["GEMINI_API_KEY"],
            max_output_tokens=300,
            temperature=0.7,
        ),
        tts=tts,
        vad=silero.VAD.load(),
    )

    _turn_start: dict[str, float] = {"t": 0.0}

    @session.on("user_input_transcribed")
    def _on_transcribed(ev) -> None:
        if not ev.is_final:
            return
        _turn_start["t"] = time.perf_counter()
        log.info(
            "turn.stt",
            transcript=ev.transcript,
            stt_ms=round(stt.last_latency_ms, 1),
            stt_engine=stt.name,
        )

    @session.on("agent_state_changed")
    def _on_state_changed(ev) -> None:
        # log a complete turn trace when the agent transitions back to listening/idle
        if ev.old_state == "speaking" and ev.new_state in ("listening", "idle"):
            elapsed = (time.perf_counter() - _turn_start["t"]) * 1000
            log.info(
                "turn",
                stt_ms=round(stt.last_latency_ms, 1),
                stt_engine=stt.name,
                tts_ttfb_ms=round(tts.last_ttfb_ms, 1),
                tts_engine=tts.name,
                llm_first_token_ms=0.0,  # TODO(phase2): wire via Langfuse
                llm_total_ms=0.0,
                tools_called=[],
                e2e_ms=round(elapsed + stt.last_latency_ms, 1),
                interrupted=False,
            )

    await session.start(
        agent=Agent(instructions=system_prompt),
        room=ctx.room,
    )
