from __future__ import annotations

import json
import logging
import time
import uuid

from livekit.agents import Agent, AgentSession, JobContext
from livekit.agents import vad as lk_vad
from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    LLMStream,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from agent.providers.stt.sarvam import SarvamSTT
from agent.providers.tts.sarvam import SarvamLKTTS

log = logging.getLogger(__name__)


def _trace(**fields) -> None:
    log.info(json.dumps(fields))


class _PydanticAIStream(LLMStream):
    def __init__(
        self,
        llm: PydanticAILLM,
        *,
        chat_ctx: ChatContext,
        tools: list,
        conn_options: APIConnectOptions,
        session_id: str,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._session_id = session_id

    async def _run(self) -> None:
        from agent.brain import vidur_agent

        user_text = ""
        for msg in reversed(self._chat_ctx.messages()):
            if msg.role == "user":
                user_text = msg.raw_text_content or ""
                break

        result = await vidur_agent.run(
            user_text,
            deps={"session_id": self._session_id},
        )

        self._event_ch.send_nowait(
            ChatChunk(
                id=str(uuid.uuid4()),
                delta=ChoiceDelta(role="assistant", content=result.output),
            )
        )


class PydanticAILLM(LLM):
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self._session_id = session_id

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs,
    ) -> LLMStream:
        return _PydanticAIStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            session_id=self._session_id,
        )


async def run_session(ctx: JobContext, *, vad: lk_vad.VAD) -> None:
    await ctx.connect()

    stt = SarvamSTT()
    tts = SarvamLKTTS()

    session = AgentSession(
        stt=stt,
        llm=PydanticAILLM(session_id=ctx.room.name),
        tts=tts,
        vad=vad,
    )

    _turn_start: dict[str, float] = {"t": 0.0}

    @session.on("user_input_transcribed")
    def _on_transcribed(ev) -> None:
        if not ev.is_final:
            return
        _turn_start["t"] = time.perf_counter()
        _trace(
            event="turn.stt",
            transcript=ev.transcript,
            stt_ms=round(stt.last_latency_ms, 1),
            stt_engine=stt.name,
        )

    @session.on("agent_state_changed")
    def _on_state_changed(ev) -> None:
        if ev.old_state == "speaking" and ev.new_state in ("listening", "idle"):
            elapsed = (time.perf_counter() - _turn_start["t"]) * 1000
            _trace(
                event="turn",
                stt_ms=round(stt.last_latency_ms, 1),
                stt_engine=stt.name,
                tts_ttfb_ms=round(tts.last_ttfb_ms, 1),
                tts_engine=tts.name,
                llm_first_token_ms=0.0,  # TODO 2.10: wire via Langfuse
                llm_total_ms=0.0,
                tools_called=[],
                e2e_ms=round(elapsed + stt.last_latency_ms, 1),
                interrupted=False,
            )

    import pathlib

    system_prompt = (pathlib.Path(__file__).parent / "prompts" / "tutor_v1.md").read_text()

    await session.start(
        agent=Agent(instructions=system_prompt),
        room=ctx.room,
    )
