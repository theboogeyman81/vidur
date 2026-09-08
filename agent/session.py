from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import time
import uuid

from langfuse import Langfuse
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
from pydantic_ai.messages import ModelResponse, ToolCallPart

from agent.providers.stt.sarvam import SarvamSTT
from agent.providers.tts.sarvam import SarvamLKTTS
from api.db import log_session, log_turn

log = logging.getLogger(__name__)

try:
    _lf: Langfuse | None = Langfuse()
except Exception:
    _lf = None


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
        active_trace: dict,
        llm_metrics: dict,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._session_id = session_id
        self._active_trace = active_trace
        self._llm_metrics = llm_metrics

    async def _run(self) -> None:
        from agent.brain import vidur_agent

        user_text = ""
        for msg in reversed(self._chat_ctx.messages()):
            if msg.role == "user":
                user_text = msg.raw_text_content or ""
                break

        t0 = time.perf_counter()
        result = await vidur_agent.run(
            user_text,
            deps={"session_id": self._session_id},
        )
        llm_total_ms = round((time.perf_counter() - t0) * 1000, 1)

        tools_called = [
            part.tool_name
            for msg in result.new_messages()
            if isinstance(msg, ModelResponse)
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]

        self._llm_metrics["llm_total_ms"] = llm_total_ms
        self._llm_metrics["tools_called"] = tools_called

        trace = self._active_trace.get("ref")
        if trace:
            trace.span(
                name="llm",
                output={"llm_total_ms": llm_total_ms, "tools_called": tools_called},
            )

        self._event_ch.send_nowait(
            ChatChunk(
                id=str(uuid.uuid4()),
                delta=ChoiceDelta(role="assistant", content=result.output),
            )
        )


class PydanticAILLM(LLM):
    def __init__(
        self,
        *,
        session_id: str,
        active_trace: dict,
        llm_metrics: dict,
    ) -> None:
        super().__init__()
        self._session_id = session_id
        self._active_trace = active_trace
        self._llm_metrics = llm_metrics

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
            active_trace=self._active_trace,
            llm_metrics=self._llm_metrics,
        )


async def run_session(ctx: JobContext, *, vad: lk_vad.VAD) -> None:
    await ctx.connect()
    await log_session(ctx.room.name)

    stt = SarvamSTT()
    tts = SarvamLKTTS()

    _active_trace: dict = {}
    _llm_metrics: dict = {}
    _turn_start: dict[str, float] = {"t": 0.0}

    session = AgentSession(
        stt=stt,
        llm=PydanticAILLM(
            session_id=ctx.room.name,
            active_trace=_active_trace,
            llm_metrics=_llm_metrics,
        ),
        tts=tts,
        vad=vad,
    )

    @session.on("user_input_transcribed")
    def _on_transcribed(ev) -> None:
        if not ev.is_final:
            return
        _turn_start["t"] = time.perf_counter()
        trace = _lf.trace(name="turn", session_id=ctx.room.name) if _lf else None
        _active_trace["ref"] = trace
        if trace:
            trace.span(
                name="stt",
                output={
                    "transcript": ev.transcript,
                    "latency_ms": round(stt.last_latency_ms, 1),
                    "engine": stt.name,
                },
            )
        _trace(event="turn.stt", transcript=ev.transcript, stt_ms=round(stt.last_latency_ms, 1))

    @session.on("agent_state_changed")
    def _on_state_changed(ev) -> None:
        if ev.old_state == "speaking" and ev.new_state in ("listening", "idle"):
            elapsed = (time.perf_counter() - _turn_start["t"]) * 1000
            llm_total = _llm_metrics.pop("llm_total_ms", 0.0)
            tools = _llm_metrics.pop("tools_called", [])
            turn_data = {
                "stt_ms": round(stt.last_latency_ms, 1),
                "stt_engine": stt.name,
                "tts_ttfb_ms": round(tts.last_ttfb_ms, 1),
                "tts_engine": tts.name,
                "llm_first_token_ms": llm_total,
                "llm_total_ms": llm_total,
                "tools_called": tools,
                "e2e_ms": round(elapsed + stt.last_latency_ms, 1),
                "interrupted": False,
            }
            trace = _active_trace.get("ref")
            if trace:
                trace.span(
                    name="tts",
                    output={"ttfb_ms": turn_data["tts_ttfb_ms"], "engine": tts.name},
                )
                trace.update(output=turn_data)
            _trace(event="turn", **turn_data)
            asyncio.create_task(log_turn(ctx.room.name, turn_data))

    system_prompt = (pathlib.Path(__file__).parent / "prompts" / "tutor_v1.md").read_text()

    await session.start(
        agent=Agent(instructions=system_prompt),
        room=ctx.room,
    )
