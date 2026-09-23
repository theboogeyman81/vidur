"""Barge-in measurement. Pure state machine fed LiveKit event timestamps — no LiveKit imports.

An *episode* opens when the user starts speaking while the agent is speaking, and closes as:
- ``ignored``            — the user stopped and the agent never paused (overlap not handled)
- ``false_interruption`` — the agent paused, then LiveKit classified it as false (backchannel)
- ``interrupted``        — the agent paused and the user's turn went ahead

All timestamps are wall-clock seconds (``ev.created_at``), so they share one clock.
``detected_at`` is when VAD fired, which lags true speech onset: the measured
``interruption_handled_ms`` is a lower bound on what the user experiences.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Outcome = Literal["ignored", "false_interruption", "interrupted"]


class BargeInEpisode(BaseModel):
    detected_at: float
    cancel_at: float | None = None
    user_done_at: float | None = None
    outcome: Outcome | None = None
    resumed: bool | None = None

    @property
    def handled_ms(self) -> float | None:
        if self.cancel_at is None:
            return None
        return round((self.cancel_at - self.detected_at) * 1000, 1)

    @property
    def success(self) -> bool:
        """The agent stopped before the user finished talking."""
        if self.cancel_at is None:
            return False
        return self.user_done_at is None or self.cancel_at <= self.user_done_at


class BargeInMetrics(BaseModel):
    interrupted: bool = False
    interruption_handled_ms: float | None = None
    barge_in_success: bool | None = None  # None when the user never talked over the agent
    false_interruption: bool = False
    false_interruption_resumed: bool | None = None
    episodes: list[dict] = []


class BargeInTracker:
    def __init__(self) -> None:
        self._agent_speaking = False
        self._open: BargeInEpisode | None = None
        self._closed: list[BargeInEpisode] = []

    @property
    def episode_open(self) -> bool:
        return self._open is not None

    @property
    def awaiting_user_turn(self) -> bool:
        """Agent paused for the user; the turn outcome isn't known yet."""
        return self._open is not None and self._open.cancel_at is not None

    def _close(self, outcome: Outcome) -> BargeInEpisode:
        assert self._open is not None
        ep = self._open
        ep.outcome = outcome
        self._closed.append(ep)
        self._open = None
        return ep

    def on_agent_state(self, old: str, new: str, at: float) -> None:
        self._agent_speaking = new == "speaking"
        if old == "speaking" and new != "speaking" and self._open and self._open.cancel_at is None:
            self._open.cancel_at = at

    def on_user_state(self, old: str, new: str, at: float) -> BargeInEpisode | None:
        if new == "speaking" and self._agent_speaking and self._open is None:
            self._open = BargeInEpisode(detected_at=at)
            return None
        if old == "speaking" and new != "speaking" and self._open:
            if self._open.user_done_at is None:
                self._open.user_done_at = at
            if self._open.cancel_at is None and self._agent_speaking:
                return self._close("ignored")
        return None

    def on_false_interruption(self, resumed: bool, at: float) -> BargeInEpisode | None:
        if self._open is None:
            return None
        self._open.resumed = resumed
        return self._close("false_interruption")

    def on_user_turn(self, at: float) -> BargeInEpisode | None:
        """A final user transcript arrived: a paused agent has been interrupted for real."""
        if self.awaiting_user_turn:
            return self._close("interrupted")
        return None

    def pop_turn_metrics(self) -> BargeInMetrics:
        """Summarise episodes closed since the last pop, then reset them."""
        eps, self._closed = self._closed, []
        if not eps:
            return BargeInMetrics()
        cut = next((e for e in eps if e.outcome == "interrupted"), None)
        false = [e for e in eps if e.outcome == "false_interruption"]
        # The episode that defines the turn: the real interruption, else the first overlap
        key = cut or eps[0]
        return BargeInMetrics(
            interrupted=cut is not None,
            interruption_handled_ms=key.handled_ms,
            barge_in_success=key.success,
            false_interruption=bool(false),
            false_interruption_resumed=false[-1].resumed if false else None,
            episodes=[
                {
                    "outcome": e.outcome,
                    "handled_ms": e.handled_ms,
                    "success": e.success,
                    "resumed": e.resumed,
                }
                for e in eps
            ],
        )
