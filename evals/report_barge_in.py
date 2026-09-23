"""Summarise barge-in episodes logged by the live agent (spec 5.7 / D12).

Usage:
    uv run python -m evals.report_barge_in --sessions room-abc,room-def [--label default]

Reads the turns table, counts episodes by outcome, and reports p50/p95 of
interruption_handled_ms. Writes evals/results/bargein_{timestamp}.json (never overwrites).
Report counts, not rates — 10 trials per config is a small sample.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
from collections import Counter
from datetime import UTC, datetime

from pydantic import BaseModel

from api.db import DB_PATH
from evals.metrics.latency import percentile
from evals.run_stt_eval import _fmt, _git_sha

_RESULTS_DIR = pathlib.Path(__file__).parent / "results"


class SessionSummary(BaseModel):
    session_id: str
    interruption_config: dict | None
    turns: int
    episodes: int
    outcomes: dict[str, int]  # interrupted / false_interruption / ignored
    successes: int
    p50_handled_ms: float | None
    p95_handled_ms: float | None


class BargeInReport(BaseModel):
    run_at: str
    git_sha: str
    label: str | None
    sessions: list[SessionSummary]


def summarise(session_id: str, traces: list[dict]) -> SessionSummary:
    episodes = [ep for t in traces for ep in t.get("barge_in_episodes", [])]
    handled = [ep["handled_ms"] for ep in episodes if ep.get("handled_ms") is not None]
    config = next((t["interruption_config"] for t in traces if "interruption_config" in t), None)
    return SessionSummary(
        session_id=session_id,
        interruption_config=config,
        turns=len(traces),
        episodes=len(episodes),
        outcomes=dict(Counter(ep["outcome"] for ep in episodes)),
        successes=sum(bool(ep.get("success")) for ep in episodes),
        p50_handled_ms=round(percentile(handled, 50), 1) if handled else None,
        p95_handled_ms=round(percentile(handled, 95), 1) if handled else None,
    )


def main(sessions: list[str], db: pathlib.Path, label: str | None) -> None:
    con = sqlite3.connect(db)
    summaries = []
    for sid in sessions:
        rows = con.execute("SELECT trace FROM turns WHERE session_id = ? ORDER BY id", (sid,))
        summaries.append(summarise(sid, [json.loads(r[0]) for r in rows]))
    con.close()

    print(
        "| session | min_duration | episodes | interrupted | false | ignored | success "
        "| p50 ms | p95 ms |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        cfg = s.interruption_config or {}
        print(
            f"| {s.session_id} | {cfg.get('min_duration', '?')} | {s.episodes} "
            f"| {s.outcomes.get('interrupted', 0)} | {s.outcomes.get('false_interruption', 0)} "
            f"| {s.outcomes.get('ignored', 0)} | {s.successes}/{s.episodes} "
            f"| {_fmt(s.p50_handled_ms, 0)} | {_fmt(s.p95_handled_ms, 0)} |"
        )

    report = BargeInReport(
        run_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        label=label,
        sessions=summaries,
    )
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RESULTS_DIR / f"bargein_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    with out.open("x", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", required=True, help="comma-separated LiveKit room names")
    parser.add_argument("--db", type=pathlib.Path, default=DB_PATH)
    parser.add_argument("--label", default=None, help="e.g. default / min0.3-backchannel")
    args = parser.parse_args()
    main(args.sessions.split(","), args.db, args.label)
