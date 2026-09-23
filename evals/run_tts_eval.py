"""TTS eval runner — TTFB / total latency + round-trip intelligibility across engines.

Usage:
    uv run python -m evals.run_tts_eval --engines sarvam,cartesia,elevenlabs,piper
    uv run python -m evals.run_tts_eval --engines piper --limit 3 --no-roundtrip --yes

Two passes, never interleaved: (1) synthesize every sentence × engine, saving WAVs;
(2) transcribe every saved WAV with one fixed judge STT and score WER against the input
(spec D6). Engines run sequentially with one discarded warm-up call (spec D8).
Writes evals/results/tts_{timestamp}.json. Never overwrites an existing results file.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from agent.providers.stt import _REGISTRY as STT_REGISTRY
from agent.providers.tts import _REGISTRY
from evals.datasets.tts_codemix.models import (
    BCP47,
    SENTENCES,
    Script,
    SentenceRow,
    entity_hits,
)
from evals.datasets.tts_codemix.validate import SentencesError, validate
from evals.metrics.audio import wav_info
from evals.metrics.latency import percentile
from evals.metrics.wer import compute_wer, corpus_wer
from evals.run_stt_eval import _fmt, _git_sha

load_dotenv()

_RESULTS_DIR = pathlib.Path(__file__).parent / "results"
_WORST_N = 5
_CONFIRM_CHARS = 2000  # per engine; guards the ElevenLabs free-tier quota


class SentenceResult(BaseModel):
    id: str
    script: Script
    mix: str
    length: str
    text: str
    audio_path: str | None = None  # relative to evals/results/
    ttfb_ms: float | None = None
    total_ms: float | None = None
    rtf: float | None = None  # total / audio duration
    duration_s: float | None = None
    sample_rate: int | None = None
    roundtrip_hyp: str | None = None
    roundtrip_wer: float | None = None
    entities_hit: int = 0
    entities_total: int = 0
    error: str | None = None
    judge_error: str | None = None


class EngineResult(BaseModel):
    engine: str
    status: Literal["ok", "skipped"]
    skip_reason: str | None = None
    config: dict = {}
    p50_ttfb_ms: float | None = None
    p95_ttfb_ms: float | None = None
    p50_total_ms: float | None = None
    p95_total_ms: float | None = None
    roundtrip_wer: float | None = None
    entity_hit_rate: float | None = None
    failures: int = 0
    by_mix: dict[str, float] = {}
    by_script: dict[str, float] = {}
    per_sentence: list[SentenceResult] = []


class JudgeInfo(BaseModel):
    engine: str
    phase4_clean_wer: float | None = None


class DatasetInfo(BaseModel):
    n_sentences: int
    n_roman: int
    sentences_sha256: str
    by_mix: dict[str, int]


class TTSRun(BaseModel):
    run_at: str
    git_sha: str
    dataset: DatasetInfo
    judge: JudgeInfo | None
    engines: list[EngineResult]


def _jobs(rows: list[SentenceRow], script: str) -> list[tuple[SentenceRow, Script]]:
    scripts: list[Script] = ["native", "roman"] if script == "both" else [script]  # type: ignore[list-item]
    return [(r, s) for s in scripts for r in rows if r.text_for(s)]


async def _synthesize_engine(
    key: str, jobs: list[tuple[SentenceRow, Script]], audio_dir: pathlib.Path
) -> tuple[EngineResult, list[SentenceResult]]:
    try:
        provider = _REGISTRY[key]()
    except Exception as exc:  # missing API key, missing model file, etc.
        print(f"  [{key}] skipped: {exc!r}")
        return EngineResult(engine=key, status="skipped", skip_reason=repr(exc)), []

    name = provider.name
    first = jobs[0][0]
    print(f"\n[{name}] warm-up...")
    try:
        await provider.synthesize(first.text, BCP47[first.lang])
    except Exception as exc:
        print(f"  warm-up failed: {exc!r}")

    engine_dir = audio_dir / name
    engine_dir.mkdir(parents=True, exist_ok=True)
    results: list[SentenceResult] = []
    for i, (row, script) in enumerate(jobs, 1):
        text = row.text_for(script) or ""
        res = SentenceResult(
            id=row.id,
            script=script,
            mix=row.mix,
            length=row.length,
            text=text,
            entities_total=len(row.entities),
        )
        try:
            out = await provider.synthesize(text, BCP47[row.lang])
            wav_path = engine_dir / f"{row.id}_{script}.wav"
            wav_path.write_bytes(out.audio)
            sr, dur = wav_info(out.audio)
            res.audio_path = str(wav_path.relative_to(_RESULTS_DIR))
            res.ttfb_ms = round(out.time_to_first_byte_ms, 1)
            res.total_ms = round(out.total_ms, 1)
            res.sample_rate, res.duration_s = sr, round(dur, 3)
            res.rtf = round(out.total_ms / 1000 / dur, 3) if dur else None
        except Exception as exc:  # counts as a failure, scored WER 1.0 (spec D9)
            res.error = repr(exc)
        results.append(res)
        lat = f"ttfb={res.ttfb_ms:.0f}ms total={res.total_ms:.0f}ms" if res.error is None else "ERR"
        print(f"  {i:>3}/{len(jobs)} {row.id}/{script} {lat}")

    return EngineResult(engine=name, status="ok", config=dict(getattr(provider, "config", {}))), (
        results
    )


async def _roundtrip(
    judge,
    results: list[SentenceResult],
    native_text: dict[str, str],
    entities: dict[str, list[str]],
) -> None:
    """Transcribe each saved WAV with the judge. Failures stay scored as WER 1.0."""
    for res in results:
        ents = entities[res.id]
        if res.audio_path is None:
            res.roundtrip_hyp, res.roundtrip_wer = "", 1.0
            continue
        wav = (_RESULTS_DIR / res.audio_path).read_bytes()
        try:
            stt = await judge.transcribe(wav, res.sample_rate or 16000)
            res.roundtrip_hyp = stt.text
        except Exception as exc:
            res.judge_error = repr(exc)
            res.roundtrip_hyp = ""
        # Score against the native text for both scripts, so roman vs native is paired
        res.roundtrip_wer = round(compute_wer(res.roundtrip_hyp, native_text[res.id]), 4)
        res.entities_hit = entity_hits(ents, res.roundtrip_hyp)


def _aggregate(
    engine: EngineResult,
    results: list[SentenceResult],
    native_text: dict[str, str],
    roundtrip: bool,
) -> None:
    native = [r for r in results if r.script == "native"]
    ttfbs = [r.ttfb_ms for r in native if r.ttfb_ms is not None]
    totals = [r.total_ms for r in native if r.total_ms is not None]
    engine.p50_ttfb_ms = round(percentile(ttfbs, 50), 1) if ttfbs else None
    engine.p95_ttfb_ms = round(percentile(ttfbs, 95), 1) if ttfbs else None
    engine.p50_total_ms = round(percentile(totals, 50), 1) if totals else None
    engine.p95_total_ms = round(percentile(totals, 95), 1) if totals else None
    engine.failures = sum(r.error is not None for r in results)
    engine.per_sentence = results
    if not roundtrip:
        return

    def cwer(rs: list[SentenceResult]) -> float:
        return round(
            corpus_wer([r.roundtrip_hyp or "" for r in rs], [native_text[r.id] for r in rs]), 4
        )

    engine.roundtrip_wer = cwer(native) if native else None
    ent_total = sum(r.entities_total for r in native)
    engine.entity_hit_rate = (
        round(sum(r.entities_hit for r in native) / ent_total, 4) if ent_total else None
    )

    groups: dict[str, list[SentenceResult]] = defaultdict(list)
    for r in native:
        groups[r.mix].append(r)
    engine.by_mix = {k: cwer(g) for k, g in sorted(groups.items())}

    # Paired: native vs roman on the same sentence ids only
    roman = [r for r in results if r.script == "roman"]
    roman_ids = {r.id for r in roman}
    if roman:
        engine.by_script = {
            "native": cwer([r for r in native if r.id in roman_ids]),
            "roman": cwer(roman),
        }


def _print_summary(run: TTSRun) -> None:
    print(
        "\n| Engine | p50 TTFB | p95 TTFB | p50 total | p95 total "
        "| RT-WER | RT-WER roman | entity hit | fails |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for e in run.engines:
        if e.status == "skipped":
            print(f"| {e.engine} | skipped | | | | | | | |")
            continue
        mark = "" if e.config.get("streaming", True) else "¹"
        print(
            f"| {e.engine} | {_fmt(e.p50_ttfb_ms, 0)}{mark} | {_fmt(e.p95_ttfb_ms, 0)}{mark} "
            f"| {_fmt(e.p50_total_ms, 0)} | {_fmt(e.p95_total_ms, 0)} | {_fmt(e.roundtrip_wer)} "
            f"| {_fmt(e.by_script.get('roman'))} | {_fmt(e.entity_hit_rate)} | {e.failures} |"
        )
    if any(not e.config.get("streaming", True) for e in run.engines):
        print("\n¹ non-streaming: TTFB == total")
    if run.judge:
        print(f"\nJudge STT: {run.judge.engine}")

    for e in run.engines:
        if e.status == "skipped" or e.roundtrip_wer is None:
            continue
        print(f"\nWorst {_WORST_N} — {e.engine}")
        worst = sorted(e.per_sentence, key=lambda r: r.roundtrip_wer or 0, reverse=True)[:_WORST_N]
        for r in worst:
            print(f"  {r.id}/{r.script} ({r.mix}) rt-wer={r.roundtrip_wer:.2f}  {r.audio_path}")
            print(f"    text: {r.text}")
            print(
                f"    heard: {r.roundtrip_hyp or '<empty>'}" + (f"  [{r.error}]" if r.error else "")
            )


def _write_results(run: TTSRun, ts: str) -> pathlib.Path:
    out_path = _RESULTS_DIR / f"tts_{ts}.json"
    with out_path.open("x", encoding="utf-8") as f:  # "x" fails loudly rather than overwrite
        f.write(run.model_dump_json(indent=2))
    return out_path


def _confirm_budget(jobs: list[tuple[SentenceRow, Script]], engines: list[str], yes: bool) -> None:
    chars = sum(len(r.text_for(s) or "") for r, s in jobs)
    print(f"{len(jobs)} syntheses × {len(engines)} engines, {chars} chars per engine")
    hosted = [e for e in engines if e != "piper"]
    if chars > _CONFIRM_CHARS and hosted and not yes:
        answer = input(f"{chars} chars × {hosted} spends real quota. Continue? [y/N] ")
        if answer.strip().lower() != "y":
            sys.exit("aborted")


async def main(args: argparse.Namespace) -> None:
    engines = args.engines.split(",")
    unknown = [e for e in engines if e not in _REGISTRY]
    if unknown:
        sys.exit(f"unknown engine(s) {unknown}. Options: {list(_REGISTRY)}")
    try:
        rows = validate(args.sentences)
    except SentencesError as exc:
        sys.exit(f"sentences invalid — fix before spending API calls:\n{exc}")

    if args.mix:
        rows = [r for r in rows if r.mix == args.mix]
    if args.limit:
        rows = rows[: args.limit]
    jobs = _jobs(rows, args.script)
    if not jobs:
        sys.exit("no sentences selected")

    judge = None
    if not args.no_roundtrip:
        if args.judge not in STT_REGISTRY:
            sys.exit(f"unknown judge {args.judge!r}. Options: {list(STT_REGISTRY)}")
        try:
            judge = STT_REGISTRY[args.judge]()
        except Exception as exc:
            sys.exit(f"judge STT {args.judge!r} failed to start: {exc!r} (or pass --no-roundtrip)")

    _confirm_budget(jobs, engines, args.yes)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    audio_dir = _RESULTS_DIR / "audio" / ts
    synthesized = [await _synthesize_engine(key, jobs, audio_dir) for key in engines]

    native_text = {r.id: r.text for r in rows}
    entities = {r.id: r.entities for r in rows}
    if judge is not None:
        print(f"\nRound-trip pass with judge {judge.name}...")
        for engine, results in synthesized:
            if engine.status == "ok":
                print(f"  [{engine.engine}] {len(results)} clips")
                await _roundtrip(judge, results, native_text, entities)

    for engine, results in synthesized:
        if engine.status == "ok":
            _aggregate(engine, results, native_text, roundtrip=judge is not None)

    run = TTSRun(
        run_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        dataset=DatasetInfo(
            n_sentences=len(rows),
            n_roman=sum(r.text_roman is not None for r in rows) if args.script != "native" else 0,
            sentences_sha256=hashlib.sha256(args.sentences.read_bytes()).hexdigest(),
            by_mix=dict(Counter(r.mix for r in rows)),
        ),
        judge=JudgeInfo(engine=judge.name, phase4_clean_wer=args.judge_clean_wer)
        if judge is not None
        else None,
        engines=[e for e, _ in synthesized],
    )
    out_path = _write_results(run, ts)
    _print_summary(run)
    print(f"\nResults written to {out_path}\nAudio in {audio_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", required=True, help="comma-separated TTS registry keys")
    parser.add_argument("--sentences", type=pathlib.Path, default=SENTENCES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mix", choices=["heavy", "light", "pure_hi", "pure_en", "numeric"])
    parser.add_argument("--script", choices=["native", "roman", "both"], default="both")
    parser.add_argument("--judge", default="deepgram", help="STT registry key (spec D6)")
    parser.add_argument(
        "--judge-clean-wer", type=float, default=None, help="judge's Phase 4 WER, the noise floor"
    )
    parser.add_argument("--no-roundtrip", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the quota confirmation")
    asyncio.run(main(parser.parse_args()))
