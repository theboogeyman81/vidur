"""STT eval runner — WER / CER / latency across engines on the Hinglish clip set.

Usage:
    uv run python -m evals.run_stt_eval --engines sarvam,deepgram,google,whisper
    uv run python -m evals.run_stt_eval --engines deepgram --limit 3 --lang hi-en

Writes evals/results/stt_{timestamp}.json. Never overwrites an existing results file.
Engines run sequentially, clips one at a time, with one discarded warm-up call per
engine — concurrency would contaminate the latency numbers (spec D7).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import pathlib
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from agent.providers.stt import _REGISTRY
from evals.datasets.stt_hinglish.models import MANIFEST, ManifestRow
from evals.datasets.stt_hinglish.validate import ManifestError, validate
from evals.metrics.latency import percentile
from evals.metrics.wer import compute_cer, compute_wer, corpus_cer, corpus_wer, romanize

load_dotenv()

_RESULTS_DIR = pathlib.Path(__file__).parent / "results"
_WORST_N = 5


class ClipResult(BaseModel):
    file: str
    lang: str
    condition: str
    ref: str
    hyp: str
    wer: float
    cer: float
    cer_roman: float
    latency_ms: float | None
    rtf: float | None  # latency / audio duration
    error: str | None = None


class EngineResult(BaseModel):
    engine: str
    status: Literal["ok", "skipped"]
    skip_reason: str | None = None
    config: dict = {}
    wer: float | None = None
    cer: float | None = None
    cer_roman: float | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    failures: int = 0
    by_lang: dict[str, float] = {}
    by_condition: dict[str, float] = {}
    per_clip: list[ClipResult] = []


class DatasetInfo(BaseModel):
    n_clips: int
    manifest_sha256: str
    by_lang: dict[str, int]
    by_condition: dict[str, int]


class STTRun(BaseModel):
    run_at: str
    git_sha: str
    dataset: DatasetInfo
    engines: list[EngineResult]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _group_wer(clips: list[ClipResult], key: str) -> dict[str, float]:
    groups: dict[str, list[ClipResult]] = defaultdict(list)
    for c in clips:
        groups[getattr(c, key)].append(c)
    return {
        k: round(corpus_wer([c.hyp for c in g], [c.ref for c in g]), 4)
        for k, g in sorted(groups.items())
    }


async def _run_engine(key: str, rows: list[ManifestRow], data_dir: pathlib.Path) -> EngineResult:
    try:
        provider = _REGISTRY[key]()
    except Exception as exc:  # missing API key, missing credentials, etc.
        print(f"  [{key}] skipped: {exc!r}")
        return EngineResult(engine=key, status="skipped", skip_reason=repr(exc))

    name = provider.name
    print(f"\n[{name}] warm-up...")
    try:
        await provider.transcribe((data_dir / rows[0].file).read_bytes(), 16000)
    except Exception as exc:
        print(f"  warm-up failed: {exc!r}")

    clips: list[ClipResult] = []
    for i, row in enumerate(rows, 1):
        wav = (data_dir / row.file).read_bytes()
        hyp, latency, error = "", None, None
        try:
            result = await provider.transcribe(wav, 16000)
            hyp, latency = result.text, result.latency_ms
        except Exception as exc:  # counts as an empty hypothesis (spec D8)
            error = repr(exc)

        clip = ClipResult(
            file=row.file,
            lang=row.lang,
            condition=row.condition,
            ref=row.transcript,
            hyp=hyp,
            wer=round(compute_wer(hyp, row.transcript), 4),
            cer=round(compute_cer(hyp, row.transcript), 4),
            cer_roman=round(compute_cer(romanize(hyp), romanize(row.transcript)), 4),
            latency_ms=round(latency, 1) if latency is not None else None,
            rtf=round(latency / 1000 / row.duration_s, 3) if latency is not None else None,
            error=error,
        )
        clips.append(clip)
        lat = f"{clip.latency_ms:.0f}ms" if clip.latency_ms is not None else "ERR"
        print(f"  {i:>3}/{len(rows)} {row.file} wer={clip.wer:.2f} {lat}")

    hyps, refs = [c.hyp for c in clips], [c.ref for c in clips]
    latencies = [c.latency_ms for c in clips if c.latency_ms is not None]
    return EngineResult(
        engine=name,
        status="ok",
        config=dict(getattr(provider, "config", {})),
        wer=round(corpus_wer(hyps, refs), 4),
        cer=round(corpus_cer(hyps, refs), 4),
        cer_roman=round(corpus_cer([romanize(h) for h in hyps], [romanize(r) for r in refs]), 4),
        p50_ms=round(percentile(latencies, 50), 1) if latencies else None,
        p95_ms=round(percentile(latencies, 95), 1) if latencies else None,
        failures=sum(c.error is not None for c in clips),
        by_lang=_group_wer(clips, "lang"),
        by_condition=_group_wer(clips, "condition"),
        per_clip=clips,
    )


def _fmt(v: float | None, digits: int = 3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def _print_summary(run: STTRun) -> None:
    print("\n| Engine | WER | CER | CER (roman) | hi-en WER | p50 ms | p95 ms | fails |")
    print("|---|---|---|---|---|---|---|---|")
    for e in run.engines:
        if e.status == "skipped":
            print(f"| {e.engine} | skipped | | | | | | |")
            continue
        print(
            f"| {e.engine} | {_fmt(e.wer)} | {_fmt(e.cer)} | {_fmt(e.cer_roman)} "
            f"| {_fmt(e.by_lang.get('hi-en'))} | {_fmt(e.p50_ms, 0)} | {_fmt(e.p95_ms, 0)} "
            f"| {e.failures} |"
        )

    for e in run.engines:
        if e.status == "skipped":
            continue
        print(f"\nWorst {_WORST_N} — {e.engine}")
        for c in sorted(e.per_clip, key=lambda c: c.wer, reverse=True)[:_WORST_N]:
            print(f"  {c.file} ({c.lang}/{c.condition}) wer={c.wer:.2f}")
            print(f"    ref: {c.ref}")
            print(f"    hyp: {c.hyp or '<empty>'}" + (f"  [{c.error}]" if c.error else ""))


def _write_results(run: STTRun) -> pathlib.Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = _RESULTS_DIR / f"stt_{ts}.json"
    with out_path.open("x", encoding="utf-8") as f:  # "x" fails loudly rather than overwrite
        f.write(run.model_dump_json(indent=2))
    return out_path


async def main(engines: list[str], manifest: pathlib.Path, limit: int | None, lang: str | None):
    unknown = [e for e in engines if e not in _REGISTRY]
    if unknown:
        sys.exit(f"unknown engine(s) {unknown}. Options: {list(_REGISTRY)}")
    try:
        rows = validate(manifest)
    except ManifestError as exc:
        sys.exit(f"manifest invalid — fix before spending API calls:\n{exc}")

    if lang:
        rows = [r for r in rows if r.lang == lang]
    if limit:
        rows = rows[:limit]
    if not rows:
        sys.exit("no clips selected")
    print(f"{len(rows)} clips × {len(engines)} engines")

    results = [await _run_engine(key, rows, manifest.parent) for key in engines]
    run = STTRun(
        run_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        dataset=DatasetInfo(
            n_clips=len(rows),
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            by_lang=dict(Counter(r.lang for r in rows)),
            by_condition=dict(Counter(r.condition for r in rows)),
        ),
        engines=results,
    )
    out_path = _write_results(run)
    _print_summary(run)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", required=True, help="comma-separated registry keys")
    parser.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lang", choices=["hi-en", "hi", "en-IN"], default=None)
    args = parser.parse_args()
    asyncio.run(main(args.engines.split(","), args.manifest, args.limit, args.lang))
