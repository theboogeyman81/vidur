"""Check manifest.jsonl before any eval spends API calls.

Usage:
    uv run python -m evals.datasets.stt_hinglish.validate
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

import soundfile as sf

from evals.datasets.stt_hinglish.models import (
    MANIFEST,
    SAMPLE_RATE,
    ManifestRow,
    load_manifest,
)
from evals.metrics.wer import normalize

_MIN_S, _MAX_S = 0.3, 15.0


class ManifestError(Exception):
    pass


def validate(manifest: pathlib.Path = MANIFEST) -> list[ManifestRow]:
    if not manifest.exists():
        raise ManifestError(f"{manifest} not found — record some clips first")
    rows = load_manifest(manifest)
    if not rows:
        raise ManifestError(f"{manifest} is empty")

    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.file in seen:
            errors.append(f"{row.file}: duplicate manifest entry")
        seen.add(row.file)

        wav = manifest.parent / row.file
        if not wav.exists():
            errors.append(f"{row.file}: missing on disk")
            continue
        info = sf.info(wav)
        if info.samplerate != SAMPLE_RATE:
            errors.append(f"{row.file}: {info.samplerate} Hz, want {SAMPLE_RATE}")
        if info.channels != 1:
            errors.append(f"{row.file}: {info.channels} channels, want mono")
        if info.subtype != "PCM_16":
            errors.append(f"{row.file}: {info.subtype}, want PCM_16")
        if not _MIN_S <= info.duration <= _MAX_S:
            errors.append(f"{row.file}: {info.duration:.2f}s outside {_MIN_S}-{_MAX_S}s")
        if not normalize(row.transcript):
            errors.append(f"{row.file}: empty transcript after normalisation")

    if errors:
        raise ManifestError("\n".join(errors))
    return rows


def _print_counts(rows: list[ManifestRow]) -> None:
    total_s = sum(r.duration_s for r in rows)
    print(f"{len(rows)} clips, {total_s / 60:.1f} min of audio")
    for field in ("lang", "condition", "speaker", "source"):
        counts = Counter(getattr(r, field) for r in rows)
        print(f"  {field:<10} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    try:
        _print_counts(validate())
    except ManifestError as exc:
        print(f"manifest invalid:\n{exc}", file=sys.stderr)
        sys.exit(1)
