"""Import N Common Voice Hindi clips (CC0) into the `hi` bucket, resampled to 16 kHz.

Usage:
    uv run python -m evals.datasets.stt_hinglish.import_cv \
        --tsv ~/Downloads/cv-hi/validated.tsv --n 10

Expects the standard Common Voice layout: `clips/` next to the TSV. Picks one clip per
speaker first for accent spread. Decoding uses faster-whisper's PyAV decoder, so no ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random

import numpy as np
import soundfile as sf
from faster_whisper.audio import decode_audio

from evals.datasets.stt_hinglish.models import (
    SAMPLE_RATE,
    ManifestRow,
    append_row,
    load_manifest,
    next_clip_path,
)

_MAX_S = 12.0


def main(tsv: pathlib.Path, n: int, seed: int) -> None:
    with tsv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    clips_dir = tsv.parent / "clips"

    already = {r.speaker for r in load_manifest() if r.source == "cv"}
    random.Random(seed).shuffle(rows)
    # one clip per unseen speaker first, then fill with the rest
    by_speaker: dict[str, dict] = {}
    for row in rows:
        by_speaker.setdefault(row["client_id"], row)
    ordered = list(by_speaker.values()) + [r for r in rows if r not in by_speaker.values()]

    imported = 0
    for row in ordered:
        if imported >= n:
            break
        speaker = f"cv_{row['client_id'][:6]}"
        if speaker in already:
            continue
        audio = decode_audio(str(clips_dir / row["path"]), sampling_rate=SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        if duration > _MAX_S:
            continue

        out = next_clip_path()
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        sf.write(out, pcm, SAMPLE_RATE, subtype="PCM_16")
        append_row(
            ManifestRow(
                file=out.name,
                transcript=row["sentence"],
                lang="hi",
                condition="clean",
                speaker=speaker,
                source="cv",
                duration_s=round(duration, 2),
            )
        )
        already.add(speaker)
        imported += 1
        print(f"  {out.name} ← {row['path']} ({duration:.1f}s)")

    print(f"imported {imported}/{n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=pathlib.Path, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.tsv.expanduser(), args.n, args.seed)
