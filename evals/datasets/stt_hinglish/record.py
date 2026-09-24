"""Walk prompts.txt and record one 16 kHz mono clip per prompt.

Usage:
    uv run python -m evals.datasets.stt_hinglish.record --speaker p1
    uv run python -m evals.datasets.stt_hinglish.record --speaker p2 --source friend --start-at 40

Each prompt line is `lang|condition|reference transcript`. The transcript is the
reference as-is — if your reading slipped, fix that row in manifest.jsonl by hand.
Controls: Enter starts, Enter stops, then Enter keeps / r redoes / s skips / q quits.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import sounddevice as sd
import soundfile as sf

from evals.datasets.stt_hinglish.models import (
    DATASET_DIR,
    SAMPLE_RATE,
    ManifestRow,
    append_row,
    next_clip_path,
)

_PROMPTS = DATASET_DIR / "prompts.txt"


def _load_prompts(path: pathlib.Path) -> list[tuple[str, str, str]]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lang, condition, text = (part.strip() for part in line.split("|", 2))
        prompts.append((lang, condition, text))
    return prompts


def _record_until_enter() -> np.ndarray:
    chunks: list[np.ndarray] = []

    def _callback(indata, _frames, _time, status):
        if status:
            print(f"  (audio status: {status})")
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_callback):
        input("  ● recording — Enter to stop ")
    return np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype="int16")


def main(speaker: str, source: str, start_at: int, prompts_path: pathlib.Path) -> None:
    prompts = _load_prompts(prompts_path)
    mic = sd.query_devices(kind="input")["name"]
    print(f"{len(prompts)} prompts, starting at #{start_at}. Mic: {mic}")

    i = start_at - 1
    while i < len(prompts):
        lang, condition, text = prompts[i]
        print(f"\n[{i + 1}/{len(prompts)}] ({lang}, {condition})\n  {text}")
        if condition == "fast":
            print("  → say it fast")
        elif condition == "noisy":
            print("  → background noise on, phone/mic at arm's length")
        input("  Enter to start ")
        audio = _record_until_enter()
        duration = len(audio) / SAMPLE_RATE

        choice = input(f"  {duration:.1f}s — Enter keep / r redo / s skip / q quit: ")
        choice = choice.strip().lower()
        if choice == "q":
            break
        if choice == "r":
            continue
        if choice == "s":
            i += 1
            continue

        out = next_clip_path()
        sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")
        append_row(
            ManifestRow(
                file=out.name,
                transcript=text,
                lang=lang,
                condition=condition,
                speaker=speaker,
                source=source,
                duration_s=round(duration, 2),
            )
        )
        print(f"  saved {out.name}")
        i += 1

    print(f"\nStopped at prompt #{i + 1}. Resume with --start-at {i + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True, help="short id, e.g. p1")
    parser.add_argument("--source", default="self", choices=["self", "friend"])
    parser.add_argument("--start-at", type=int, default=1, help="1-based prompt number")
    parser.add_argument("--prompts", type=pathlib.Path, default=_PROMPTS)
    args = parser.parse_args()
    main(args.speaker, args.source, args.start_at, args.prompts)
