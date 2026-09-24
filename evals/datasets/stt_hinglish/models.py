"""Manifest schema + helpers shared by record.py, import_cv.py, validate.py and the runner."""

from __future__ import annotations

import pathlib
import re
from typing import Literal

from pydantic import BaseModel

DATASET_DIR = pathlib.Path(__file__).parent
MANIFEST = DATASET_DIR / "manifest.jsonl"
SAMPLE_RATE = 16000

Lang = Literal["hi-en", "hi", "en-IN"]
Condition = Literal["clean", "fast", "noisy", "spontaneous"]
Source = Literal["self", "friend", "cv", "synth"]

_CLIP_RE = re.compile(r"clip_(\d+)\.wav$")


class ManifestRow(BaseModel):
    file: str
    transcript: str
    lang: Lang
    condition: Condition
    speaker: str
    source: Source
    duration_s: float


def load_manifest(path: pathlib.Path = MANIFEST) -> list[ManifestRow]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ManifestRow.model_validate_json(line) for line in lines if line.strip()]


def append_row(row: ManifestRow, path: pathlib.Path = MANIFEST) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(row.model_dump_json() + "\n")


def next_clip_path() -> pathlib.Path:
    matches = (_CLIP_RE.search(p.name) for p in DATASET_DIR.glob("clip_*.wav"))
    nums = [int(m.group(1)) for m in matches if m]
    return DATASET_DIR / f"clip_{max(nums, default=0) + 1:03d}.wav"
