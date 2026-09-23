"""Sentence schema + helpers shared by validate.py and the TTS runner."""

from __future__ import annotations

import pathlib
from typing import Literal

from pydantic import BaseModel

from evals.metrics.wer import romanize

DATASET_DIR = pathlib.Path(__file__).parent
SENTENCES = DATASET_DIR / "sentences.jsonl"

Lang = Literal["hi-en", "hi", "en-IN"]
Mix = Literal["heavy", "light", "pure_hi", "pure_en", "numeric"]
Length = Literal["short", "medium", "long"]
Source = Literal["llm-edited", "hand"]
Script = Literal["native", "roman"]

# Dataset lang bucket → BCP-47 code passed to TTSProvider.synthesize()
BCP47: dict[str, str] = {"hi-en": "hi-IN", "hi": "hi-IN", "en-IN": "en-IN"}

# Word-count bounds per length bucket (inclusive)
LENGTH_WORDS: dict[str, tuple[int, int]] = {"short": (1, 8), "medium": (9, 20), "long": (21, 35)}


class SentenceRow(BaseModel):
    id: str
    text: str
    text_roman: str | None = None
    lang: Lang
    mix: Mix
    length: Length
    entities: list[str] = []
    source: Source

    def text_for(self, script: Script) -> str | None:
        return self.text if script == "native" else self.text_roman


def load_sentences(path: pathlib.Path = SENTENCES) -> list[SentenceRow]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [SentenceRow.model_validate_json(line) for line in lines if line.strip()]


def _squash(text: str) -> str:
    # Spaces dropped so "follow-up" and "follow up" match
    return romanize(text).replace(" ", "")


def entity_hits(entities: list[str], hypothesis: str) -> int:
    """Count entities whose romanized form appears in the romanized hypothesis (spec D7)."""
    hyp = _squash(hypothesis)
    return sum(1 for e in entities if _squash(e) and _squash(e) in hyp)
