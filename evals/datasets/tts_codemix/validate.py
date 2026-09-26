"""Check sentences.jsonl before any TTS eval spends characters.

Usage:
    uv run python -m evals.datasets.tts_codemix.validate
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter

from evals.datasets.tts_codemix.models import (
    LENGTH_WORDS,
    SENTENCES,
    SentenceRow,
    _squash,
    load_sentences,
)


class SentencesError(Exception):
    pass


def validate(path: pathlib.Path = SENTENCES) -> list[SentenceRow]:
    if not path.exists():
        raise SentencesError(f"{path} not found")
    try:
        rows = load_sentences(path)
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        raise SentencesError(str(exc)) from exc
    if not rows:
        raise SentencesError(f"{path} is empty")

    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.id in seen:
            errors.append(f"{row.id}: duplicate id")
        seen.add(row.id)
        if not row.text.strip():
            errors.append(f"{row.id}: empty text")

        lo, hi = LENGTH_WORDS[row.length]
        n = len(row.text.split())
        if not lo <= n <= hi:
            errors.append(f"{row.id}: {n} words, but length={row.length} wants {lo}-{hi}")

        for ent in row.entities:
            if _squash(ent) not in _squash(row.text):
                errors.append(f"{row.id}: entity {ent!r} not in text")
            if row.text_roman and _squash(ent) not in _squash(row.text_roman):
                errors.append(f"{row.id}: entity {ent!r} not in text_roman")

    if errors:
        raise SentencesError("\n".join(errors))
    return rows


def main() -> None:
    try:
        rows = validate()
    except SentencesError as exc:
        sys.exit(f"INVALID:\n{exc}")
    print(f"OK — {len(rows)} sentences, {sum(r.text_roman is not None for r in rows)} with roman")
    for field in ("mix", "length", "lang", "source"):
        print(f"  {field}: {dict(Counter(getattr(r, field) for r in rows))}")


if __name__ == "__main__":
    main()
