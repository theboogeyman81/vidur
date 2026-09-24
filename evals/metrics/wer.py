"""WER / CER for mixed-script Hinglish (Devanagari + Latin in one string).

Primary metric is scored on native script (spec D1). `romanize()` backs the
secondary `cer_roman` diagnostic (D2). Corpus-level scoring (D3) runs jiwer
over the whole list at once rather than averaging per-clip scores.
"""

from __future__ import annotations

import re
import unicodedata

import jiwer
from indic_transliteration import sanscript

_NUKTA = "़"
_CHANDRABINDU = "ँ"
_ANUSVARA = "ं"
_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def normalize(text: str) -> str:
    """Script-preserving normaliser applied identically to reference and hypothesis."""
    # NFC leaves क़/ज़/फ़ decomposed (they're composition exclusions), so the nukta is
    # always a separate code point after this and can just be dropped.
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace(_NUKTA, "").replace(_CHANDRABINDU, _ANUSVARA)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf":  # ZWJ / ZWNJ
            continue
        # punctuation (incl. danda ।, ॥) and symbols become word breaks
        out.append(" " if cat[0] in "PS" else ch)
    return _WS.sub(" ", "".join(out)).strip()


def _prep(hyps: list[str], refs: list[str]) -> tuple[list[str], list[str]]:
    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses vs {len(refs)} references")
    norm_refs = [normalize(r) for r in refs]
    if any(not r for r in norm_refs):
        raise ValueError("empty reference after normalisation — manifest bug")
    return [normalize(h) for h in hyps], norm_refs


def corpus_wer(hyps: list[str], refs: list[str]) -> float:
    h, r = _prep(hyps, refs)
    return float(jiwer.wer(reference=r, hypothesis=h))


def corpus_cer(hyps: list[str], refs: list[str]) -> float:
    h, r = _prep(hyps, refs)
    return float(jiwer.cer(reference=r, hypothesis=h))


def compute_wer(hypothesis: str, reference: str) -> float:
    return corpus_wer([hypothesis], [reference])


def compute_cer(hypothesis: str, reference: str) -> float:
    return corpus_cer([hypothesis], [reference])


def romanize(text: str) -> str:
    """Lossy Devanagari→Latin projection. Diagnostic only — see spec D2."""
    roman = sanscript.transliterate(normalize(text), sanscript.DEVANAGARI, sanscript.ITRANS)
    return _WS.sub(" ", _NON_ALNUM.sub("", roman.lower())).strip()
