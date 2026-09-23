import pytest

from evals.metrics.wer import (
    compute_cer,
    compute_wer,
    corpus_wer,
    normalize,
    romanize,
)


def test_identical_is_zero():
    s = "आज हम polity पढ़ रहे हैं"
    assert compute_wer(s, s) == 0.0
    assert compute_cer(s, s) == 0.0


def test_nukta_variants_ignored():
    # precomposed U+095D (ढ़) vs base + combining nukta
    assert compute_wer("\u092a\u095d\u0928\u093e", "\u092a\u0922\u093c\u0928\u093e") == 0.0
    assert compute_wer("\u095d\u0928\u093e", "\u0922\u0928\u093e") == 0.0


def test_chandrabindu_folds_to_anusvara():
    assert compute_wer("हाँ", "हां") == 0.0


def test_danda_and_punctuation_ignored():
    assert compute_wer("मौलिक अधिकार क्या हैं।", "मौलिक अधिकार, क्या हैं?") == 0.0


def test_latin_case_ignored():
    assert compute_wer("Article 370 Lok Sabha", "article 370 lok sabha") == 0.0


def test_zero_width_joiners_stripped():
    assert normalize("क्‍ष") == normalize("क्ष")


def test_one_substitution_in_four():
    assert compute_wer("aaj hum history padh", "aaj hum polity padh") == pytest.approx(0.25)


def test_empty_hypothesis_is_one():
    assert compute_wer("", "aaj hum polity padh") == 1.0


def test_empty_reference_raises():
    with pytest.raises(ValueError):
        compute_wer("anything", "।")


def test_corpus_wer_is_not_mean_of_clips():
    refs = ["a b", "c d e f g h i j"]
    hyps = ["x b", "c d e f g h i j"]
    per_clip_mean = (compute_wer(hyps[0], refs[0]) + compute_wer(hyps[1], refs[1])) / 2
    assert per_clip_mean == pytest.approx(0.25)
    assert corpus_wer(hyps, refs) == pytest.approx(0.1)


def test_romanize_projects_devanagari_and_keeps_latin():
    out = romanize("आज polity")
    assert out.isascii()
    assert out.endswith("polity")
