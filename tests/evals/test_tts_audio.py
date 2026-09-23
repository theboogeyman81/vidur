from evals.datasets.tts_codemix.models import BCP47, entity_hits
from evals.datasets.tts_codemix.validate import validate
from evals.metrics.audio import pcm16_to_wav, wav_info, wav_to_pcm16


def test_wav_roundtrip_preserves_pcm_and_rate():
    pcm = bytes(range(256)) * 10  # 2560 bytes = 1280 samples
    wav = pcm16_to_wav(pcm, 24000)
    out, sr, ch = wav_to_pcm16(wav)
    assert (out, sr, ch) == (pcm, 24000, 1)


def test_wav_info_duration():
    wav = pcm16_to_wav(b"\x00\x00" * 22050, 22050)
    assert wav_info(wav) == (22050, 1.0)


def test_lang_mapping():
    assert BCP47 == {"hi-en": "hi-IN", "hi": "hi-IN", "en-IN": "en-IN"}


def test_entity_hit_ignores_case_and_hyphen():
    assert (
        entity_hits(["follow-up question", "Preamble"], "ek follow up Question aur preamble") == 2
    )


def test_entity_miss_when_spoken_as_devanagari_garble():
    assert entity_hits(["Preamble"], "पियाम बलका में नादी") == 0


def test_dataset_is_valid():
    rows = validate()
    assert len(rows) >= 50
    assert sum(r.text_roman is not None for r in rows) == 15
