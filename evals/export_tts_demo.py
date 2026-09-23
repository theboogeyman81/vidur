"""Copy a curated set of TTS clips into the committed demo folder for the A/B player (spec D13).

Usage:
    uv run python -m evals.export_tts_demo evals/results/tts_{ts}.json s003,s017,s048

Copies every engine's clip for the listed ids (both scripts, if present) into
evals/results/audio_demo/{engine}/ and rewrites audio_demo/index.json.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

from pydantic import BaseModel

from evals.run_tts_eval import TTSRun

_RESULTS_DIR = pathlib.Path(__file__).parent / "results"
_DEMO_DIR = _RESULTS_DIR / "audio_demo"


class DemoClip(BaseModel):
    id: str
    script: str
    engine: str
    text: str
    path: str  # relative to audio_demo/
    ttfb_ms: float | None
    roundtrip_wer: float | None


class DemoIndex(BaseModel):
    source_run: str
    clips: list[DemoClip]


def export(results_path: pathlib.Path, ids: list[str]) -> DemoIndex:
    run = TTSRun.model_validate_json(results_path.read_text(encoding="utf-8"))
    clips: list[DemoClip] = []
    for engine in run.engines:
        for r in engine.per_sentence:
            if r.id not in ids or r.audio_path is None:
                continue
            src = _RESULTS_DIR / r.audio_path
            dest = _DEMO_DIR / engine.engine / f"{r.id}_{r.script}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            clips.append(
                DemoClip(
                    id=r.id,
                    script=r.script,
                    engine=engine.engine,
                    text=r.text,
                    path=str(dest.relative_to(_DEMO_DIR)),
                    ttfb_ms=r.ttfb_ms,
                    roundtrip_wer=r.roundtrip_wer,
                )
            )
    missing = set(ids) - {c.id for c in clips}
    if missing:
        sys.exit(f"ids not found in {results_path.name}: {sorted(missing)}")
    index = DemoIndex(source_run=results_path.name, clips=clips)
    (_DEMO_DIR / "index.json").write_text(index.model_dump_json(indent=2), encoding="utf-8")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("ids", help="comma-separated sentence ids")
    args = parser.parse_args()
    index = export(args.results, args.ids.split(","))
    print(f"{len(index.clips)} clips → {_DEMO_DIR}")
