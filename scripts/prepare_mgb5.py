"""Prepare MGB-5 (Moroccan) → JSONL. Same Kaldi-style ingestion as MGB-3.

Gated; register at https://arabicspeech.org/mgb5/ and pass --audio-dir and --text-file.
This is a thin wrapper over prepare_mgb3 with Moroccan defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.prepare_mgb3 import parse_kaldi_text
import json
import soundfile as sf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--text-file", type=Path, required=True)
    p.add_argument("--dialect", default="maghrebi")
    p.add_argument("--out", type=Path, default=Path("test_sets/mgb5_moroccan_train.jsonl"))
    p.add_argument("--source-tag", default="mgb5")
    args = p.parse_args()

    transcripts = parse_kaldi_text(args.text_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as out:
        for utt_id, ref in transcripts.items():
            wav = args.audio_dir / f"{utt_id}.wav"
            if not wav.exists():
                continue
            info = sf.info(str(wav))
            duration_s = float(info.frames) / float(info.samplerate)
            out.write(
                json.dumps(
                    {
                        "audio": str(wav.resolve()),
                        "reference": ref.strip(),
                        "dialect": args.dialect,
                        "source_dataset": args.source_tag,
                        "duration_s": duration_s,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
