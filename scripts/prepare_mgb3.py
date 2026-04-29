"""Prepare MGB-3 (Egyptian) → JSONL.

MGB-3 is gated. You must register at the MGB challenge website
(https://arabicspeech.org/mgb3/) and download:
  - audio/wav/{split}/<utt>.wav
  - text/{split}/text         (Kaldi-style: "<utt-id> <transcription>")

This script consumes those local files and emits the standard JSONL schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf


def parse_kaldi_text(text_file: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        utt_id, ref = parts
        out[utt_id] = ref
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, required=True,
                   help="dir containing <utt-id>.wav files")
    p.add_argument("--text-file", type=Path, required=True,
                   help="Kaldi-style text file: '<utt-id> <transcription>' per line")
    p.add_argument("--dialect", default="egyptian")
    p.add_argument("--out", type=Path, default=Path("test_sets/mgb3_egyptian_train.jsonl"))
    p.add_argument("--source-tag", default="mgb3")
    args = p.parse_args()

    transcripts = parse_kaldi_text(args.text_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
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
            n_written += 1
    print(f"wrote {n_written} rows to {args.out}")


if __name__ == "__main__":
    main()
