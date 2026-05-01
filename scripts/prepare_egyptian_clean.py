"""Prepare cleaned Egyptian Arabic broadcast → JSONL.

`MohamedGomaa30/Egyptian-Speech-Clean-MGB3` is a community re-release of
MGB-3 with background music/noise separated from the speech track. Dirty
MGB-3 hurt v1 because the noise floor leaked into the encoder; the cleaned
variant is the same content but with a cleaner acoustic signal.

We use the `original_audio` track (not `separated_target_audio`) because the
separated track sometimes drops voiced segments at the edges; the original
track keeps all speech and the noise reduction it gives us is incidental.
For v3 use the cleaned `separated_target_audio` if val WER moves the right
direction (set `--audio-key separated_target_audio`).

Tagged Egyptian.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="MohamedGomaa30/Egyptian-Speech-Clean-MGB3")
    p.add_argument("--split", default="train")
    p.add_argument("--dialect", default="egyptian")
    p.add_argument("--audio-key", default="original_audio",
                   choices=["original_audio", "separated_target_audio"])
    p.add_argument("--out", type=Path, default=Path("test_sets/egyptian_clean_mgb3_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/egyptian_clean_mgb3"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id=args.dataset_id,
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=lambda r: r.get("text", ""),
        dialect=args.dialect,
        max_samples=args.max_samples,
        audio_column=args.audio_key,
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
