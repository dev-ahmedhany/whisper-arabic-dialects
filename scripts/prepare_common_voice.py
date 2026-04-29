"""Prepare Common Voice 17.0 (ar) splits → JSONL.

Common Voice is largely MSA, with some dialect drift in user-recorded clips.
We treat it as MSA for the dialect mix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out", type=Path, default=Path("test_sets/common_voice_17_ar_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/common_voice_17_ar"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id="mozilla-foundation/common_voice_17_0",
        name="ar",
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=lambda r: r.get("sentence", ""),
        dialect="msa",
        max_samples=args.max_samples,
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
