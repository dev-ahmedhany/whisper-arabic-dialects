"""Prepare FLEURS Arabic test set → JSONL. Held-out evaluation only — never in train."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--out", type=Path, default=Path("test_sets/fleurs_msa_test.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/fleurs_ar"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id="google/fleurs",
        name="ar_eg",
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=lambda r: r.get("transcription", ""),
        dialect="msa",
        max_samples=args.max_samples,
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
