"""Prepare MGB-3 (Egyptian) → JSONL via HuggingFace Hub.

Uses the public mirror at `ArabicSpeech/MGB-3` (no registration needed). The dataset
is entirely Egyptian Arabic by design, so dialect is hardcoded to "egyptian".

Schema on HF Hub: { id, audio, text }.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="ArabicSpeech/MGB-3")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out", type=Path, default=Path("test_sets/mgb3_egyptian_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/mgb3"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id=args.dataset_id,
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=lambda r: r.get("text") or r.get("transcription") or r.get("sentence", ""),
        dialect="egyptian",
        max_samples=args.max_samples,
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
