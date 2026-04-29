"""Prepare MASC (Massive Arabic Speech Corpus) → JSONL, tagged Levantine.

MASC has Levantine and other dialects mixed; we filter on the dialect column where
available and otherwise pass through as Levantine (override via --dialect).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="pain/MASC")
    p.add_argument("--split", default="train")
    p.add_argument("--dialect", default="levantine")
    p.add_argument("--out", type=Path, default=Path("test_sets/masc_levantine_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/masc"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id=args.dataset_id,
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=lambda r: r.get("transcription") or r.get("text") or r.get("sentence", ""),
        dialect=args.dialect,
        max_samples=args.max_samples,
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
