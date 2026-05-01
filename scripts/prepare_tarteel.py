"""Prepare Tarteel everyayah (Quran recitation) → JSONL, MSA register.

Tarteel-AI hosts everyayah on HF: ~150 h of human Quran recitation by professional
reciters with verbatim Arabic-script transcripts (Quranic verses). Use cap to keep
the narrow Quran register from dominating the broader MSA pool.

Schema: {audio, duration, reciter, text}; sr=16000 native (no resample needed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="tarteel-ai/everyayah")
    p.add_argument("--split", default="train")
    p.add_argument("--out", type=Path, default=Path("test_sets/tarteel_msa_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/tarteel"))
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap row count; combine with build_v4_mix --max-msa to bound hours")
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id=args.dataset_id,
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=lambda r: r.get("text", ""),
        dialect="msa",
        max_samples=args.max_samples,
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
