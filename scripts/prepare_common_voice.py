"""Prepare Common Voice (Arabic) → JSONL.

The official `mozilla-foundation/common_voice_*` repos on HF Hub were emptied
in October 2025 — Mozilla now distributes Common Voice exclusively through the
Mozilla Data Collective. This script pulls from a community mirror that still
contains the audio data: `MohamedRashad/common-voice-18-arabic` (CV18 ar,
ungated, ~28k train rows, 48 kHz).

Treated as MSA in the dialect mix (CV is largely MSA with some user-recorded
dialect drift).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="MohamedRashad/common-voice-18-arabic",
                   help="HF dataset id; default is the community CV18 ar mirror")
    p.add_argument("--split", default="train", choices=["train", "validation", "test", "other"])
    p.add_argument("--out", type=Path, default=Path("test_sets/common_voice_18_ar_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/common_voice_18_ar"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id=args.dataset_id,
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
