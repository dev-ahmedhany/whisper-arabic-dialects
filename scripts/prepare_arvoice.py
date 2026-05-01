"""Prepare MBZUAI/ArVoice → JSONL, MSA, **human portion only**.

ArVoice is a multi-speaker MSA dataset that mixes real human recordings with
Google Wavenet TTS synth. The HF dataset card says ~10 h human / ~73 h synth.
The synth rows have speaker_id values like `ar-XA-Standard-A`, `ar-XA-Wavenet-B`,
`ar-XA-Chirp3-HD-...`, `ar-XA-Neural2-...` — i.e. anything matching `ar-XA-*`.
The human rows have plain identifiers (no `ar-XA-*` prefix).

We filter `speaker_id` to drop synth and keep only humans, so the resulting
JSONL is fully human-labeled per memory/maghrebi_excluded.md and the v4 mix
"no pseudo, no synth" rule.

Schema: keys = normalized_wav, original_wav, speaker_id, transcription. We
use original_wav (raw human voice) at 24kHz native — _hf_audio_to_jsonl will
resample to 16 kHz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl

SYNTH_PREFIXES = ("ar-XA-",)


def _is_human(speaker_id: str) -> bool:
    if not speaker_id:
        return False
    return not any(speaker_id.startswith(p) for p in SYNTH_PREFIXES)


def _human_text(row: dict) -> str:
    if not _is_human(row.get("speaker_id", "")):
        return ""
    return row.get("transcription", "") or ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="MBZUAI/ArVoice")
    p.add_argument("--split", default="train")
    p.add_argument("--out", type=Path, default=Path("test_sets/arvoice_msa_human_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/arvoice_human"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id=args.dataset_id,
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=_human_text,
        dialect="msa",
        audio_column="original_wav",
        max_samples=args.max_samples,
    )
    print(f"wrote {n} human rows to {args.out}")


if __name__ == "__main__":
    main()
