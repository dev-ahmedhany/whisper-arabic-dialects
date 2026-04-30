"""Prepare ArzEn (Egyptian Arabic conversational speech) → JSONL.

ArzEn-ST is a 12-hour Egyptian-Arabic ↔ English code-switching speech corpus
collected at the German University in Cairo. We pull the Arabic-only side
(spkr language=ar transcripts) for v3's Egyptian conversational pool — adds a
genre slice that Casablanca alone doesn't cover (informal interview vs. media).

HF dataset id: `LanaJG/ArzEn-ST` (ungated, audio + transcripts). The text field
is `transcription_ar` (raw Arabic transcript without the English code-switches);
rows where speakers code-switch are partially Arabic — we drop rows with <3
Arabic words to stay closer to monolingual training signal.

Tagged Egyptian. We do not over-filter for code-switching since Whisper
handles brief English insertions, but rows that are majority English are
filtered out by the empty-Arabic check.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl

ARABIC_RE = re.compile(r"[؀-ۿ]+")


def _arabic_text(row: dict) -> str:
    txt = (
        row.get("transcription_ar")
        or row.get("transcript_ar")
        or row.get("text_ar")
        or row.get("transcription")
        or row.get("text")
        or ""
    )
    if not txt:
        return ""
    arabic_tokens = ARABIC_RE.findall(txt)
    if len(arabic_tokens) < 3:
        return ""
    return txt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="LanaJG/ArzEn-ST")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--dialect", default="egyptian")
    p.add_argument("--out", type=Path, default=Path("test_sets/arzen_egyptian_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/arzen"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id=args.dataset_id,
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=_arabic_text,
        dialect=args.dialect,
        max_samples=args.max_samples,
        trust_remote_code=True,
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
