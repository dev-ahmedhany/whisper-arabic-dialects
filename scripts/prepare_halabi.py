"""Prepare halabi2016/arabic_speech_corpus → JSONL, Levantine (Damascene).

This is a 3 h studio-quality Levantine corpus from a PhD project at U.
Southampton. All transcripts checked by humans. Audio at 48 kHz native.

CAVEAT: the dataset stores text in **Buckwalter transliteration** (e.g.
`waraj~aHa Alt~aqoriyru`), not Arabic script. We back-convert to Arabic
script using the deterministic Buckwalter→Unicode mapping before writing.

Without this conversion the WER would be meaningless (Whisper outputs
Arabic script; comparing to Buckwalter is apples-to-oranges).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl

# Buckwalter (canonical) -> Unicode Arabic. Source: Tim Buckwalter's
# transliteration table. Covers the characters used by halabi2016.
BW_TO_AR = {
    "'": "ء", "|": "آ", ">": "أ", "&": "ؤ", "<": "إ", "}": "ئ",
    "A": "ا", "b": "ب", "p": "ة", "t": "ت", "v": "ث", "j": "ج",
    "H": "ح", "x": "خ", "d": "د", "*": "ذ", "r": "ر", "z": "ز",
    "s": "س", "$": "ش", "S": "ص", "D": "ض", "T": "ط", "Z": "ظ",
    "E": "ع", "g": "غ", "_": "ـ", "f": "ف", "q": "ق", "k": "ك",
    "l": "ل", "m": "م", "n": "ن", "h": "ه", "w": "و", "Y": "ى",
    "y": "ي",
    "F": "ً", "N": "ٌ", "K": "ٍ",
    "a": "َ", "u": "ُ", "i": "ِ",
    "~": "ّ", "o": "ْ",
    "`": "ٰ", "{": "ٱ",
}


def buckwalter_to_arabic(bw: str) -> str:
    return "".join(BW_TO_AR.get(c, c) for c in bw)


def _text(row: dict) -> str:
    bw = row.get("text") or row.get("orthographic") or ""
    return buckwalter_to_arabic(bw).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="halabi2016/arabic_speech_corpus")
    p.add_argument("--split", default="train")
    p.add_argument("--out", type=Path, default=Path("test_sets/halabi_levantine_train.jsonl"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/halabi"))
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    n = stream_to_jsonl(
        dataset_id=args.dataset_id,
        split=args.split,
        output_jsonl=args.out,
        audio_dir=args.audio_dir,
        text_fn=_text,
        dialect="levantine",
        max_samples=args.max_samples,
        trust_remote_code=True,
    )
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
