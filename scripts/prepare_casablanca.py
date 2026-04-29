"""Prepare Casablanca multi-dialect benchmark → per-dialect JSONL files.

Held-out evaluation only.
Dialect column varies; adjust mapping if the upstream schema changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

import soundfile as sf

DIALECT_MAP = {
    "msa": "msa",
    "egyptian": "egyptian",
    "egy": "egyptian",
    "levantine": "levantine",
    "lev": "levantine",
    "gulf": "gulf",
    "glf": "gulf",
    "maghrebi": "maghrebi",
    "mor": "maghrebi",
    "morocco": "maghrebi",
    "moroccan": "maghrebi",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="MBZUAI-Paris/Casablanca",
                   help="HF dataset id (override if the public mirror differs)")
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", type=Path, default=Path("test_sets"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/casablanca"))
    p.add_argument("--max-per-dialect", type=int, default=500)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_id, split=args.split, streaming=True)
    files: dict[str, any] = {}
    counts: dict[str, int] = {}

    for i, row in enumerate(tqdm(ds, desc="casablanca")):
        raw_dialect = str(row.get("dialect") or row.get("language") or "").lower().strip()
        dialect = DIALECT_MAP.get(raw_dialect)
        if dialect is None:
            continue
        if counts.get(dialect, 0) >= args.max_per_dialect:
            continue

        text = (row.get("transcription") or row.get("text") or row.get("sentence") or "").strip()
        if not text:
            continue

        audio = row["audio"]
        array = audio["array"]
        sr = audio["sampling_rate"]
        if sr != 16000:
            import librosa

            array = librosa.resample(array, orig_sr=sr, target_sr=16000)
            sr = 16000
        wav_path = args.audio_dir / f"casablanca_{dialect}_{i:07d}.wav"
        sf.write(wav_path, array, sr, subtype="PCM_16")
        duration_s = float(len(array)) / float(sr)

        out_path = args.out_dir / f"casablanca_{dialect}_test.jsonl"
        if out_path not in files:
            files[out_path] = out_path.open("w")
        files[out_path].write(
            json.dumps(
                {
                    "audio": str(wav_path.resolve()),
                    "reference": text,
                    "dialect": dialect,
                    "source_dataset": "casablanca",
                    "duration_s": duration_s,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        counts[dialect] = counts.get(dialect, 0) + 1

    for f in files.values():
        f.close()
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
