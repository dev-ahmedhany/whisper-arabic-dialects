"""Shared helper: stream a HuggingFace audio dataset → JSONL of {audio, reference, dialect, duration_s}.

Each prepare_*.py script wraps this with the dataset-specific column names and dialect tags.
Audio is materialized to local 16kHz WAVs (faster_whisper expects file paths).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


def stream_to_jsonl(
    dataset_id: str,
    split: str,
    output_jsonl: Path,
    audio_dir: Path,
    text_fn: Callable[[dict], str],
    dialect: str,
    name: str | None = None,
    max_samples: int | None = None,
    streaming: bool = True,
    audio_column: str = "audio",
    trust_remote_code: bool = False,
    extra_load_kwargs: dict | None = None,
) -> int:
    """Returns count of rows written. Skips rows where text_fn returns empty string.

    `trust_remote_code=True` is required for datasets that ship a custom loading
    script (e.g. pain/MASC). Only enable for datasets you've inspected and trust.
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        dataset_id,
        name=name,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
        **(extra_load_kwargs or {}),
    )
    if not streaming:
        ds = ds.cast_column(audio_column, Audio(sampling_rate=16000))
    n_written = 0
    with output_jsonl.open("w") as out:
        it: Iterable[dict] = ds
        for i, row in enumerate(tqdm(it, desc=f"{dataset_id}/{split}")):
            if max_samples is not None and n_written >= max_samples:
                break
            text = text_fn(row)
            if not text or not text.strip():
                continue
            audio = row[audio_column]
            if streaming:
                array = audio["array"]
                sr = audio["sampling_rate"]
            else:
                array = audio["array"]
                sr = audio["sampling_rate"]
            wav_path = audio_dir / f"{dataset_id.replace('/', '__')}_{split}_{i:07d}.wav"
            if sr != 16000:
                import librosa

                array = librosa.resample(array, orig_sr=sr, target_sr=16000)
                sr = 16000
            sf.write(wav_path, array, sr, subtype="PCM_16")
            duration_s = float(len(array)) / float(sr)
            out.write(
                json.dumps(
                    {
                        "audio": str(wav_path.resolve()),
                        "reference": text.strip(),
                        "dialect": dialect,
                        "source_dataset": dataset_id,
                        "duration_s": duration_s,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1
    return n_written
