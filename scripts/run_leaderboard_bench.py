"""Open Universal Arabic ASR Leaderboard submission script.

Mirrors the leaderboard's `models/whisper.py` pattern (HF transformers
pipeline with batch_size + chunk_length_s), so the output manifest is
directly compatible with their `eval.py:calculate_wer`.

Per-dataset HF stream loaders for the 4 ungated leaderboard test sets:
  - cv18         : mozilla-foundation/common_voice_18_0  ar  test
  - masc-clean   : pain/MASC                              clean_test
  - masc-noisy   : pain/MASC                              noisy_test
  - casablanca   : UBC-NLP/Casablanca                     test (multi-country, all dialects)

Output manifest format (one JSON line per utterance):
  {"audio_filepath": "<source_id>", "text": "<reference>", "pred_text": "<hypothesis>"}

Usage on L4:
    python scripts/run_leaderboard_bench.py \\
        --dataset cv18 \\
        --model dev-ahmedhany/whisper-large-v3-arabic-ft-v3 \\
        --output runs/leaderboard/ft-v3-cv18.jsonl \\
        --batch-size 8

Then run their evaluator (clone https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard):
    python eval.py runs/leaderboard/ft-v3-cv18.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset, Audio
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


DATASETS = {
    "cv18": {
        "id": "mozilla-foundation/common_voice_18_0",
        "name": "ar",
        "split": "test",
        "text_field": "sentence",
        "id_field": "path",
        "trust_remote_code": True,
    },
    "masc-clean": {
        "id": "pain/MASC",
        "split": "clean_test",
        "text_field": "transcript",
        "id_field": "audio_id",
        "trust_remote_code": True,
    },
    "masc-noisy": {
        "id": "pain/MASC",
        "split": "noisy_test",
        "text_field": "transcript",
        "id_field": "audio_id",
        "trust_remote_code": True,
    },
    "casablanca": {
        "id": "UBC-NLP/Casablanca",
        # All countries combined; submit per-country if leaderboard requires it
        "name": None,
        "split": "test",
        "text_field": "transcription",
        "id_field": None,
        "trust_remote_code": True,
    },
}


def stream_dataset(spec: dict):
    """Stream a HF dataset row by row."""
    kw = dict(split=spec["split"], streaming=True,
              trust_remote_code=spec.get("trust_remote_code", False))
    if spec.get("name"):
        kw["name"] = spec["name"]
    ds = load_dataset(spec["id"], **kw)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    text_field = spec["text_field"]
    id_field = spec.get("id_field")
    for i, row in enumerate(ds):
        ref = (row.get(text_field) or
               row.get("text") or
               row.get("sentence") or "")
        rid = (row.get(id_field) if id_field else None) or f"row_{i}"
        yield rid, ref, row["audio"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--model", default="dev-ahmedhany/whisper-large-v3-arabic-ft-v3")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-length-s", type=int, default=30)
    p.add_argument("--max-rows", type=int, default=0,
                   help="cap rows for testing; 0 means full dataset")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[args.dataset]

    print(f"loading {args.model} on {args.device} ...", flush=True)
    dtype = torch.float16 if "cuda" in args.device else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(args.device)
    proc = AutoProcessor.from_pretrained(args.model)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model, tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
        torch_dtype=dtype, device=args.device,
        chunk_length_s=args.chunk_length_s, batch_size=args.batch_size,
        return_timestamps=False,
    )
    gen_kwargs = {"language": "<|ar|>", "task": "transcribe", "max_new_tokens": 128}

    # Buffer rows then batch through pipeline; persist after each batch so a
    # mid-run kill loses at most one batch.
    written = 0
    t0 = time.perf_counter()
    audios, refs, ids = [], [], []
    BATCH = args.batch_size
    out = args.output.open("w")

    def flush():
        nonlocal written, audios, refs, ids
        if not audios:
            return
        results = pipe(audios, generate_kwargs=gen_kwargs)
        for rid, ref, res in zip(ids, refs, results):
            out.write(json.dumps({
                "audio_filepath": rid,
                "text": ref,
                "pred_text": (res.get("text") if isinstance(res, dict) else "").strip(),
            }, ensure_ascii=False) + "\n")
            out.flush()
        written += len(audios)
        audios, refs, ids = [], [], []

    for i, (rid, ref, audio) in enumerate(tqdm(stream_dataset(spec), desc=args.dataset)):
        if args.max_rows and i >= args.max_rows:
            break
        if not ref or len(ref.split()) < 1:
            continue
        audios.append(audio["array"])
        refs.append(ref)
        ids.append(str(rid))
        if len(audios) >= BATCH:
            flush()
            if written and written % 500 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  [{written}] elapsed={elapsed/60:.1f}min  "
                      f"throughput={written/elapsed:.2f} utt/s", flush=True)
    flush()
    out.close()

    elapsed = time.perf_counter() - t0
    print(f"=== {args.dataset} done: n={written}  wall={elapsed/60:.1f}min  "
          f"throughput={written/elapsed:.2f} utt/s", flush=True)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
