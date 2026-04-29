"""Merge LoRA into base, then sweep CT2 quantization variants for the benchmark matrix.

Produces:
  whisper-merged/                         (HF format, fp32 weights)
  whisper-faster-<model>-float32/         (CT2)
  whisper-faster-<model>-float16/         (CT2)
  whisper-faster-<model>-int8_float32/    (CT2)
  whisper-faster-<model>-int8_float16/    (CT2)
  whisper-faster-<model>-int8/            (CT2)

All five variants are then ready to feed `faster_whisper.WhisperModel(local_dir, ...)`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

QUANTS = ["float32", "float16", "int8_float32", "int8_float16", "int8"]


def merge_lora(base_model: str, lora_dir: Path, merged_dir: Path) -> None:
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    base = WhisperForConditionalGeneration.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model = model.merge_and_unload()
    model.save_pretrained(str(merged_dir))
    processor = WhisperProcessor.from_pretrained(base_model)
    processor.save_pretrained(str(merged_dir))


def convert_to_ct2(merged_dir: Path, output_dir: Path, quant: str) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    cmd = [
        "ct2-transformers-converter",
        "--model", str(merged_dir),
        "--output_dir", str(output_dir),
        "--quantization", quant,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="HF model id, e.g. openai/whisper-large-v3-turbo")
    p.add_argument("--lora-dir", type=Path, required=True)
    p.add_argument("--merged-dir", type=Path, default=Path("whisper-merged"))
    p.add_argument("--output-prefix", required=True,
                   help="prefix used for CT2 dirs, e.g. whisper-faster-turbo-ar")
    p.add_argument("--quantizations", nargs="+", default=QUANTS, choices=QUANTS)
    p.add_argument("--skip-merge", action="store_true",
                   help="reuse existing merged dir (skip the LoRA merge step)")
    args = p.parse_args()

    if not args.skip_merge:
        print(f"Merging LoRA from {args.lora_dir} into {args.base_model} → {args.merged_dir}")
        merge_lora(args.base_model, args.lora_dir, args.merged_dir)

    for q in args.quantizations:
        out = Path(f"{args.output_prefix}-{q}")
        print(f"\n=== Converting to {q} → {out} ===")
        convert_to_ct2(args.merged_dir, out, q)

    print("\nDone. CT2 variants written:")
    for q in args.quantizations:
        print(f"  {args.output_prefix}-{q}")


if __name__ == "__main__":
    main()
