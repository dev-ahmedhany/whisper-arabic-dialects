"""Bench CohereLabs/cohere-transcribe-03-2026 on Arabic test sets via HF transformers.

2B-param Conformer encoder + Transformer decoder, Apache-2.0. CPU-deployable.

Output schema mirrors src.eval_harness so rows land in runs/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from dataclasses import asdict
from pathlib import Path

import jiwer
import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap_ci import bootstrap_cer_ci, bootstrap_wer_ci
from src.eval_harness import EvalConfig, EvalResult, get_git_commit, get_hardware_id
from src.normalization import NORMALIZER_VERSION, normalize_arabic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--model-name", default="cohere-transcribe-03")
    p.add_argument("--test-set", required=True, type=Path)
    p.add_argument("--dialect", default="mediaspeech")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--language", default="ar")
    p.add_argument("--platform-label", default="cpu")
    p.add_argument("--output-log", default=Path("runs/results.jsonl"), type=Path)
    p.add_argument("--predictions-dir", default=Path("runs/predictions"), type=Path)
    args = p.parse_args()

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} on {args.device} ...", flush=True)
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    from transformers.audio_utils import load_audio

    proc_obj = psutil.Process()
    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, device_map=args.device,
    )
    model.train(False)
    print(f"loaded; mem={proc_obj.memory_info().rss/1e9:.1f} GB", flush=True)

    samples = [json.loads(l) for l in args.test_set.open() if l.strip()][:args.max_samples]
    print(f"n={len(samples)} dialect={args.dialect}", flush=True)

    refs, hyps, ttfts_ms = [], [], []
    total_audio = 0.0
    peak_rss = proc_obj.memory_info().rss
    start = time.perf_counter()
    for i, s in enumerate(samples):
        try:
            t0 = time.perf_counter()
            audio = load_audio(s["audio"], sampling_rate=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt",
                               language=args.language).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            text = processor.decode(outputs[0], skip_special_tokens=True)
            ttfts_ms.append((time.perf_counter() - t0) * 1000.0)
            refs.append(normalize_arabic(s["reference"]))
            hyps.append(normalize_arabic(text))
            total_audio += float(s.get("duration_s", 0.0))
            peak_rss = max(peak_rss, proc_obj.memory_info().rss)
            if i % 10 == 0:
                print(f"  [{i}/{len(samples)}] hyp={text[:80]!r}", flush=True)
        except Exception as e:
            print(f"  fail @ {i}: {type(e).__name__}: {e}", flush=True)
    elapsed = time.perf_counter() - start

    if not refs:
        raise RuntimeError("all samples failed")

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    wer_m, wer_lo, wer_hi = bootstrap_wer_ci(refs, hyps, n_bootstrap=1000)
    cer_m, cer_lo, cer_hi = bootstrap_cer_ci(refs, hyps, n_bootstrap=1000)
    ttft_arr = np.array(ttfts_ms) if ttfts_ms else np.array([np.nan])
    print(f"=== {args.dialect}: WER={wer:.4f} CI=[{wer_lo:.4f},{wer_hi:.4f}] "
          f"RTF={elapsed/total_audio:.3f}", flush=True)

    result = EvalResult(
        config=asdict(EvalConfig(
            model_path=args.model, model_name=args.model_name,
            compute_type=str(dtype).split(".")[-1], beam_size=1,
            cpu_threads=psutil.cpu_count() if args.device == "cpu" else 0,
            device=args.device,
        )),
        test_set=str(args.test_set), dialect=args.dialect,
        n_samples=len(refs), n_failed=len(samples) - len(refs),
        total_audio_seconds=total_audio, total_compute_seconds=elapsed,
        rtf=elapsed/total_audio if total_audio > 0 else float("nan"),
        throughput_realtime_x=total_audio/elapsed if elapsed > 0 else float("nan"),
        peak_memory_mb=peak_rss / (1024**2),
        wer=wer, wer_ci_lo=wer_lo, wer_ci_hi=wer_hi,
        cer=cer, cer_ci_lo=cer_lo, cer_ci_hi=cer_hi,
        ttft_ms_mean=float(np.nanmean(ttft_arr)),
        ttft_ms_p50=float(np.nanpercentile(ttft_arr, 50)),
        ttft_ms_p95=float(np.nanpercentile(ttft_arr, 95)),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hardware_id=get_hardware_id(), platform_label=args.platform_label,
        normalizer_version=NORMALIZER_VERSION, git_commit=get_git_commit(),
        extra={"backend": "cohere-transcribe-03", "language": args.language},
    )
    with args.output_log.open("a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    pp = args.predictions_dir / (
        f"preds_{args.model_name}_{args.dialect}_{args.platform_label}.jsonl"
    )
    with pp.open("w") as f:
        for s, hp, rf in zip(samples, hyps, refs):
            f.write(json.dumps({"audio": s["audio"], "reference": rf,
                                "hypothesis": hp}, ensure_ascii=False) + "\n")
    print(f"wrote {pp}", flush=True)


if __name__ == "__main__":
    main()
