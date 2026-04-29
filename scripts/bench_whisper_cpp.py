"""whisper.cpp benchmark — measures WER + RTF + TTFT for the C++ inference engine.

whisper.cpp is OpenAI's whisper compiled to C++ using GGML. Often 2-3x faster
than CT2 on CPU for streaming workloads (chunked audio, sub-second TTFT). Uses
its own GGML quantization formats (q4_0, q5_0, q8_0, fp16, fp32) — different
from CT2's int8/int8_fp32/etc.

This script wraps pywhispercpp (the Python bindings) and emits rows compatible
with src.eval_harness so they join cleanly in runs/results.jsonl with
extra.backend = "whisper.cpp".

Usage:
    pip install pywhispercpp
    python -m scripts.bench_whisper_cpp \\
        --model tiny --quantization q5_0 \\
        --test-set test_sets/test_msa_fleurs_msa_test.jsonl \\
        --dialect msa --max-samples 50 \\
        --platform-label gcp-c3-standard-8 \\
        --output-log runs/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import jiwer

from src.bootstrap_ci import bootstrap_cer_ci, bootstrap_wer_ci
from src.eval_harness import (
    EvalConfig,
    EvalResult,
    _PeakMemorySampler,
    get_git_commit,
    get_hardware_id,
)
from src.normalization import NORMALIZER_VERSION, normalize_arabic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
                   help="whisper.cpp model size (downloads GGML file on first use)")
    p.add_argument("--quantization", default="q5_0",
                   choices=["q4_0", "q5_0", "q5_1", "q8_0", "fp16", "fp32"],
                   help="GGML quantization level (pywhispercpp ships q5_1 for "
                        "tiny/base/small and q5_0 for medium/large)")
    p.add_argument("--test-set", required=True, type=Path)
    p.add_argument("--dialect", required=True)
    p.add_argument("--platform-label", required=True)
    p.add_argument("--output-log", default=Path("runs/results.jsonl"), type=Path)
    p.add_argument("--predictions-dir", default=Path("runs/predictions"), type=Path)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--beam-size", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    args = p.parse_args()

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    samples = [json.loads(l) for l in args.test_set.read_text().splitlines() if l.strip()]
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"no samples in {args.test_set}")

    # whisper.cpp model name convention: ggml-{model}-{quantization}.bin
    # pywhispercpp resolves shorthand like "tiny" / "base.en" automatically; for
    # quantized variants we pass the full GGML name.
    if args.quantization == "fp32":
        model_id = args.model
    else:
        model_id = f"{args.model}-{args.quantization}"

    config = EvalConfig(
        model_path=model_id,
        model_name=f"zero-shot-{args.model}-cpp",
        compute_type=args.quantization,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        device="cpu",
        language="ar",
        task="transcribe",
    )

    print(f"loading whisper.cpp model: {model_id} (threads={args.cpu_threads})", flush=True)
    from pywhispercpp.model import Model
    model = Model(model_id, n_threads=args.cpu_threads, language="ar")

    sampler = _PeakMemorySampler()
    sampler.start()
    hypotheses, references, ttfts_ms = [], [], []
    total_audio = 0.0
    n_failed = 0

    print("warmup...", flush=True)
    try:
        _ = model.transcribe(samples[0]["audio"])
    except Exception as exc:
        print(f"[warn] warmup failed: {exc}", flush=True)

    print(f"running {len(samples)} samples...", flush=True)
    start = time.perf_counter()
    for sample in samples:
        try:
            t_call = time.perf_counter()
            segments = model.transcribe(sample["audio"])
            ttfts_ms.append((time.perf_counter() - t_call) * 1000.0)
            text = " ".join(s.text for s in segments)
            hypotheses.append(normalize_arabic(text))
            references.append(normalize_arabic(sample["reference"]))
            # whisper.cpp doesn't expose audio duration; compute from file
            import soundfile as sf
            info = sf.info(sample["audio"])
            total_audio += float(info.frames) / float(info.samplerate)
        except Exception as exc:
            n_failed += 1
            print(f"[warn] sample failed {sample.get('audio')!r}: {exc}", flush=True)
    elapsed = time.perf_counter() - start
    peak_mem = sampler.stop()

    if not references:
        raise RuntimeError("all samples failed; nothing to score")

    wer_mean, wer_lo, wer_hi = bootstrap_wer_ci(references, hypotheses, n_bootstrap=args.n_bootstrap)
    cer_mean, cer_lo, cer_hi = bootstrap_cer_ci(references, hypotheses, n_bootstrap=args.n_bootstrap)
    ttft_arr = np.array(ttfts_ms, dtype=np.float64) if ttfts_ms else np.array([np.nan])

    result = EvalResult(
        config=asdict(config),
        test_set=str(args.test_set),
        dialect=args.dialect,
        n_samples=len(references),
        n_failed=n_failed,
        total_audio_seconds=total_audio,
        total_compute_seconds=elapsed,
        rtf=elapsed / total_audio if total_audio > 0 else float("nan"),
        throughput_realtime_x=total_audio / elapsed if elapsed > 0 else float("nan"),
        peak_memory_mb=peak_mem,
        wer=jiwer.wer(references, hypotheses),
        wer_ci_lo=wer_lo,
        wer_ci_hi=wer_hi,
        cer=jiwer.cer(references, hypotheses),
        cer_ci_lo=cer_lo,
        cer_ci_hi=cer_hi,
        ttft_ms_mean=float(np.nanmean(ttft_arr)),
        ttft_ms_p50=float(np.nanpercentile(ttft_arr, 50)),
        ttft_ms_p95=float(np.nanpercentile(ttft_arr, 95)),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hardware_id=get_hardware_id(),
        platform_label=args.platform_label,
        normalizer_version=NORMALIZER_VERSION,
        git_commit=get_git_commit(),
        extra={"backend": "whisper.cpp"},
    )

    with args.output_log.open("a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    pred_filename = (
        f"preds_{config.model_name}_{config.compute_type}_b{config.beam_size}_t{config.cpu_threads}_"
        f"{args.dialect}_{args.platform_label}.jsonl"
    )
    with (args.predictions_dir / pred_filename).open("w") as f:
        for sample, hyp, ref in zip(samples, hypotheses, references):
            f.write(json.dumps({"audio": sample["audio"], "reference": ref, "hypothesis": hyp}, ensure_ascii=False) + "\n")

    print(
        f"[{config.model_name} | whisper.cpp | {args.quantization} | t={args.cpu_threads} | {args.dialect}] "
        f"WER={result.wer:.4f} [{result.wer_ci_lo:.4f}, {result.wer_ci_hi:.4f}]  "
        f"RTF={result.rtf:.3f}  TTFT_p95={result.ttft_ms_p95:.0f}ms  n={result.n_samples}"
    )


if __name__ == "__main__":
    main()
