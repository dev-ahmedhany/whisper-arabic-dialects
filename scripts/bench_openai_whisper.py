"""OpenAI's `whisper` package benchmark — measures WER/RTF/TTFT for the
original reference inference path.

OpenAI's `whisper` PyPI package is the FIRST inference path released for
Whisper (2022). It's distinct from:
  - transformers.WhisperForConditionalGeneration (HF's port)
  - faster-whisper / CT2 (production runtime)
  - whisper.cpp (C++ + GGML)

It uses ffmpeg for audio loading, fp32 PyTorch tensors, and OpenAI's hand-rolled
beam search. Slightly different numerical paths from the HF port produce small
WER differences. Reference-of-references for the paper.

Usage:
    pip install -U openai-whisper
    python -m scripts.bench_openai_whisper \\
        --model large-v3-turbo \\
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
                   choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "large-v3-turbo"])
    p.add_argument("--test-set", required=True, type=Path)
    p.add_argument("--dialect", required=True)
    p.add_argument("--platform-label", required=True)
    p.add_argument("--output-log", default=Path("runs/results.jsonl"), type=Path)
    p.add_argument("--predictions-dir", default=Path("runs/predictions"), type=Path)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
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

    config = EvalConfig(
        model_path=f"openai-whisper:{args.model}",
        model_name=f"zero-shot-{args.model}-openai",
        compute_type="float32",
        beam_size=args.beam_size,
        cpu_threads=0,  # openai whisper doesn't expose CPU thread control directly
        device=args.device,
        language="ar",
        task="transcribe",
    )

    print(f"loading openai-whisper {args.model} on {args.device}...", flush=True)
    import whisper
    model = whisper.load_model(args.model, device=args.device)

    sampler = _PeakMemorySampler()
    sampler.start()
    hypotheses, references, ttfts_ms = [], [], []
    total_audio = 0.0
    n_failed = 0

    print("warmup...", flush=True)
    try:
        _ = model.transcribe(samples[0]["audio"], language="ar", task="transcribe", beam_size=args.beam_size, fp16=False)
    except Exception as exc:
        print(f"[warn] warmup failed: {exc}", flush=True)

    print(f"running {len(samples)} samples...", flush=True)
    start = time.perf_counter()
    for sample in samples:
        try:
            t_call = time.perf_counter()
            result = model.transcribe(
                sample["audio"],
                language="ar",
                task="transcribe",
                beam_size=args.beam_size,
                fp16=(args.device == "cuda"),
            )
            ttfts_ms.append((time.perf_counter() - t_call) * 1000.0)
            text = result["text"]
            hypotheses.append(normalize_arabic(text))
            references.append(normalize_arabic(sample["reference"]))
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
        extra={"backend": "openai-whisper"},
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
        f"[{config.model_name} | openai-whisper | float32 | b={args.beam_size} | {args.dialect}] "
        f"WER={result.wer:.4f} [{result.wer_ci_lo:.4f}, {result.wer_ci_hi:.4f}]  "
        f"RTF={result.rtf:.3f}  TTFT_p95={result.ttft_ms_p95:.0f}ms  n={result.n_samples}"
    )


if __name__ == "__main__":
    main()
