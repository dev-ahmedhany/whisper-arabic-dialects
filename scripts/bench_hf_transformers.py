"""HF transformers reference benchmark — measures the WER + RTF effect of using
the original `transformers.WhisperForConditionalGeneration` vs the CT2-converted
faster-whisper variants the rest of the benchmark uses.

Drops a row into runs/results.jsonl with the same schema as src.eval_harness
but `extra.backend = "hf-transformers"` so it's joinable with the CT2 rows
in the paper's tables. The point is paper §3.3 methodology validation — does
CT2 conversion meaningfully shift WER?

Usage on a bench VM (or anywhere with torch + transformers):
    python -m scripts.bench_hf_transformers \\
        --model openai/whisper-large-v3-turbo \\
        --model-name zero-shot-turbo-hf \\
        --test-set test_sets/test_msa_fleurs_msa_test.jsonl \\
        --dialect msa \\
        --max-samples 100 \\
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
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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


def _load_audio_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id, e.g. openai/whisper-large-v3-turbo")
    p.add_argument("--model-name", required=True, help="short label, e.g. zero-shot-turbo-hf")
    p.add_argument("--test-set", required=True, type=Path)
    p.add_argument("--dialect", required=True)
    p.add_argument("--platform-label", required=True)
    p.add_argument("--output-log", default=Path("runs/results.jsonl"), type=Path)
    p.add_argument("--predictions-dir", default=Path("runs/predictions"), type=Path)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
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
        model_path=args.model,
        model_name=args.model_name,
        compute_type=args.dtype,
        beam_size=args.beam_size,
        cpu_threads=torch.get_num_threads() if args.device == "cpu" else 0,
        device=args.device,
        language="ar",
        task="transcribe",
    )

    print(f"loading {args.model} on {args.device} as {args.dtype}...", flush=True)
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device)
    # PyTorch eval-mode (disables dropout/batchnorm-train-mode) — not Python's eval()
    model = model.eval()
    forced_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")

    sampler = _PeakMemorySampler()
    sampler.start()
    hypotheses, references, ttfts_ms = [], [], []
    total_audio = 0.0
    n_failed = 0

    print("warmup...", flush=True)
    with torch.inference_mode():
        warm_audio = _load_audio_16k(samples[0]["audio"])
        inputs = processor(warm_audio, sampling_rate=16000, return_tensors="pt").to(args.device)
        if args.dtype != "float32":
            inputs["input_features"] = inputs["input_features"].to(torch_dtype)
        _ = model.generate(inputs["input_features"], forced_decoder_ids=forced_ids,
                           num_beams=args.beam_size, max_new_tokens=440)

    print(f"running {len(samples)} samples...", flush=True)
    start = time.perf_counter()
    with torch.inference_mode():
        for sample in samples:
            try:
                audio = _load_audio_16k(sample["audio"])
                t_call = time.perf_counter()
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(args.device)
                if args.dtype != "float32":
                    inputs["input_features"] = inputs["input_features"].to(torch_dtype)
                generated = model.generate(
                    inputs["input_features"],
                    forced_decoder_ids=forced_ids,
                    num_beams=args.beam_size,
                    max_new_tokens=440,
                )
                ttfts_ms.append((time.perf_counter() - t_call) * 1000.0)
                text = processor.batch_decode(generated, skip_special_tokens=True)[0]
                hypotheses.append(normalize_arabic(text))
                references.append(normalize_arabic(sample["reference"]))
                total_audio += float(len(audio)) / 16000.0
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
        extra={"backend": "hf-transformers"},
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
        f"[{config.model_name} | hf-transformers | {args.dtype} | b={args.beam_size} | {args.dialect}] "
        f"WER={result.wer:.4f} [{result.wer_ci_lo:.4f}, {result.wer_ci_hi:.4f}]  "
        f"RTF={result.rtf:.3f}  TTFT_p95={result.ttft_ms_p95:.0f}ms  n={result.n_samples}"
    )


if __name__ == "__main__":
    main()
