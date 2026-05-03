"""Benchmark NVIDIA Conformer-CTC-large-Arabic via Sherpa-onnx on CPU.

NVIDIA's `nvidia/stt_ar_conformer_ctc_large` is the top-ranked open
Arabic ASR model on the Open Universal Arabic ASR Leaderboard
(25.71 % avg WER, Wang et al. 2024). Architecture is Conformer encoder
+ CTC head — completely different from Whisper, no autoregressive
decoder, ~120 M params. Cannot be CT2-converted (no decoder graph),
but exports cleanly to ONNX and runs efficiently on CPU via Sherpa-onnx.

This script writes EvalResult rows to `runs/results.jsonl` in the same
schema as `src.eval_harness` so the model can be plotted / tabled
alongside Whisper rows; tagged `backend=sherpa-onnx-conformer-ctc` in
extra.

Setup (on a fresh c3-standard-8 with python 3.10):

    pip install -U sherpa-onnx onnxruntime soundfile librosa jiwer numpy psutil
    # Download the NeMo-exported Sherpa-ready bundle (nvidia stt_ar_conformer_ctc_large
    # was pre-exported by csukuangfj/k2-fsa team and lives at:
    #   sherpa-onnx-nemo-ctc-stt-ar-1.0.0  (or download the .nemo and export ourselves)
    # We download the Sherpa-prepared bundle via the bundled CLI helper:
    python -c "from sherpa_onnx.utils import download_file; ..."

Usage:

    python scripts/run_conformer_ctc_bench.py \\
        --model-dir /opt/sherpa-onnx-nemo-ctc-ar \\
        --test-set test_sets/test_v3_clean_egyptian.jsonl \\
        --dialect egyptian --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jiwer
import librosa
import numpy as np
import psutil

from src.bootstrap_ci import bootstrap_cer_ci, bootstrap_wer_ci
from src.eval_harness import EvalConfig, EvalResult, get_git_commit, get_hardware_id
from src.normalization import NORMALIZER_VERSION, normalize_arabic


def build_recognizer(model_dir: Path, num_threads: int):
    """Build a Sherpa-onnx OfflineRecognizer from a NeMo-CTC bundle.

    Expects model_dir to contain `model.onnx` (or `model.int8.onnx`) plus
    `tokens.txt`. The Sherpa-prepared bundles published by k2-fsa for
    NVIDIA NeMo models follow this layout.
    """
    import sherpa_onnx

    model_files = list(model_dir.glob("*.onnx"))
    if not model_files:
        raise FileNotFoundError(f"no .onnx file in {model_dir}")
    # Prefer int8 if available (smaller, faster on CPU)
    int8_files = [f for f in model_files if "int8" in f.name]
    onnx_file = (int8_files or model_files)[0]
    tokens_file = model_dir / "tokens.txt"
    if not tokens_file.exists():
        raise FileNotFoundError(f"no tokens.txt in {model_dir}")
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(onnx_file),
        tokens=str(tokens_file),
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    ), str(onnx_file)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--model-name", default="nvidia-conformer-ctc-large-ar")
    p.add_argument("--test-set", required=True, type=Path)
    p.add_argument("--dialect", required=True)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--platform-label", default="c3-standard-8-clean")
    p.add_argument("--output-log", default=Path("runs/results.jsonl"), type=Path)
    p.add_argument("--predictions-dir", default=Path("runs/predictions"), type=Path)
    args = p.parse_args()

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_dir} ...", flush=True)
    recognizer, onnx_path = build_recognizer(args.model_dir, args.num_threads)
    print(f"loaded ({onnx_path})", flush=True)

    samples = [json.loads(l) for l in args.test_set.open() if l.strip()][:args.max_samples]
    print(f"n={len(samples)} dialect={args.dialect}", flush=True)

    proc = psutil.Process()
    refs, hyps, ttfts_ms = [], [], []
    total_audio = 0.0
    peak_rss = proc.memory_info().rss
    start = time.perf_counter()
    for i, s in enumerate(samples):
        try:
            audio, sr = librosa.load(s["audio"], sr=16000)
            t0 = time.perf_counter()
            stream = recognizer.create_stream()
            stream.accept_waveform(sr, audio)
            recognizer.decode_stream(stream)
            text = stream.result.text
            ttfts_ms.append((time.perf_counter() - t0) * 1000.0)
            refs.append(normalize_arabic(s["reference"]))
            hyps.append(normalize_arabic(text))
            total_audio += float(s.get("duration_s", len(audio) / sr))
            peak_rss = max(peak_rss, proc.memory_info().rss)
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
    print(f"=== {args.dialect}: WER={wer:.4f}  CI=[{wer_lo:.4f}, {wer_hi:.4f}]  "
          f"RTF={elapsed/total_audio:.3f}", flush=True)

    result = EvalResult(
        config=asdict(EvalConfig(
            model_path=onnx_path,
            model_name=args.model_name,
            compute_type="int8" if "int8" in onnx_path else "fp32",
            beam_size=1,  # greedy CTC
            cpu_threads=args.num_threads,
            device="cpu",
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
        hardware_id=get_hardware_id(),
        platform_label=args.platform_label,
        normalizer_version=NORMALIZER_VERSION,
        git_commit=get_git_commit(),
        extra={"backend": "sherpa-onnx-conformer-ctc"},
    )
    with args.output_log.open("a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    pred_path = args.predictions_dir / (
        f"preds_{args.model_name}_{args.dialect}_{args.platform_label}.jsonl"
    )
    with pred_path.open("w") as f:
        for s, hyp, ref in zip(samples, hyps, refs):
            f.write(json.dumps({"audio": s["audio"], "reference": ref, "hypothesis": hyp},
                               ensure_ascii=False) + "\n")
    print(f"wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()
