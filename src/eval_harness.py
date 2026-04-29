"""CPU evaluation harness — the foundation every WER number in the paper passes through.

Cardinal rules:
  1. Identical normalization across all rows.
  2. Log to JSONL immediately, never reconstruct from memory.
  3. CPU only by default. GPU only for an FP16 ceiling reference, clearly labeled.
  4. Hardware fingerprint baked into every row.
  5. Same image runs on GCP and Hetzner.

Run as a module:
    python -m src.eval_harness \\
        --model openai/whisper-large-v3-turbo \\
        --model-name zero-shot-turbo \\
        --compute-type int8_float32 \\
        --beam-size 1 --cpu-threads 4 \\
        --test-set test_sets/fleurs_msa.jsonl --dialect msa \\
        --platform-label gcp-c3-standard-8 \\
        --output-log runs/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import jiwer
import psutil
from faster_whisper import WhisperModel

from src.bootstrap_ci import bootstrap_cer_ci, bootstrap_wer_ci
from src.normalization import NORMALIZER_VERSION, normalize_arabic


@dataclass
class EvalConfig:
    model_path: str
    model_name: str
    compute_type: str
    beam_size: int
    cpu_threads: int
    device: str = "cpu"
    language: str = "ar"
    task: str = "transcribe"


@dataclass
class EvalResult:
    config: dict
    test_set: str
    dialect: str
    n_samples: int
    n_failed: int
    total_audio_seconds: float
    total_compute_seconds: float
    rtf: float
    throughput_realtime_x: float
    peak_memory_mb: float
    wer: float
    wer_ci_lo: float
    wer_ci_hi: float
    cer: float
    cer_ci_lo: float
    cer_ci_hi: float
    # Time-to-first-token: ms from model.transcribe(audio) call to first segment yield.
    # End-to-end including audio load + mel extraction + first-segment decode. The
    # paper's Real-time captioning recommendation row uses ttft_ms_p95 as its latency
    # constraint (sub-second tail latency is the bar for usable live captioning).
    ttft_ms_mean: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    timestamp: str
    hardware_id: str
    platform_label: str
    normalizer_version: str
    git_commit: str
    extra: dict = field(default_factory=dict)


class _PeakMemorySampler(threading.Thread):
    """Polls RSS in the background. Main-thread polling misses transient peaks."""

    def __init__(self, interval_s: float = 0.1):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        # NOT `self._stop` — that name shadows threading.Thread._stop, a private
        # method CPython calls internally on thread shutdown. Shadowing it with
        # an Event raises "'Event' object is not callable" at exit.
        self._stop_event = threading.Event()
        self._proc = psutil.Process(os.getpid())
        self.peak_mb = self._proc.memory_info().rss / (1024 * 1024)

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                if rss_mb > self.peak_mb:
                    self.peak_mb = rss_mb
            except psutil.Error:
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self) -> float:
        self._stop_event.set()
        self.join(timeout=2.0)
        return self.peak_mb


def get_hardware_id() -> str:
    cpu_brand = platform.processor() or "unknown"
    cpu_brand = cpu_brand.replace(" ", "_").replace(",", "")[:48]
    return f"{cpu_brand}_{psutil.cpu_count(logical=True)}t_{int(psutil.virtual_memory().total / (1024 ** 3))}gb"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def _load_test_set(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_model(
    config: EvalConfig,
    test_set_path: Path,
    dialect: str,
    output_log: Path,
    platform_label: str,
    predictions_dir: Path,
    n_bootstrap: int = 1000,
    max_samples: Optional[int] = None,
) -> EvalResult:
    output_log.parent.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_test_set(test_set_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(f"no samples in {test_set_path}")

    sampler = _PeakMemorySampler()
    sampler.start()

    try:
        model = WhisperModel(
            config.model_path,
            device=config.device,
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
        )

        segments_iter, _ = model.transcribe(
            samples[0]["audio"],
            beam_size=config.beam_size,
            language=config.language,
            task=config.task,
        )
        list(segments_iter)

        hypotheses: list[str] = []
        references: list[str] = []
        ttfts_ms: list[float] = []
        total_audio = 0.0
        n_failed = 0

        start = time.perf_counter()
        for sample in samples:
            try:
                t_call = time.perf_counter()
                segs, info = model.transcribe(
                    sample["audio"],
                    beam_size=config.beam_size,
                    language=config.language,
                    task=config.task,
                )
                # The segments object is a lazy generator — actual decoding work doesn't
                # happen until next() is called. Time-to-first-token is the wall-clock
                # from the transcribe() call to the first segment yielding.
                seg_iter = iter(segs)
                first_seg = next(seg_iter, None)
                ttfts_ms.append((time.perf_counter() - t_call) * 1000.0)
                pieces = [first_seg.text] if first_seg is not None else []
                pieces.extend(s.text for s in seg_iter)
                text = " ".join(pieces)
                hypotheses.append(normalize_arabic(text))
                references.append(normalize_arabic(sample["reference"]))
                total_audio += float(info.duration)
            except Exception as exc:
                n_failed += 1
                print(f"[warn] sample failed: {sample.get('audio')!r}: {exc}", flush=True)
        elapsed = time.perf_counter() - start
    finally:
        peak_mem = sampler.stop()

    if not references:
        raise RuntimeError("all samples failed; nothing to score")

    wer_mean, wer_lo, wer_hi = bootstrap_wer_ci(references, hypotheses, n_bootstrap=n_bootstrap)
    cer_mean, cer_lo, cer_hi = bootstrap_cer_ci(references, hypotheses, n_bootstrap=n_bootstrap)

    import numpy as _np
    ttft_arr = _np.array(ttfts_ms, dtype=_np.float64) if ttfts_ms else _np.array([_np.nan])
    ttft_mean = float(_np.nanmean(ttft_arr))
    ttft_p50 = float(_np.nanpercentile(ttft_arr, 50))
    ttft_p95 = float(_np.nanpercentile(ttft_arr, 95))

    result = EvalResult(
        config=asdict(config),
        test_set=str(test_set_path),
        dialect=dialect,
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
        ttft_ms_mean=ttft_mean,
        ttft_ms_p50=ttft_p50,
        ttft_ms_p95=ttft_p95,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hardware_id=get_hardware_id(),
        platform_label=platform_label,
        normalizer_version=NORMALIZER_VERSION,
        git_commit=get_git_commit(),
    )

    with output_log.open("a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    pred_filename = (
        f"preds_{config.model_name}_{config.compute_type}"
        f"_b{config.beam_size}_t{config.cpu_threads}"
        f"_{dialect}_{platform_label}.jsonl"
    )
    with (predictions_dir / pred_filename).open("w") as f:
        for sample, hyp, ref in zip(samples, hypotheses, references):
            f.write(
                json.dumps(
                    {"audio": sample["audio"], "reference": ref, "hypothesis": hyp},
                    ensure_ascii=False,
                )
                + "\n"
            )

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single CPU evaluation cell.")
    p.add_argument("--model", required=True, help="HF model id or local CT2 dir")
    p.add_argument("--model-name", required=True, help="short label for filenames/logs")
    p.add_argument(
        "--compute-type",
        required=True,
        choices=["float32", "float16", "int8_float32", "int8_float16", "int8"],
    )
    p.add_argument("--beam-size", type=int, default=1)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--language", default="ar")
    p.add_argument("--test-set", required=True, type=Path)
    p.add_argument("--dialect", required=True)
    p.add_argument("--platform-label", required=True, help="e.g. gcp-c3-standard-8 or hetzner-cx53")
    p.add_argument("--output-log", default=Path("runs/results.jsonl"), type=Path)
    p.add_argument("--predictions-dir", default=Path("runs/predictions"), type=Path)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap samples for smoke runs; omit for full eval")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = EvalConfig(
        model_path=args.model,
        model_name=args.model_name,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        device=args.device,
        language=args.language,
    )
    try:
        result = evaluate_model(
            config=config,
            test_set_path=args.test_set,
            dialect=args.dialect,
            output_log=args.output_log,
            platform_label=args.platform_label,
            predictions_dir=args.predictions_dir,
            n_bootstrap=args.n_bootstrap,
            max_samples=args.max_samples,
        )
    except Exception:
        traceback.print_exc()
        raise

    print(
        f"[{config.model_name} | {config.compute_type} | b={config.beam_size} | "
        f"t={config.cpu_threads} | {args.dialect}] "
        f"WER={result.wer:.4f} [{result.wer_ci_lo:.4f}, {result.wer_ci_hi:.4f}]  "
        f"RTF={result.rtf:.3f}  "
        f"peak={result.peak_memory_mb:.0f}MB  "
        f"n={result.n_samples} (failed={result.n_failed})"
    )


if __name__ == "__main__":
    main()
