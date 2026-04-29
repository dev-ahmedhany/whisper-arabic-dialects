"""Run zero-shot Whisper baselines on FLEURS Arabic — locally or on the benchmark host.

`--tiny` is the local smoke run: 50 FLEURS samples, single config, takes a few minutes
on a laptop. Confirms the pipeline wires up before any cloud spend.

Without --tiny it drives the full quality matrix from configs/benchmark_matrix_quality.yaml,
restricted to the two zero-shot models.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.eval_harness import EvalConfig, evaluate_model


def _ensure_fleurs_tiny(out_path: Path) -> None:
    if out_path.exists():
        return
    print(f"[info] preparing 50-sample FLEURS subset at {out_path}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "scripts.prepare_fleurs",
            "--split", "test",
            "--out", str(out_path),
            "--audio-dir", "audio/fleurs_ar_tiny",
            "--max-samples", "50",
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true", help="50-sample local smoke run")
    p.add_argument("--platform-label", default="local")
    p.add_argument("--output-log", type=Path, default=Path("runs/results.jsonl"))
    args = p.parse_args()

    if args.tiny:
        test_set = Path("test_sets/fleurs_ar_tiny.jsonl")
        _ensure_fleurs_tiny(test_set)
        cfg = EvalConfig(
            model_path="openai/whisper-large-v3-turbo",
            model_name="zero-shot-turbo",
            compute_type="int8_float32",
            beam_size=1,
            cpu_threads=4,
        )
        result = evaluate_model(
            config=cfg,
            test_set_path=test_set,
            dialect="msa",
            output_log=args.output_log,
            platform_label=args.platform_label,
            predictions_dir=Path("runs/predictions"),
            n_bootstrap=200,
        )
        wer_pct = result.wer * 100
        ok = 25 <= wer_pct <= 45
        print(f"[smoke] zero-shot turbo on FLEURS-50: WER={wer_pct:.1f}% "
              f"(expected 25–45%); RTF={result.rtf:.2f}; "
              f"{'OK' if ok else 'OUT OF EXPECTED RANGE — check pipeline'}")
        return

    print("[info] full zero-shot baseline → call run_benchmark_matrix instead:")
    print("  python -m scripts.run_benchmark_matrix \\")
    print("      --config configs/benchmark_matrix_quality.yaml \\")
    print("      --platform-label gcp-c3-standard-8 \\")
    print("      --include-models zero-shot-large-v3 zero-shot-turbo")


if __name__ == "__main__":
    main()
