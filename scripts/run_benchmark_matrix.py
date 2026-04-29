"""Drive the eval harness across a config-defined matrix.

Iterates over (model × compute_type × beam_size × cpu_threads × test_set), invoking
`evaluate_model` for each cell. Skips cells already present in the output log
(model_name + compute + beam + threads + dialect + platform_label form a key) so
re-runs are idempotent.

Usage (Docker, on the benchmark host):
    docker run --rm \\
      -v $(pwd)/runs:/app/runs \\
      -v $(pwd)/test_sets:/app/test_sets \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      whisper-arabic-bench:latest \\
      --config configs/benchmark_matrix_quality.yaml \\
      --platform-label gcp-c3-standard-8

Local (no Docker):
    python -m scripts.run_benchmark_matrix \\
      --config configs/benchmark_matrix_quality.yaml \\
      --platform-label local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import yaml

from src.eval_harness import EvalConfig, evaluate_model


def _existing_keys(log_path: Path) -> set[tuple]:
    if not log_path.exists():
        return set()
    keys = set()
    with log_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cfg = row.get("config", {})
            keys.add(
                (
                    cfg.get("model_name"),
                    cfg.get("compute_type"),
                    cfg.get("beam_size"),
                    cfg.get("cpu_threads"),
                    row.get("dialect"),
                    row.get("test_set"),
                    row.get("platform_label"),
                )
            )
    return keys


def _resolve_path(template: str, compute_type: str) -> str:
    return template.format(compute_type=compute_type)


def _iter_cells(cfg: dict) -> Iterable[tuple[dict, str, int, int, dict]]:
    for model in cfg["models"]:
        for ct in cfg["compute_types"]:
            for bs in cfg["beam_sizes"]:
                for t in cfg["cpu_threads"]:
                    for ts in cfg["test_sets"]:
                        yield model, ct, bs, t, ts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--platform-label", required=True,
                   help="e.g. gcp-c3-standard-8 / hetzner-cx53 / local")
    p.add_argument("--output-log", type=Path, default=Path("runs/results.jsonl"))
    p.add_argument("--predictions-dir", type=Path, default=Path("runs/predictions"))
    p.add_argument("--max-samples", type=int, default=None,
                   help="cap samples per cell (smoke runs); omit for full eval")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true",
                   help="print cells without running")
    p.add_argument("--include-models", nargs="+", default=None,
                   help="restrict to a subset of model names from the config")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seen = _existing_keys(args.output_log)

    all_cells = list(_iter_cells(cfg))
    if args.include_models:
        all_cells = [c for c in all_cells if c[0]["name"] in args.include_models]
    print(f"[info] {len(all_cells)} cells in matrix; {len(seen)} already logged")

    n_run = 0
    n_skip = 0
    for model, compute_type, beam_size, cpu_threads, test_set in all_cells:
        model_path = _resolve_path(model["path"], compute_type)
        key = (
            model["name"],
            compute_type,
            beam_size,
            cpu_threads,
            test_set["dialect"],
            test_set["path"],
            args.platform_label,
        )
        if key in seen:
            n_skip += 1
            continue

        if args.dry_run:
            print(f"  [dry] {model['name']} {compute_type} b={beam_size} t={cpu_threads} "
                  f"{test_set['dialect']} {test_set['path']}")
            continue

        ec = EvalConfig(
            model_path=model_path,
            model_name=model["name"],
            compute_type=compute_type,
            beam_size=beam_size,
            cpu_threads=cpu_threads,
        )
        try:
            evaluate_model(
                config=ec,
                test_set_path=Path(test_set["path"]),
                dialect=test_set["dialect"],
                output_log=args.output_log,
                platform_label=args.platform_label,
                predictions_dir=args.predictions_dir,
                n_bootstrap=args.n_bootstrap,
                max_samples=args.max_samples,
            )
            n_run += 1
        except Exception as exc:
            print(f"[err] cell failed {key}: {exc}", file=sys.stderr)

    print(f"[done] ran {n_run} cells, skipped {n_skip} already-logged cells")


if __name__ == "__main__":
    main()
