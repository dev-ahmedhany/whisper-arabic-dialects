"""Build dialect-balanced train / val / test JSONL splits from per-dataset JSONL files.

Reads `configs/dataset_mix.yaml` describing which per-dataset files to draw from and
the target hours per dialect. Test sets (FLEURS, Casablanca) are NEVER mixed into
train — they are passed through unchanged for held-out evaluation.

Each output row schema:
    {
      "audio":          str (absolute path),
      "reference":      str (raw text, normalization happens later),
      "dialect":        str ("msa", "egyptian", "levantine", "gulf", "maghrebi"),
      "source_dataset": str,
      "duration_s":     float
    }
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import yaml


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _take_hours(rows: list[dict], target_hours: float, rng: random.Random) -> list[dict]:
    rng.shuffle(rows)
    target_seconds = target_hours * 3600.0
    selected: list[dict] = []
    cum = 0.0
    for row in rows:
        if cum >= target_seconds:
            break
        selected.append(row)
        cum += float(row.get("duration_s", 0.0))
    return selected


def build_splits(config_path: Path, output_dir: Path, seed: int = 42) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    rng = random.Random(seed)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    per_dialect_hours: dict[str, float] = defaultdict(float)

    for dialect, spec in cfg["train"].items():
        target_hours = float(spec["hours"])
        sources = [Path(p) for p in spec["sources"]]
        pooled: list[dict] = []
        for src in sources:
            rows = _load_jsonl(src)
            for r in rows:
                r["dialect"] = dialect
                r["source_dataset"] = src.stem
            pooled.extend(rows)
        chosen = _take_hours(pooled, target_hours, rng)
        rng.shuffle(chosen)
        n_val = max(50, int(0.05 * len(chosen)))
        val_rows.extend(chosen[:n_val])
        train_rows.extend(chosen[n_val:])
        per_dialect_hours[dialect] = sum(float(r["duration_s"]) for r in chosen) / 3600.0

    rng.shuffle(train_rows)
    _write_jsonl(train_rows, output_dir / "train.jsonl")
    _write_jsonl(val_rows, output_dir / "val.jsonl")

    for dialect, spec in cfg["test"].items():
        for src in spec["sources"]:
            src_path = Path(src)
            rows = _load_jsonl(src_path)
            for r in rows:
                r["dialect"] = dialect
                r["source_dataset"] = src_path.stem
            stem = src_path.stem
            _write_jsonl(rows, output_dir / f"test_{dialect}_{stem}.jsonl")

    summary = {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "hours_per_dialect": dict(per_dialect_hours),
        "total_train_hours": round(sum(per_dialect_hours.values()), 2),
        "seed": seed,
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/dataset_mix.yaml"))
    p.add_argument("--output-dir", type=Path, default=Path("test_sets"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_splits(args.config, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
