"""Deterministically split a single-dialect JSONL into train/val 95/5.

Used to build per-dialect FT comparison configs (configs/train_*_<dialect>.yaml)
that train a specialist adapter on one dialect's source data only. Pairs with
the multi-dialect train.jsonl in answering: does single multi-dialect FT
outperform per-dialect specialist FT on that dialect's test set?

Usage:
    python -m scripts.prep_per_dialect_split \\
        --in test_sets/mgb3_egyptian_train.jsonl \\
        --train-out test_sets/eg_train.jsonl \\
        --val-out   test_sets/eg_val.jsonl \\
        --val-fraction 0.05 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", type=Path, required=True)
    p.add_argument("--train-out", type=Path, required=True)
    p.add_argument("--val-out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(50, int(args.val_fraction * len(rows)))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    with args.train_out.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with args.val_out.open("w") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train: {len(train_rows)} rows -> {args.train_out}")
    print(f"val:   {len(val_rows)} rows -> {args.val_out}")


if __name__ == "__main__":
    main()
