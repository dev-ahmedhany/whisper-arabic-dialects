"""Build the curated v4 training mix at ~234h.

Caps each source at a target hour budget so MASC + SADA don't dominate
the dataset. Audio durations are read from the per-row duration_s field
written by build_v4_full_mix.py.

Per-source budgets (in hours):
    cv18:           65  (full)
    masc:           65  (sampled from 690h, seed=42)
    sada:           65  (sampled from 647h, seed=42)
    casablanca:     35  (full, all 5 countries)
    fleurs:          4  (full)
    -- TOTAL:      ~234

Also builds val_v4.jsonl: 200 rows per source held out before sampling
the train cap, so val is in-domain but disjoint from train.

Usage:
    python -m scripts.curate_v4_train \\
        --in-dir test_sets \\
        --train-out test_sets/train_v4.jsonl \\
        --val-out   test_sets/val_v4.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# (source_pattern, dialect_label, target_hours, files_to_merge)
SOURCES = [
    ("cv18",       "msa",       65.0, ["v4_cv18_train.jsonl"]),
    ("masc",       "levantine", 65.0, ["v4_masc_train.jsonl"]),
    ("sada",       "gulf",      65.0, ["v4_sada_train.jsonl", "v4_sada_val_train.jsonl"]),
    ("casablanca", "multi",     None, sorted(Path("test_sets").glob("v4_casablanca_*_train.jsonl")) if False else None),
    ("fleurs",     "msa",       None, ["v4_fleurs_train.jsonl"]),
]

VAL_PER_SOURCE = 200    # held out before train sampling
SEED = 42


def hours_of(rows):
    return sum(r.get("duration_s", 0) for r in rows) / 3600.0


def cap_to_hours(rows, max_hours):
    """Sample rows in order until hour budget is hit."""
    if max_hours is None:
        return rows
    out = []
    total = 0.0
    for r in rows:
        d = r.get("duration_s", 0)
        if total + d > max_hours * 3600.0:
            continue
        out.append(r)
        total += d
        if total >= max_hours * 3600.0:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=Path("test_sets"))
    p.add_argument("--train-out", type=Path, default=Path("test_sets/train_v4.jsonl"))
    p.add_argument("--val-out", type=Path, default=Path("test_sets/val_v4.jsonl"))
    args = p.parse_args()

    rng = random.Random(SEED)

    # Collect Casablanca files dynamically
    casa_files = sorted(args.in_dir.glob("v4_casablanca_*_train.jsonl"))

    plan = [
        ("cv18",       "msa",       65.0, [args.in_dir / "v4_cv18_train.jsonl"]),
        ("masc",       "levantine", 65.0, [args.in_dir / "v4_masc_train.jsonl"]),
        ("sada",       "gulf",      65.0,
         [args.in_dir / "v4_sada_train.jsonl",
          args.in_dir / "v4_sada_val_train.jsonl"]),
        ("casablanca", "multi",     None, casa_files),
        ("fleurs",     "msa",       None, [args.in_dir / "v4_fleurs_train.jsonl"]),
    ]

    train_rows = []
    val_rows = []

    print(f"{'source':12s}{'rows_avail':>12s}{'hr_avail':>10s}{'val':>5s}"
          f"{'rows_train':>12s}{'hr_train':>10s}")
    print("-" * 65)
    for tag, dialect, cap_h, files in plan:
        avail = []
        for f in files:
            if not f.exists():
                continue
            for line in f.open():
                r = json.loads(line)
                avail.append(r)
        if not avail:
            print(f"{tag:12s}{'(none)':>12s}{'':>10s}{'':>5s}{'':>12s}{'':>10s}")
            continue
        rng.shuffle(avail)
        n_avail = len(avail)
        h_avail = hours_of(avail)

        # Carve val from the head, then cap from rest
        n_val = min(VAL_PER_SOURCE, len(avail) // 10)
        val_part = avail[:n_val]
        train_part_full = avail[n_val:]

        capped = cap_to_hours(train_part_full, cap_h)
        h_train = hours_of(capped)

        for r in val_part:
            r["dialect"] = r.get("dialect", dialect)
            r["source_dataset"] = tag
        for r in capped:
            r["dialect"] = r.get("dialect", dialect)
            r["source_dataset"] = tag

        train_rows.extend(capped)
        val_rows.extend(val_part)

        print(f"{tag:12s}{n_avail:>12d}{h_avail:>9.1f}h{n_val:>5d}"
              f"{len(capped):>12d}{h_train:>9.1f}h")

    # Final shuffle
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    with args.train_out.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with args.val_out.open("w") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== v4 curated mix written ===")
    print(f"  train: {args.train_out}  ({len(train_rows)} rows, {hours_of(train_rows):.1f}h)")
    print(f"  val:   {args.val_out}    ({len(val_rows)} rows, {hours_of(val_rows):.1f}h)")


if __name__ == "__main__":
    main()
