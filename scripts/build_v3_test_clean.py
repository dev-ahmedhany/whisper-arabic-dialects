"""Build CONTAMINATION-FREE v3 test sets.

The original `test_v3_<dialect>_mixed.jsonl` files contained 50 broadcast
rows sampled from the SAME MGB-3-train / MASC-train pools that the v3
training mix drew from — 96 of 728 utterances were memorized by the model.
Headline numbers were inflated by ~2 pp avg-4 (see deploy/06 §8 for the
postmortem).

This script builds replacements that are clean by construction:
  - Egyptian:  100 rows from Casablanca Egypt **test** split
  - Levantine: 100 rows from Casablanca Jordan **test** split
  - Gulf:      100 rows from Casablanca UAE **test** split
  - MSA:       (unchanged) FLEURS **test** split — already clean

Casablanca is `validation`+`test`; we always train on `validation` and only
ever test on `test`, so the splits don't overlap by construction. The script
asserts no audio path appears in any provided training JSONL before writing.

Usage on L4 (after deploy/06 §1 setup):
    python -m scripts.build_v3_test_clean \\
        --casa-egyptian  test_sets/casablanca_egyptian_test.jsonl \\
        --casa-levantine test_sets/casablanca_levantine_test.jsonl \\
        --casa-gulf      test_sets/casablanca_gulf_test.jsonl \\
        --train-pool     test_sets/train_v3.jsonl test_sets/train.jsonl \\
        --out-dir        test_sets/ \\
        --n-per-dialect  100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def basenames_of(rows: list[dict]) -> set[str]:
    return {r["audio"].rsplit("/", 1)[-1] for r in rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--casa-egyptian", type=Path, required=True)
    p.add_argument("--casa-levantine", type=Path, required=True)
    p.add_argument("--casa-gulf", type=Path, required=True)
    p.add_argument("--train-pool", type=Path, nargs="+", default=[],
                   help="JSONLs to assert no overlap against")
    p.add_argument("--out-dir", type=Path, default=Path("test_sets"))
    p.add_argument("--n-per-dialect", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_basenames: set[str] = set()
    for tp in args.train_pool:
        if not tp.exists():
            print(f"[warn] train pool {tp} not found — overlap check will be partial")
            continue
        bn = basenames_of(load_jsonl(tp))
        print(f"[pool] {tp.name}: {len(bn)} basenames")
        train_basenames |= bn
    print(f"[pool] total unique train basenames: {len(train_basenames)}")

    rng = random.Random(args.seed)

    sources = [
        ("egyptian",  args.casa_egyptian),
        ("levantine", args.casa_levantine),
        ("gulf",      args.casa_gulf),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dialect, src_path in sources:
        rows = load_jsonl(src_path)
        rng.shuffle(rows)
        # filter rows already in train pool (defensive — Casablanca test
        # vs train should never overlap, but the assertion is cheap)
        clean = [r for r in rows if r["audio"].rsplit("/", 1)[-1] not in train_basenames]
        skipped = len(rows) - len(clean)
        if skipped:
            print(f"[{dialect}] skipped {skipped} rows that overlap with train pool")
        sample = clean[: args.n_per_dialect]
        if len(sample) < args.n_per_dialect:
            print(f"[warn] {dialect}: only {len(sample)} clean rows available "
                  f"(asked for {args.n_per_dialect})")
        # tag each row with explicit clean=true for downstream filtering
        for r in sample:
            r["dialect"] = dialect
            r["source_dataset"] = "UBC-NLP/Casablanca"
            r["test_source"] = "casablanca_test_clean"
        out_path = args.out_dir / f"test_v3_clean_{dialect}.jsonl"
        with out_path.open("w") as f:
            for r in sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[write] {out_path}: {len(sample)} rows")


if __name__ == "__main__":
    main()
