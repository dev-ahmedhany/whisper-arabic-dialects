"""v4 training mix: ~74h human-labeled Arabic, no synth, no pseudo.

Builds on v3 (~38 h Casablanca + MGB-3 + Clean-MGB-3 + MASC + CV18 capped 15 h)
by:
  - Uncapping Common Voice 18 ar (+13h MSA, ~28 h total)
  - Adding Tarteel everyayah at cap 10 h (MSA Quran register)
  - Adding ArVoice human portion ~10 h (MSA, real voices only)
  - Adding halabi2016 ~3 h (Levantine, Buckwalter → Arabic converted)

Per-dialect caps designed to keep MSA at ~38% to mitigate the v3 MSA drift
documented in paper §6.x (catastrophic-forgetting on MSA when the dialect
share dominates training).

Usage on the L4 (after running prepare_tarteel.py / prepare_arvoice.py /
prepare_halabi.py to populate the new test_sets JSONLs):

    python -m scripts.build_v4_mix
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path("/home/ahmedhany/whisper-arabic-dialects")
SETS = ROOT / "test_sets"


def _filter(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        ref = (r.get("reference") or "").strip()
        if not ref or len(ref.split()) < 2:
            continue
        if any(c in ref for c in "[]()"):
            continue
        d = r.get("duration_s", 0.0)
        if d and (d < 1.0 or d > 30.0):
            continue
        out.append(r)
    return out


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _filter([json.loads(l) for l in path.open() if l.strip()])


def _hours(rows: list[dict]) -> float:
    return sum(r.get("duration_s", 0) for r in rows) / 3600.0


def _cap_h(rows: list[dict], max_h: float) -> list[dict]:
    s, out = 0.0, []
    for r in rows:
        d = r.get("duration_s", 0.0)
        if s + d > max_h * 3600:
            break
        out.append(r)
        s += d
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-train", type=Path, default=Path("test_sets/train_v4.jsonl"))
    p.add_argument("--out-val", type=Path, default=Path("test_sets/val_v4.jsonl"))
    p.add_argument("--max-msa", type=float, default=38.0,
                   help="cap MSA hours (CV uncapped + Tarteel + ArVoice combined)")
    p.add_argument("--max-egyptian", type=float, default=25.0)
    p.add_argument("--max-levantine", type=float, default=22.0)
    p.add_argument("--max-gulf", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    # Same v3 sources, plus new MSA boosters
    msa_pool = (
        _load(SETS / "common_voice_18_ar_train.jsonl")     # uncapped
        + _load(SETS / "tarteel_msa_train.jsonl")           # Quran register
        + _load(SETS / "arvoice_msa_human_train.jsonl")     # real voices only
    )
    rng.shuffle(msa_pool)

    eg_pool = (
        _load(SETS / "casablanca_egypt_train.jsonl")
        + _load(SETS / "mgb3_egyptian_train.jsonl")
        + _load(SETS / "egyptian_clean_mgb3_train.jsonl")
    )
    rng.shuffle(eg_pool)

    lv_pool = (
        _load(SETS / "casablanca_jordan_train.jsonl")
        + _load(SETS / "casablanca_palestine_train.jsonl")
        + _load(SETS / "masc_levantine_train.jsonl")
        + _load(SETS / "halabi_levantine_train.jsonl")
    )
    rng.shuffle(lv_pool)

    gu_pool = (
        _load(SETS / "casablanca_uae_train.jsonl")
        + _load(SETS / "casablanca_yemen_train.jsonl")
    )
    rng.shuffle(gu_pool)

    pools = {
        "msa": _cap_h(msa_pool, args.max_msa),
        "egyptian": _cap_h(eg_pool, args.max_egyptian),
        "levantine": _cap_h(lv_pool, args.max_levantine),
        "gulf": _cap_h(gu_pool, args.max_gulf),
    }

    train, val, dist = [], [], {}
    for dialect, rows in pools.items():
        n_val = max(30, len(rows) // 30)
        for r in rows[:n_val]:
            val.append({**r, "dialect": dialect})
        for r in rows[n_val:]:
            train.append({**r, "dialect": dialect})
        dist[dialect] = {
            "train": len(rows) - n_val, "val": n_val,
            "hours": round(_hours(rows), 2),
        }
    rng.shuffle(train)
    rng.shuffle(val)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with args.out_train.open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with args.out_val.open("w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_h = sum(d["hours"] for d in dist.values())
    print(f"\n=== v4 mix ===")
    print(f"  train: {args.out_train} ({len(train)} rows, {_hours(train):.2f}h)")
    print(f"  val:   {args.out_val}   ({len(val)} rows,   {_hours(val):.2f}h)")
    print(f"  total: {total_h:.2f}h")
    for d, s in dist.items():
        pct = (s["hours"] / total_h * 100) if total_h else 0
        print(f"    {d:10s} {s['hours']:5.2f}h ({pct:.1f}%)  train={s['train']}, val={s['val']}")


if __name__ == "__main__":
    main()
