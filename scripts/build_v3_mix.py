"""Build the v3 training mix on a fresh L4: Casablanca validation splits
(Egypt/Jordan/UAE) → Egyptian/Levantine/Gulf train, plus capped Common
Voice MSA. Same recipe as the v2 mix that was lost to the deleted L4.

Casablanca's published splits are validation+test only — no train. We
treat 'validation' as our training data and keep the 'test' split as the
held-out eval target. Maghrebi is excluded (paper §3.7 / memory).

Usage on the new L4 (after bootstrap):
    python -m scripts.build_v3_mix \\
        --out-train test_sets/train_v3.jsonl \\
        --out-val   test_sets/val_v3.jsonl \\
        --max-msa 2000 --max-per-dialect 600

Outputs JSONL rows with the same schema as data_prep.py:
    {audio: <abspath>, reference: <text>, dialect, source_dataset, duration_s}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl

CASABLANCA_TRAIN = {
    # country -> dialect (we use validation split as train)
    "Egypt": "egyptian",
    "Jordan": "levantine",
    "UAE": "gulf",
}


def _build_casablanca_train(audio_root: Path, out_dir: Path) -> dict[str, Path]:
    """Pull Casablanca validation split per country → casablanca_<dialect>_train.jsonl."""
    paths: dict[str, Path] = {}
    for country, dialect in CASABLANCA_TRAIN.items():
        out_path = out_dir / f"casablanca_{dialect}_train.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[skip] {out_path} already exists")
            paths[dialect] = out_path
            continue
        print(f"[pull] casablanca {country} -> {dialect} -> {out_path}")
        n = stream_to_jsonl(
            dataset_id="UBC-NLP/Casablanca",
            name=country,
            split="validation",
            output_jsonl=out_path,
            audio_dir=audio_root / "casablanca" / country,
            text_fn=lambda r: r.get("transcription") or r.get("text") or r.get("sentence", ""),
            dialect=dialect,
            trust_remote_code=True,
        )
        print(f"  wrote {n} rows")
        paths[dialect] = out_path
    return paths


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def _filter(rows: list[dict]) -> list[dict]:
    """Drop rows that would hurt train quality."""
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cv-jsonl", default="test_sets/common_voice_18_ar_train.jsonl",
                   help="Common Voice 18 Arabic JSONL (already in GCS)")
    p.add_argument("--audio-root", type=Path, default=Path("audio"))
    p.add_argument("--out-train", type=Path, default=Path("test_sets/train_v3.jsonl"))
    p.add_argument("--out-val", type=Path, default=Path("test_sets/val_v3.jsonl"))
    p.add_argument("--out-dir-casablanca", type=Path, default=Path("test_sets"))
    p.add_argument("--max-msa", type=int, default=2000, help="cap MSA train rows")
    p.add_argument("--max-msa-val", type=int, default=100)
    p.add_argument("--max-per-dialect", type=int, default=600,
                   help="cap each dialect train rows so MSA stays ~50%")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="per-dialect val fraction")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    args.out_dir_casablanca.mkdir(parents=True, exist_ok=True)

    casa_paths = _build_casablanca_train(args.audio_root, args.out_dir_casablanca)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    dist: dict[str, dict] = {}

    cv_path = Path(args.cv_jsonl)
    if cv_path.exists():
        cv_rows = _filter(_load_jsonl(cv_path))
        rng.shuffle(cv_rows)
        for r in cv_rows[:args.max_msa]:
            train_rows.append({**r, "dialect": "msa", "source_dataset": "common_voice_18_ar"})
        for r in cv_rows[args.max_msa:args.max_msa + args.max_msa_val]:
            val_rows.append({**r, "dialect": "msa", "source_dataset": "common_voice_18_ar"})
        dist["msa"] = {"train": min(args.max_msa, len(cv_rows)),
                       "val": min(args.max_msa_val, max(0, len(cv_rows) - args.max_msa))}
        print(f"[mix] msa: {dist['msa']}")
    else:
        print(f"[warn] {cv_path} missing — MSA portion will be empty")

    for dialect, path in casa_paths.items():
        rows = _filter(_load_jsonl(path))
        rng.shuffle(rows)
        capped = rows[:args.max_per_dialect]
        n_val = max(20, int(len(capped) * args.val_frac))
        for r in capped[:n_val]:
            val_rows.append({**r, "dialect": dialect,
                             "source_dataset": f"casablanca_{dialect}"})
        for r in capped[n_val:]:
            train_rows.append({**r, "dialect": dialect,
                               "source_dataset": f"casablanca_{dialect}"})
        dist[dialect] = {"train": len(capped) - n_val, "val": n_val,
                         "available": len(rows), "capped_at": args.max_per_dialect}
        print(f"[mix] {dialect}: {dist[dialect]}")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    with args.out_train.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with args.out_val.open("w") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"=== v3 mix built ===")
    print(f"  train: {args.out_train}  ({len(train_rows)} rows)")
    print(f"  val:   {args.out_val}    ({len(val_rows)} rows)")
    print(f"  per-dialect distribution: {dist}")


if __name__ == "__main__":
    main()
