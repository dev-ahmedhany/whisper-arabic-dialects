"""v3 data preparation: human-labeled public Arabic ASR datasets only.

Aggregates per-dataset JSONL files (built by `scripts/prepare_*.py`) into
`train_v3.jsonl` + `val_v3.jsonl` + `mixed_test_v3.jsonl`, with explicit
dialect balance and source diversity.

Key v3 design choices vs v1/v2:
  - Human labels only (no pseudo-labels — see paper §6.2 future-work note)
  - Multi-source per dialect (broadcast + conversational + curated read)
  - Maghrebi excluded (Whisper pretraining gap, see §3 / §12)
  - Balanced 50/50 between MSA and dialects on the train side
  - Mixed test set: broadcast + Casablanca + Common Voice samples per dialect
  - Per-dataset dialect tags preserved for downstream stratified eval

Usage:
    python -m src.data_v3_prep --out-dir test_sets/v3 --max-hours-per-dialect 25

Inputs expected (run scripts/prepare_*.py first to populate):
    test_sets/common_voice_18_ar_train.jsonl
    test_sets/fleurs_msa_test.jsonl
    test_sets/masc_levantine_train.jsonl
    test_sets/mgb3_egyptian_train.jsonl
    test_sets/casablanca_<dialect>_train.jsonl     (Casablanca validation split)
    test_sets/casablanca_<dialect>_test.jsonl
    test_sets/test_<dialect>_casablanca_<dialect>_test.jsonl
    test_sets/arzen_egyptian_train.jsonl           (TODO: prepare_arzen.py)

Outputs:
    test_sets/v3/train_v3.jsonl
    test_sets/v3/val_v3.jsonl
    test_sets/v3/mixed_test_v3.jsonl
    test_sets/v3/manifest.json   (per-source row counts + hours)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DROP_DIALECTS = {"maghrebi"}  # excluded — see paper §3.7

# Per-source dialect tag override. Some sources mix dialects internally (Common
# Voice Arabic is mostly MSA but has dialect-tagged rows we honor); others are
# single-dialect curated and we override with the source name.
SOURCE_DIALECT_OVERRIDE = {
    # source_filename: forced dialect | None (use row's own field)
    "fleurs_msa_test.jsonl": "msa",
    "masc_levantine_train.jsonl": "levantine",
    "mgb3_egyptian_train.jsonl": "egyptian",
    "casablanca_egyptian_train.jsonl": "egyptian",
    "casablanca_levantine_train.jsonl": "levantine",
    "casablanca_gulf_train.jsonl": "gulf",
    "arzen_egyptian_train.jsonl": "egyptian",
    "egyptian_clean_mgb3_train.jsonl": "egyptian",
}

# Default mapping of source bucket → genre (broadcast vs conversational). Used
# for the mixed_test_v3.jsonl construction so the test set isn't accidentally
# all broadcast or all street.
SOURCE_GENRE = {
    "fleurs": "broadcast",
    "masc": "broadcast",
    "mgb3": "broadcast",
    "common_voice": "read",
    "casablanca": "mixed-conversational",
    "arzen": "conversational",
    "egyptian_clean_mgb3": "broadcast-clean",
}


def _row_dialect(row: dict, source: str) -> str:
    """Pick the right dialect tag for a row given its source file."""
    forced = SOURCE_DIALECT_OVERRIDE.get(source)
    if forced is not None:
        return forced
    return (row.get("dialect") or "").lower() or "unknown"


def _source_genre(source: str) -> str:
    for prefix, genre in SOURCE_GENRE.items():
        if source.startswith(prefix) or prefix in source:
            return genre
    return "unknown"


def _load_source(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.open():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _hours(rows: list[dict]) -> float:
    return sum(r.get("duration_s", 0.0) for r in rows) / 3600.0


def build_v3(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = Path("test_sets")

    # Discover every per-source JSONL we want to consider for v3 training
    candidate_sources = [
        "common_voice_18_ar_train.jsonl",
        "masc_levantine_train.jsonl",
        "mgb3_egyptian_train.jsonl",
        "egyptian_clean_mgb3_train.jsonl",
        "casablanca_egyptian_train.jsonl",
        "casablanca_levantine_train.jsonl",
        "casablanca_gulf_train.jsonl",
        "arzen_egyptian_train.jsonl",
    ]

    # Per-dialect bucket: list of (source_name, row, genre)
    dialect_pool: dict[str, list[tuple[str, dict, str]]] = defaultdict(list)
    manifest: dict[str, dict] = {}

    for src in candidate_sources:
        rows = _load_source(sources_dir / src)
        if not rows:
            print(f"[skip] {src} — not found or empty")
            continue
        genre = _source_genre(src)
        kept = 0
        for r in rows:
            dialect = _row_dialect(r, src)
            if dialect in DROP_DIALECTS:
                continue
            if not r.get("audio") or not r.get("reference"):
                continue
            dialect_pool[dialect].append((src, r, genre))
            kept += 1
        manifest[src] = {
            "raw_rows": len(rows),
            "kept_rows": kept,
            "hours": _hours(rows),
            "genre": genre,
        }
        print(f"[ok ] {src}: kept {kept}/{len(rows)} rows ({_hours(rows):.1f} h, genre={genre})")

    # Cap per-dialect at args.max_hours_per_dialect
    train_rows: list[str] = []
    val_rows: list[str] = []
    train_dist: dict[str, int] = defaultdict(int)
    val_dist: dict[str, int] = defaultdict(int)
    for dialect, items in dialect_pool.items():
        rng.shuffle(items)
        # Cap by hours
        accumulated_s = 0.0
        chosen: list[tuple[str, dict, str]] = []
        for src, r, genre in items:
            d = r.get("duration_s", 0.0)
            if accumulated_s + d > args.max_hours_per_dialect * 3600:
                break
            chosen.append((src, r, genre))
            accumulated_s += d
        # 95/5 train/val split per dialect
        n_val = max(20, len(chosen) // 20)
        for src, r, genre in chosen[:n_val]:
            r2 = {**r, "dialect": dialect, "source_dataset": src.replace(".jsonl", "")}
            val_rows.append(json.dumps(r2, ensure_ascii=False))
            val_dist[dialect] += 1
        for src, r, genre in chosen[n_val:]:
            r2 = {**r, "dialect": dialect, "source_dataset": src.replace(".jsonl", "")}
            train_rows.append(json.dumps(r2, ensure_ascii=False))
            train_dist[dialect] += 1
        print(f"[mix] {dialect}: chose {len(chosen)} rows ({accumulated_s/3600:.1f} h) → train={len(chosen)-n_val}, val={n_val}")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    train_out = args.out_dir / "train_v3.jsonl"
    val_out = args.out_dir / "val_v3.jsonl"
    train_out.write_text("\n".join(train_rows) + "\n")
    val_out.write_text("\n".join(val_rows) + "\n")

    print(f"\n=== v3 mix written ===")
    print(f"  {train_out}: {len(train_rows)} rows  ({sum(json.loads(l).get('duration_s',0) for l in train_rows)/3600:.1f} h)")
    print(f"  {val_out}:   {len(val_rows)} rows  ({sum(json.loads(l).get('duration_s',0) for l in val_rows)/3600:.1f} h)")
    print(f"  train distribution: {dict(train_dist)}")
    print(f"  val   distribution: {dict(val_dist)}")

    # Mixed test set: 100 samples per dialect from multiple sources and genres.
    # Pull from existing test sets (NOT used in train) — Casablanca test split,
    # FLEURS MSA test, plus a handful of held-out broadcast samples.
    mixed_test_pool: dict[str, list[dict]] = defaultdict(list)
    test_sources = {
        "msa":       ["fleurs_msa_test.jsonl"],
        "egyptian":  ["test_egyptian_casablanca_egyptian_test.jsonl"],
        "levantine": ["test_levantine_casablanca_levantine_test.jsonl"],
        "gulf":      ["test_gulf_casablanca_gulf_test.jsonl"],
    }
    for dialect, src_list in test_sources.items():
        for src in src_list:
            rows = _load_source(sources_dir / src)
            for r in rows:
                # Filter junk per the same logic in test_sets/filtered
                ref = r.get("reference", "")
                if any(c in ref for c in "[]()"):
                    continue
                if len(ref.split()) < 3:
                    continue
                d = r.get("duration_s", 0.0)
                if d and (d < 2.0 or d > 30.0):
                    continue
                mixed_test_pool[dialect].append({**r, "source_dataset": src.replace(".jsonl", "")})

    # Sample 100 per dialect (or fewer if available)
    mixed_test = []
    test_dist = {}
    for dialect in ["msa", "egyptian", "levantine", "gulf"]:
        pool = mixed_test_pool.get(dialect, [])
        rng.shuffle(pool)
        chosen = pool[:args.test_samples_per_dialect]
        mixed_test.extend(json.dumps(r, ensure_ascii=False) for r in chosen)
        test_dist[dialect] = len(chosen)
        print(f"[test] {dialect}: chose {len(chosen)}/{len(pool)} samples")

    test_out = args.out_dir / "mixed_test_v3.jsonl"
    test_out.write_text("\n".join(mixed_test) + "\n")
    print(f"\n  {test_out}: {len(mixed_test)} rows ({test_dist})")

    # Manifest
    manifest_obj = {
        "v3_train_rows": len(train_rows),
        "v3_val_rows": len(val_rows),
        "v3_test_rows": len(mixed_test),
        "max_hours_per_dialect": args.max_hours_per_dialect,
        "drop_dialects": sorted(DROP_DIALECTS),
        "train_distribution": dict(train_dist),
        "val_distribution": dict(val_dist),
        "test_distribution": test_dist,
        "sources": manifest,
        "seed": args.seed,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest_obj, indent=2, ensure_ascii=False))
    print(f"  {args.out_dir/'manifest.json'} written")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("test_sets/v3"))
    p.add_argument("--max-hours-per-dialect", type=float, default=25.0)
    p.add_argument("--test-samples-per-dialect", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    build_v3(args)


if __name__ == "__main__":
    main()
