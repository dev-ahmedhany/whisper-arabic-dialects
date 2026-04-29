"""Prepare Casablanca multi-dialect Arabic test sets → per-dialect JSONL files.

Casablanca [Talafha et al., 2024] is hosted at `UBC-NLP/Casablanca` (NOT
`MBZUAI-Paris/...` — the original assumption was wrong). The dataset uses one
HF config per country, with these countries available:

  Algeria   Egypt   Jordan   Mauritania   Morocco   Palestine   UAE   Yemen

We map a representative subset of these to our 5-dialect schema. Default uses
one country per dialect for clean, non-overlapping per-dialect WER:

  egyptian  <- Egypt
  maghrebi  <- Morocco       (Algeria + Mauritania are alternative Maghrebi)
  levantine <- Jordan        (Palestine is alternative Levantine)
  gulf      <- UAE           (Yemen is southern but often grouped with Gulf)

Casablanca has NO MSA section — MSA test data comes from FLEURS only.

Schema per row in the upstream dataset: { audio, seg_id, transcription, gender, duration }.
License: CC-BY-NC-ND-4.0 (research/non-commercial only). Held-out evaluation;
NEVER mixed into training (per data_prep.py contract).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts._hf_audio_to_jsonl import stream_to_jsonl


# country_config -> (dialect_tag, output_filename_dialect)
DEFAULT_COUNTRY_TO_DIALECT: dict[str, str] = {
    "Egypt": "egyptian",
    "Morocco": "maghrebi",
    "Jordan": "levantine",
    "UAE": "gulf",
}

ALL_COUNTRIES = [
    "Algeria", "Egypt", "Jordan", "Mauritania",
    "Morocco", "Palestine", "UAE", "Yemen",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-id", default="UBC-NLP/Casablanca")
    p.add_argument("--split", default="test", choices=["test", "validation"])
    p.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRY_TO_DIALECT.keys()),
                   choices=ALL_COUNTRIES,
                   help="which country configs to pull; default is one per dialect")
    p.add_argument("--out-dir", type=Path, default=Path("test_sets"))
    p.add_argument("--audio-dir", type=Path, default=Path("audio/casablanca"))
    p.add_argument("--max-per-country", type=int, default=500)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for country in args.countries:
        dialect = DEFAULT_COUNTRY_TO_DIALECT.get(country)
        if dialect is None:
            # Country chosen but no dialect mapping; tag with the lowercased country name
            dialect = country.lower()
        out_path = args.out_dir / f"casablanca_{dialect}_test.jsonl"
        country_audio_dir = args.audio_dir / country
        print(f"[casablanca] config={country} -> dialect={dialect} -> {out_path}")
        n = stream_to_jsonl(
            dataset_id=args.dataset_id,
            name=country,
            split=args.split,
            output_jsonl=out_path,
            audio_dir=country_audio_dir,
            text_fn=lambda r: r.get("transcription") or r.get("text") or r.get("sentence", ""),
            dialect=dialect,
            max_samples=args.max_per_country,
            trust_remote_code=True,
        )
        counts[country] = n

    print()
    print("=== Casablanca per-country row counts ===")
    for country, n in counts.items():
        print(f"  {country:12s}  {n:4d} rows  ->  {DEFAULT_COUNTRY_TO_DIALECT.get(country, country.lower())}")


if __name__ == "__main__":
    main()
