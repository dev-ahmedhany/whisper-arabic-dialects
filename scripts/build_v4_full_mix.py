"""Build the v4 FULL training mix: ~2,610 hours of Arabic across 6 datasets.

Successor to scripts/build_v4_mix.py (which targeted the older ~74h plan).
This builder targets the ambitious "match-NVIDIA-and-beat-it" plan from
deploy/07: 3.4× NVIDIA's 760h training data, with explicit holdout of every
public Arabic ASR test set.

Train sources (train splits only — test splits held out for evaluation):
  - QCRI/mgb2                       train  ~1,190h   broadcast MSA (Aljazeera)
  - pain/MASC                       train  ~690h     YouTube, Levantine-heavy
  - m6011/sada2022                  train  ~625h     Saudi/Najdi/Hijazi/Gulf
  - fsicoli/common_voice_18_0       train  ~65h      multi-dialect MSA
  - google/fleurs (ar_eg)           train  ~4h       clean MSA
  - UBC-NLP/Casablanca              validation       multi-dialect conversational
        (Egypt, Jordan, Palestine, UAE, Yemen — Maghrebi excluded per memory)

Held-out evaluation set (matches Wang et al. 2024 leaderboard test cells):
  - mgb2_test, masc_clean_test, masc_noisy_test, sada_test,
    cv18_test, fleurs_test, casablanca_test_<country>

Audio is materialized to local 16 kHz mono WAVs under audio/<dataset>/.
JSONL rows: {audio: <abspath>, reference: <text>, dialect: <str>,
source_dataset: <str>, duration_s: <float>, split: <"train"|"test">, ...}

Disk: ~300 GB raw WAV + ~150 GB HF cache + 50 GB other = ~500 GB peak.
Wall-clock: ~6-10 hours, network-bound for first download.

Usage on the v4 box:
    python -m scripts.build_v4_full_mix --out-dir test_sets/ --audio-root audio/

Resumable: skips datasets whose output JSONL already exists with content.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


def write_jsonl(
    dataset_id: str,
    split: str,
    name: str | None,
    output_jsonl: Path,
    audio_dir: Path,
    text_field: str | tuple[str, ...],
    dialect: str | None,
    source_tag: str,
    *,
    trust_remote_code: bool = False,
    filter_field: str | None = None,
    filter_value=None,
    streaming: bool = True,
    max_rows: int | None = None,
    extra_meta_keys: tuple[str, ...] = (),
    slice_start_field: str | None = None,
    slice_end_field: str | None = None,
) -> int:
    """Stream a HF dataset → JSONL with audio materialized to local 16 kHz WAVs.

    Resumable: returns existing-row-count if output already non-empty.
    """
    if output_jsonl.exists() and output_jsonl.stat().st_size > 0:
        n = sum(1 for _ in output_jsonl.open())
        print(f"[skip] {output_jsonl.name} already exists ({n} rows)")
        return n

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {dataset_id} name={name} split={split} streaming={streaming}", flush=True)
    kw = dict(split=split, streaming=streaming)
    if name:
        kw["name"] = name
    if trust_remote_code:
        kw["trust_remote_code"] = True
    ds = load_dataset(dataset_id, **kw)
    if not streaming:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    text_fields = (text_field,) if isinstance(text_field, str) else tuple(text_field)
    n_written = 0
    n_skipped = 0
    out = output_jsonl.open("w")
    pbar = tqdm(ds, desc=f"{source_tag}/{split}")
    try:
        for i, row in enumerate(pbar):
            if max_rows and n_written >= max_rows:
                break
            if filter_field is not None and row.get(filter_field) != filter_value:
                continue

            ref = ""
            for tf in text_fields:
                v = row.get(tf)
                if v:
                    ref = v.strip()
                    break
            if not ref or len(ref.split()) < 2:
                n_skipped += 1
                continue

            audio_path = audio_dir / f"{source_tag}_{split}_{i:08d}.wav"
            if not audio_path.exists():
                a = row.get("audio")
                if not a or "array" not in a:
                    n_skipped += 1
                    continue
                try:
                    arr = a["array"]
                    sr = a.get("sampling_rate", 16000)
                    if sr != 16000:
                        import librosa
                        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                    # SADA-style: full show audio; slice by SegmentStart/End
                    if slice_start_field and slice_end_field:
                        s = float(row.get(slice_start_field, 0.0))
                        e = float(row.get(slice_end_field, 0.0))
                        if e > s:
                            i0, i1 = int(s * 16000), int(e * 16000)
                            arr = arr[i0:i1]
                    sf.write(str(audio_path), arr, 16000, subtype="PCM_16")
                except Exception as e:
                    print(f"  [audio fail @{i}]: {e}", flush=True)
                    n_skipped += 1
                    continue

            duration = audio_path.stat().st_size / (2 * 16000)
            entry = {
                "audio": str(audio_path.resolve()),
                "reference": ref,
                "dialect": dialect or row.get("dialect", "unknown"),
                "source_dataset": source_tag,
                "duration_s": duration,
                "split": split,
            }
            for k in extra_meta_keys:
                if k in row:
                    entry[k] = row[k]
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 500 == 0:
                pbar.set_postfix({"written": n_written, "skipped": n_skipped})
    finally:
        out.close()

    print(f"[done] {output_jsonl.name}: {n_written} rows ({n_skipped} skipped)", flush=True)
    return n_written


# Per-dataset specs.
# NOTE: MGB-2 (QCRI/mgb2, ~1,190h, the largest single Arabic ASR resource)
# is intentionally excluded from v4. The dataset ships as 118GB train.tar.gz
# + dev/test tar folders that don't fit HF datasets' WebDataset loader, and
# extraction + transcription parsing (.stm-style) is non-trivial. v5 follow-up
# can add it via a custom downloader; v4 ships at 1,451h without it.
DATASET_SPECS = [
    # MGB-2 placeholder — disabled in v4. Keep here so re-enabling is a 1-line edit.
    # dict(spec_name="mgb2", ...),
    dict(
        spec_name="masc",
        train=[dict(id="pain/MASC", split="train", text="text", dialect="multi",
                    trust_remote_code=True, extra_meta=("type",))],
        test=[
            dict(id="pain/MASC", split="test", text="text", dialect="multi",
                 trust_remote_code=True, filter_field="type", filter_value="c",
                 extra_meta=("type",), suffix="clean"),
            dict(id="pain/MASC", split="test", text="text", dialect="multi",
                 trust_remote_code=True, filter_field="type", filter_value="n",
                 extra_meta=("type",), suffix="noisy"),
        ],
        audio_subdir="masc",
    ),
    dict(
        spec_name="sada",
        # MohamedRashad/SADA22 is the SADA mirror with proper segment-level
        # splits (train=242K rows ~647h, validation=5.14K ~10h, test=6.19K ~10h)
        # and pre-cleaned columns. The original m6011/sada2022 only had 6.19K
        # show-grouped rows that needed manual segment slicing.
        train=[
            dict(id="MohamedRashad/SADA22", split="train",
                 text=("cleaned_text", "text"),
                 dialect="gulf", trust_remote_code=False,
                 extra_meta=("speaker_dialect", "speaker_age", "speaker_gender")),
            dict(id="MohamedRashad/SADA22", split="validation",
                 text=("cleaned_text", "text"),
                 dialect="gulf", trust_remote_code=False,
                 extra_meta=("speaker_dialect", "speaker_age", "speaker_gender"),
                 suffix="val"),
        ],
        test=[dict(id="MohamedRashad/SADA22", split="test",
                   text=("cleaned_text", "text"),
                   dialect="gulf", trust_remote_code=False,
                   extra_meta=("speaker_dialect", "speaker_age", "speaker_gender"))],
        audio_subdir="sada",
    ),
    dict(
        spec_name="cv18",
        train=[dict(id="fsicoli/common_voice_18_0", name="ar", split="train",
                    text="sentence", dialect="msa", trust_remote_code=True)],
        test=[dict(id="fsicoli/common_voice_18_0", name="ar", split="test",
                   text="sentence", dialect="msa", trust_remote_code=True)],
        audio_subdir="cv18_ar",
    ),
    dict(
        spec_name="fleurs",
        train=[dict(id="google/fleurs", name="ar_eg", split="train",
                    text=("transcription", "raw_transcription"), dialect="msa",
                    trust_remote_code=True)],
        test=[dict(id="google/fleurs", name="ar_eg", split="test",
                   text=("transcription", "raw_transcription"), dialect="msa",
                   trust_remote_code=True)],
        audio_subdir="fleurs_ar",
    ),
    dict(
        spec_name="casablanca",
        # No `train` split — use `validation` as train.
        train=[dict(id="UBC-NLP/Casablanca", name=c, split="validation",
                    text="transcription", dialect=d, trust_remote_code=True,
                    suffix=c.lower())
               for c, d in [("Egypt","egyptian"),("Jordan","levantine"),
                            ("Palestine","levantine"),("UAE","gulf"),
                            ("Yemen","gulf")]],
        test=[dict(id="UBC-NLP/Casablanca", name=c, split="test",
                   text="transcription", dialect=d, trust_remote_code=True,
                   suffix=c.lower())
              for c, d in [("Egypt","egyptian"),("Jordan","levantine"),
                           ("Palestine","levantine"),("UAE","gulf"),
                           ("Yemen","gulf")]],
        audio_subdir="casablanca",
    ),
]


def run_spec(s: dict, audio_root: Path, out_dir: Path, only_split: str | None = None) -> None:
    audio_dir = audio_root / s["audio_subdir"]
    for split_kind, items in (("train", s.get("train", [])), ("test", s.get("test", []))):
        if only_split and split_kind != only_split:
            continue
        for item in items:
            suffix = item.get("suffix", "")
            tag = s["spec_name"] + (f"_{suffix}" if suffix else "")
            out_jsonl = out_dir / f"v4_{tag}_{split_kind}.jsonl"
            write_jsonl(
                dataset_id=item["id"],
                name=item.get("name"),
                split=item["split"],
                output_jsonl=out_jsonl,
                audio_dir=audio_dir,
                text_field=item["text"],
                dialect=item.get("dialect"),
                source_tag=tag,
                trust_remote_code=item.get("trust_remote_code", False),
                filter_field=item.get("filter_field"),
                filter_value=item.get("filter_value"),
                extra_meta_keys=item.get("extra_meta", ()),
                slice_start_field=item.get("slice_start_field"),
                slice_end_field=item.get("slice_end_field"),
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("test_sets"))
    p.add_argument("--audio-root", type=Path, default=Path("audio"))
    p.add_argument("--only", default=None,
                   help="comma-separated dataset names to run (default: all)")
    p.add_argument("--only-split", default=None, choices=[None, "train", "test"])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.audio_root.mkdir(parents=True, exist_ok=True)
    only_set = set(args.only.split(",")) if args.only else None

    t0 = time.perf_counter()
    for spec in DATASET_SPECS:
        if only_set and spec["spec_name"] not in only_set:
            continue
        print(f"\n========== {spec['spec_name']} ==========", flush=True)
        run_spec(spec, args.audio_root, args.out_dir, only_split=args.only_split)

    print(f"\nv4 build done in {(time.perf_counter()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
