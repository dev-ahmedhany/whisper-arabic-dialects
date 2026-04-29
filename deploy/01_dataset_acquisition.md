# 01 — Dataset Acquisition

All datasets used in this study are accessible via HuggingFace Hub — no external registrations required.

Recommended host: a beefy disk (~500 GB) on the same project where you'll do training. Datasets often re-decode; you don't want to re-download to GCP from Hetzner.

## Prerequisites

```bash
huggingface-cli login   # paste your HF_TOKEN
pip install -r requirements.txt -e .
```

## Datasets at a glance

| Dataset | HF repo | Hours used | Dialect tag | Script |
|---|---|---|---|---|
| Common Voice 17 (ar) | `mozilla-foundation/common_voice_17_0` | ~12 train | msa | `prepare_common_voice.py` |
| FLEURS (ar_eg) | `google/fleurs` | ~2 test | msa | `prepare_fleurs.py` |
| Casablanca | `MBZUAI-Paris/Casablanca` | ~2 test (per dialect) | all 5 | `prepare_casablanca.py` |
| MASC | `pain/MASC` | ~9 train | levantine | `prepare_masc.py` |
| SADA | `MBZUAI/sada` | ~9 train | gulf | `prepare_sada.py` |
| MGB-3 | `ArabicSpeech/MGB-3` | ~10 train | egyptian | `prepare_mgb3.py` |
| MGB-5 | `ArabicSpeech/MGB-5` | ~10 train | maghrebi | `prepare_mgb5.py` |

For Common Voice you may need to click "Agree and access" on the dataset page once before the script can pull. The rest are open access.

## Step 1 — Pull every dataset

These can run in parallel terminals; they don't share state.

```bash
# MSA + FLEURS test
python -m scripts.prepare_common_voice --split train \
  --out test_sets/common_voice_17_ar_train.jsonl \
  --audio-dir audio/common_voice_17_ar

python -m scripts.prepare_fleurs --split test \
  --out test_sets/fleurs_msa_test.jsonl \
  --audio-dir audio/fleurs_ar

# Casablanca multi-dialect (writes one JSONL per dialect)
python -m scripts.prepare_casablanca --split test \
  --out-dir test_sets --audio-dir audio/casablanca \
  --max-per-dialect 500

# Levantine + Gulf training
python -m scripts.prepare_masc --split train \
  --out test_sets/masc_levantine_train.jsonl \
  --audio-dir audio/masc

python -m scripts.prepare_sada --split train \
  --out test_sets/sada_saudi_train.jsonl \
  --audio-dir audio/sada

# Egyptian + Maghrebi training (HF Hub mirrors of MGB-3 / MGB-5)
python -m scripts.prepare_mgb3 --split train \
  --out test_sets/mgb3_egyptian_train.jsonl \
  --audio-dir audio/mgb3

python -m scripts.prepare_mgb5 --split train \
  --out test_sets/mgb5_moroccan_train.jsonl \
  --audio-dir audio/mgb5
```

Use `--max-samples` on any of the above to cap the per-dataset footprint while iterating. Once you trust the pipeline, drop the cap for the full run.

## Step 2 — Build the dialect-balanced splits

Once the per-dataset JSONLs exist:

```bash
python -m src.data_prep \
  --config configs/dataset_mix.yaml \
  --output-dir test_sets
```

Outputs:
```
test_sets/train.jsonl                ← shuffled, dialect-balanced (~50h)
test_sets/val.jsonl                  ← 5% holdout, no overlap with train
test_sets/test_<dialect>_<src>.jsonl ← copied through, never seen at train time
test_sets/split_summary.json         ← hours per dialect, totals
```

Sanity-check `split_summary.json`: hours per dialect should be within ~2× of each other.

## Step 3 — Push to GCS for the training instance

```bash
gsutil mb -l us-central1 gs://dev-ahmedhany-whisper-arabic
gsutil -m cp -r test_sets/ gs://dev-ahmedhany-whisper-arabic/test_sets/
gsutil -m cp -r audio/      gs://dev-ahmedhany-whisper-arabic/audio/
```

The GCP training instance will pull from this bucket on startup.

## Storage cost note

GCS Standard storage in `us-central1` is roughly $0.020/GB/month. ~250 GB of decoded audio = ~$5/month while you iterate. Move to Coldline (`gsutil rewrite -s coldline`) once training is done if you want long-term retention.

## License notes

| Dataset | License | Commercial use |
|---|---|---|
| Common Voice 17 | CC0 | Yes |
| FLEURS | CC-BY-4.0 | Yes (with attribution) |
| Casablanca | Per upstream dataset card | Check upstream |
| MASC | Per dataset card | Check upstream |
| SADA | Per dataset card; terms apply | Check upstream |
| MGB-3 | Research use (per ArabicSpeech) | No |
| MGB-5 | Research use (per ArabicSpeech) | No |

The model produced by training on this mix is fine for the paper and for non-commercial release on HF Hub. For commercial deployment, retrain on a subset that excludes the research-only sources (MGB-3, MGB-5) — or seek an explicit license from ArabicSpeech.
