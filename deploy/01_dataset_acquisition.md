# 01 — Dataset Acquisition

The seven datasets used in this study sit on a spectrum from "free, one HF call" to "register, sign EULA, wait for email, download manually". This runbook gets all of them onto disk in the JSONL schema the rest of the repo expects.

Recommended host: a beefy disk (~500 GB) on the same project where you'll do training. Datasets often re-decode; you don't want to re-download to GCP from Hetzner.

## Prerequisites

```bash
huggingface-cli login   # paste your HF_TOKEN (gets you Common Voice, FLEURS, MASC, SADA, Casablanca)
pip install -r requirements.txt -e .
```

For MGB-3, MGB-5, and QASR you must register externally (see below) and download manually.

## Datasets at a glance

| Dataset | Access | Hours used | Dialect tag | Script |
|---|---|---|---|---|
| Common Voice 17 (ar) | HF Hub, free + login | ~12 train | msa | `prepare_common_voice.py` |
| FLEURS (ar_eg) | HF Hub, free | ~2 test | msa | `prepare_fleurs.py` |
| Casablanca | HF Hub, MBZUAI | ~2 test (per dialect) | all 5 | `prepare_casablanca.py` |
| MASC | HF Hub, free | ~9 train | levantine | `prepare_masc.py` |
| SADA | HF Hub, terms apply | ~9 train | gulf | `prepare_sada.py` |
| MGB-3 | https://arabicspeech.org/mgb3 (registration) | ~10 train | egyptian | `prepare_mgb3.py` |
| MGB-5 | https://arabicspeech.org/mgb5 (registration) | ~10 train | maghrebi | `prepare_mgb5.py` |
| QASR | https://arabicspeech.org/qasr (registration) | shared train | levantine + gulf | `prepare_qasr.py` |

## Step 1 — HF Hub datasets

```bash
# Common Voice 17 Arabic
python -m scripts.prepare_common_voice --split train \
  --out test_sets/common_voice_17_ar_train.jsonl \
  --audio-dir audio/common_voice_17_ar

# FLEURS Arabic test
python -m scripts.prepare_fleurs --split test \
  --out test_sets/fleurs_msa_test.jsonl \
  --audio-dir audio/fleurs_ar

# Casablanca multi-dialect (writes one file per dialect)
python -m scripts.prepare_casablanca --split test \
  --out-dir test_sets --audio-dir audio/casablanca \
  --max-per-dialect 500

# MASC Levantine
python -m scripts.prepare_masc --split train \
  --out test_sets/masc_levantine_train.jsonl \
  --audio-dir audio/masc

# SADA Saudi (Gulf)
python -m scripts.prepare_sada --split train \
  --out test_sets/sada_saudi_train.jsonl \
  --audio-dir audio/sada
```

If a dataset's column names differ from what `_hf_audio_to_jsonl.py` assumes, edit the prepare script's `text_fn`.

## Step 2 — Gated datasets (MGB-3, MGB-5, QASR)

These need a one-time registration. Once you have the tarball:

```bash
# MGB-3 (Egyptian)
mkdir -p audio/mgb3
tar xf MGB3.tar.gz -C audio/mgb3
python -m scripts.prepare_mgb3 \
  --audio-dir audio/mgb3/audio/wav/train \
  --text-file audio/mgb3/text/train/text \
  --out test_sets/mgb3_egyptian_train.jsonl

# MGB-5 (Moroccan)
mkdir -p audio/mgb5
tar xf MGB5.tar.gz -C audio/mgb5
python -m scripts.prepare_mgb5 \
  --audio-dir audio/mgb5/audio/wav/train \
  --text-file audio/mgb5/text/train/text \
  --out test_sets/mgb5_moroccan_train.jsonl

# QASR (filter by dialect via spk2dialect; emits one Levantine and one Gulf file)
python -m scripts.prepare_qasr \
  --root audio/qasr \
  --split train \
  --filter-dialect levantine \
  --out test_sets/qasr_levantine_train.jsonl
python -m scripts.prepare_qasr \
  --root audio/qasr \
  --split train \
  --filter-dialect gulf \
  --out test_sets/qasr_gulf_train.jsonl
```

## Step 3 — Build the dialect-balanced splits

Once all per-dataset JSONLs exist:

```bash
python -m src.data_prep \
  --config configs/dataset_mix.yaml \
  --output-dir test_sets
```

Outputs:
```
test_sets/train.jsonl              ← shuffled, dialect-balanced (~50h)
test_sets/val.jsonl                ← 5% holdout, no overlap with train
test_sets/test_<dialect>_<src>.jsonl  ← copied through, never seen at train time
test_sets/split_summary.json       ← hours per dialect, totals
```

Sanity-check `split_summary.json`: hours per dialect should be within ~2× of each other.

## Step 4 — Push test sets to GCS for the training instance

```bash
gsutil mb -l us-central1 gs://your-project-whisper-arabic-data
gsutil -m cp -r test_sets/ gs://your-project-whisper-arabic-data/test_sets/
gsutil -m cp -r audio/  gs://your-project-whisper-arabic-data/audio/
```

The GCP training instance pulls these on startup.

## Costs

Disk: ~200–300 GB for all audio after decompression. GCS storage: ~$5/month for ~250 GB.

## License notes

- Common Voice 17: CC0.
- FLEURS: CC-BY-4.0.
- Casablanca: per the Casablanca paper's license (check upstream).
- MASC, SADA: per dataset cards on HF.
- MGB-3, MGB-5, QASR: per the MGB challenge / QCRI Hamad terms — research use only, no redistribution.
