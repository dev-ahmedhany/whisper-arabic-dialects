# 01 — Dataset Acquisition

All datasets used in this study are accessible via HuggingFace Hub — no external registrations required.

Datasets decompress to ~250 GB of audio plus per-dataset JSONLs. We do this on a small dedicated GCP VM (cheap, throwaway) rather than on a laptop, so the download stays inside Google's network all the way to GCS.

## Prerequisites

- `deploy/00_gcp_bootstrap.md` complete (gcloud authed, project set, billing live, APIs enabled).
- `HF_TOKEN` ready (from <https://huggingface.co/settings/tokens>, scope: read).

## Step 0 — Provision a data-prep VM

```bash
gcloud compute instances create whisper-dataprep \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=500GB \
  --boot-disk-type=pd-balanced \
  --scopes=cloud-platform
```

`e2-standard-4` is ~$0.13/hr — cheapest sensible spec for an I/O-bound prep job. `--scopes=cloud-platform` gives the VM ambient credentials for `gsutil` so you don't need a separate service-account dance to write to your bucket.

SSH in:

```bash
gcloud compute ssh whisper-dataprep --zone=us-central1-a
```

(The first SSH triggers the firewall rule + key creation. ~30s.)

Inside the VM, install Python 3.11, ffmpeg, and the repo:

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip git ffmpeg libsndfile1
git clone https://github.com/dev-ahmedhany/whisper-arabic-dialects.git
cd whisper-arabic-dialects
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv pip install -r requirements.txt -e .
```

Authenticate to HF Hub (the prep scripts need this for Common Voice 17):

```bash
huggingface-cli login   # paste your HF_TOKEN at the prompt
```

## Datasets at a glance

| Dataset | HF repo | Hours used | Dialect tag | Script |
|---|---|---|---|---|
| Common Voice 18 (ar) | `MohamedRashad/common-voice-18-arabic` | ~12 train | msa | `prepare_common_voice.py` |
| FLEURS (ar_eg) | `google/fleurs` | ~2 test | msa | `prepare_fleurs.py` |
| Casablanca | `MBZUAI-Paris/Casablanca` | ~2 test (per dialect) | all 5 | `prepare_casablanca.py` |
| MASC | `pain/MASC` | ~9 train | levantine | `prepare_masc.py` |
| MGB-3 | `ArabicSpeech/MGB-3` | ~10 train | egyptian | `prepare_mgb3.py` |
| MGB-5 | `ArabicSpeech/MGB-5` | ~10 train | maghrebi | `prepare_mgb5.py` |

All datasets above are openly accessible on HF Hub (no gating, no Mozilla Data Collective dependency).

**Why we use the `MohamedRashad/common-voice-18-arabic` mirror instead of Mozilla's own `common_voice_17_0`:** as of October 2025, Mozilla emptied all official `mozilla-foundation/common_voice_*` HF Hub repos and moved Common Voice exclusively to the Mozilla Data Collective platform. The community CV18 ar mirror still hosts the audio (~2.7 GB, 28k train rows) and remains a faithful drop-in for ASR training.

**Gulf is not a training dialect.** SADA 2022's audio is only on Kaggle (~600 GB, infeasible) and HF Hub mirrors are metadata-only. We retain Gulf as a held-out test dialect via Casablanca; the resulting Gulf WER is a cross-dialect transfer measurement.

## Step 1 — Pull every dataset

These can run in parallel terminals; they don't share state.

```bash
# MSA + FLEURS test
python -m scripts.prepare_common_voice --split train \
  --out test_sets/common_voice_18_ar_train.jsonl \
  --audio-dir audio/common_voice_18_ar

python -m scripts.prepare_fleurs --split test \
  --out test_sets/fleurs_msa_test.jsonl \
  --audio-dir audio/fleurs_ar

# Casablanca multi-dialect (writes one JSONL per dialect)
python -m scripts.prepare_casablanca --split test \
  --out-dir test_sets --audio-dir audio/casablanca \
  --max-per-dialect 500

# Levantine training
python -m scripts.prepare_masc --split train \
  --out test_sets/masc_levantine_train.jsonl \
  --audio-dir audio/masc

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

Bucket creation is one-time. Skip if already present (`gsutil ls -L gs://dev-ahmedhany-whisper-arabic` to check).

```bash
gsutil mb -l us-central1 gs://dev-ahmedhany-whisper-arabic
gsutil -m cp -r test_sets/ gs://dev-ahmedhany-whisper-arabic/test_sets/
gsutil -m cp -r audio/      gs://dev-ahmedhany-whisper-arabic/audio/
```

GCS-to-GCS within the same region is free egress. The training instance will pull from this bucket in `deploy/02_gcp_training.md` Step 2.

## Step 4 — Delete the data-prep VM

The VM is throwaway — once upload to GCS completes, kill it. `e2-standard-4` is cheap but adds up if forgotten.

```bash
exit                                    # leave the SSH session
gcloud compute instances delete whisper-dataprep --zone=us-central1-a --quiet
```

## Storage cost note

GCS Standard storage in `us-central1` is roughly $0.020/GB/month. ~250 GB of decoded audio = ~$5/month while you iterate. Move to Coldline (`gsutil rewrite -s coldline gs://dev-ahmedhany-whisper-arabic/audio/**`) once training is done if you want long-term retention.

## License notes

| Dataset | License | Commercial use |
|---|---|---|
| Common Voice 18 (community mirror) | CC0 (upstream) | Yes |
| FLEURS | CC-BY-4.0 | Yes (with attribution) |
| Casablanca | Per upstream dataset card | Check upstream |
| MASC | Per dataset card | Check upstream |
| MGB-3 | Research use (per ArabicSpeech) | No |
| MGB-5 | Research use (per ArabicSpeech) | No |

The model produced by training on this mix is fine for the paper and for non-commercial release on HF Hub. For commercial deployment, retrain on a subset that excludes the research-only sources (MGB-3, MGB-5) — or seek an explicit license from ArabicSpeech.
