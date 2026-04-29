# 02 — GCP Training (Vertex AI Workbench, L4)

QLoRA fine-tuning runs on a `g2-standard-16` (1× NVIDIA L4 24GB, Ada Lovelace, bf16 + Flash Attention 2 supported).

## Prerequisites

- GCP project with billing enabled.
- `gcloud` CLI authenticated: `gcloud auth login` and `gcloud config set project YOUR_PROJECT`.
- Datasets uploaded to GCS per `deploy/01_dataset_acquisition.md`.
- `WANDB_API_KEY` and `HF_TOKEN` in your shell.

## Step 1 — Provision the Workbench instance

```bash
PROJECT=$(gcloud config get-value project)
ZONE=us-central1-a       # confirm L4 availability with: gcloud compute accelerator-types list --filter="name:nvidia-l4"
INSTANCE=whisper-train

gcloud workbench instances create "$INSTANCE" \
  --location="$ZONE" \
  --machine-type=g2-standard-16 \
  --accelerator-type=NVIDIA_L4 \
  --accelerator-core-count=1 \
  --vm-image-project=deeplearning-platform-release \
  --vm-image-family=common-cu124 \
  --boot-disk-size=150 \
  --idle-shutdown-timeout=1800 \
  --metadata=install-nvidia-driver=True
```

Wait 5–7 minutes, then open the JupyterLab proxy from the Workbench UI.

## Step 2 — Set up the repo on the instance

In a Workbench terminal:

```bash
git clone https://github.com/dev-ahmedhany/whisper-arabic-dialects.git
cd whisper-arabic-dialects
pip install -r requirements.txt -e .
pip install flash-attn==2.6.3 --no-build-isolation     # L4 supports FA2

mkdir -p test_sets audio
gsutil -m cp -r gs://your-project-whisper-arabic-data/test_sets/ ./
gsutil -m cp -r gs://your-project-whisper-arabic-data/audio/ ./
```

## Step 3 — Sanity training run (~30–45 min)

This validates that the data pipeline, QLoRA config, and W&B logging all work before committing 12+ hours of compute.

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
huggingface-cli login --token "$HF_TOKEN"

bash scripts/run_sanity_train.sh
```

Watch the W&B run. Val WER should drop measurably from epoch 0 to epoch 1.

## Step 4 — Full turbo run (~12–18h)

```bash
tmux new -s train_turbo
bash scripts/run_full_train.sh turbo
# Ctrl+B then D to detach; tmux attach -t train_turbo to reattach
```

Expected: 3 epochs, ~`[NN]` GPU-hours, val WER converges in the high `[XX%]` range.

## Step 5 — Convert to CTranslate2 quantization sweep

After the run finishes:

```bash
python -m src.convert_ct2 \
  --base-model openai/whisper-large-v3-turbo \
  --lora-dir checkpoints/whisper-turbo-ar-lora \
  --merged-dir whisper-merged \
  --output-prefix whisper-faster-turbo-ar
```

Five CT2 dirs land alongside, ready for the benchmark host.

## Step 6 — Push artifacts to GCS / HF Hub

```bash
# back up checkpoints + CT2 variants
gsutil -m cp -r checkpoints/ gs://your-project-whisper-arabic-data/checkpoints/
gsutil -m cp -r whisper-faster-turbo-ar-* gs://your-project-whisper-arabic-data/ct2/

# push CT2 variants to HF Hub (one repo per quant)
huggingface-cli login
for q in float32 float16 int8_float32 int8_float16 int8; do
  huggingface-cli upload \
    "dev-ahmedhany/whisper-large-v3-turbo-ar-${q}" \
    "whisper-faster-turbo-ar-${q}" .
done
```

## Step 7 — Repeat for large-v3

Only do this after the H1 milestone gate has been evaluated (see paper §6):

```bash
bash scripts/run_full_train.sh large-v3
python -m src.convert_ct2 \
  --base-model openai/whisper-large-v3 \
  --lora-dir checkpoints/whisper-large-v3-ar-lora \
  --merged-dir whisper-merged-large-v3 \
  --output-prefix whisper-faster-large-v3-ar
```

## Step 8 — Stop the instance when not training

`g2-standard-16` is ~$0.85/hr. The 30-minute idle shutdown helps but is not foolproof.

```bash
gcloud workbench instances stop "$INSTANCE" --location="$ZONE"
```

## Cost ceiling

Approximate spend: ~$15 for turbo training, ~$25 for large-v3 training, ~$10 buffer for re-runs. Total under $50.
