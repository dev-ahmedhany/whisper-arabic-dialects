# 00 — GCP Project Bootstrap

One-time setup that takes a fresh GCP account to the state required by `deploy/02_gcp_training.md`, `deploy/03_gcp_benchmark.md`, and any GCS-backed step. Skip if you've already linked billing, set defaults, and enabled the required APIs on the project.

## Prerequisites

- A Google account.
- A GCP project (create one in the Console or with `gcloud projects create whisper-arabic-XXXXXX`). Note the project ID.
- A linked billing account (Console → Billing → "Link a billing account").

## Step 1 — Install the SDK

```bash
brew install --cask google-cloud-sdk
gcloud --version
```

The brew cask installs `gcloud`, `gsutil`, `bq`, `gcloud-crc32c`, all on PATH at `/opt/homebrew/bin/`.

## Step 2 — Authenticate

Two separate credentials are needed and they store independently:

```bash
gcloud auth login                          # interactive — opens a browser
gcloud auth application-default login      # interactive — second browser tab
```

`auth login` is for `gcloud` CLI commands. `application-default login` is for any Python library that uses Google Application Default Credentials (`gsutil` from Python, HF datasets push to Google-backed storage, etc.). You need both.

Verify:

```bash
gcloud auth list
# expect: ACTIVE *  ACCOUNT <your-email>
```

## Step 3 — Pin the project + defaults

```bash
PROJECT_ID=brain-tumor-73704                # replace with your project id
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

gcloud config list                          # snapshot
```

Default region/zone matters: `us-central1-a` is where L4 GPUs are provisioned for training, and the same zone keeps GCS-to-VM transfers in-region (free egress).

## Step 4 — Verify billing is enabled on this project

```bash
gcloud billing projects describe "$PROJECT_ID" \
  --format="value(billingEnabled,billingAccountName)"
# expect: True  billingAccounts/XXXXXX-XXXXXX-XXXXXX
```

If `billingEnabled` is `False`, link a billing account in the Console (Billing → "My billing accounts" → click into one → Linked projects → "Add"). API calls and instance creation will silently fail until this is `True`.

## Step 5 — Enable required APIs

```bash
gcloud services enable \
  compute.googleapis.com \
  notebooks.googleapis.com \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  cloudbuild.googleapis.com
```

Why each:

| API | Used in |
|---|---|
| `compute.googleapis.com` | All VM provisioning (training L4 + CPU benchmark `c3-standard-8`) |
| `notebooks.googleapis.com` + `aiplatform.googleapis.com` | Vertex AI Workbench (the JupyterLab interface for training) |
| `storage.googleapis.com` | GCS bucket for datasets, audio, checkpoints, CT2 model variants |
| `artifactregistry.googleapis.com` | Docker image registry for the CPU benchmark image |
| `iam.googleapis.com` | Service accounts and instance permissions |
| `cloudbuild.googleapis.com` | Optional — server-side `docker build` if you don't want to build locally |

Verify:

```bash
gcloud services list --enabled \
  --filter="config.name:(compute.googleapis.com OR notebooks.googleapis.com OR aiplatform.googleapis.com OR storage.googleapis.com OR artifactregistry.googleapis.com OR iam.googleapis.com OR cloudbuild.googleapis.com)" \
  --format="table(config.name)"
```

## Step 6 — Check GPU quota for training

Phase 2 needs:
- `NVIDIA_L4_GPUS ≥ 1` in `us-central1` (per-region quota)
- `GPUS_ALL_REGIONS ≥ 1` (project-wide meta-quota that gates *every* GPU)

Both must pass. Run:

```bash
# Per-region GPU quotas
gcloud compute regions describe us-central1 \
  --flatten="quotas[]" \
  --format="value(quotas.metric, quotas.limit, quotas.usage)" \
  | grep -iE "(NVIDIA|GPU)" \
  | column -t -s $'\t'

# Project-wide GPU meta-quota
gcloud compute project-info describe \
  --flatten="quotas[]" \
  --format="value(quotas.metric, quotas.limit, quotas.usage)" \
  | grep -iE "GPU" \
  | column -t -s $'\t'
```

You're looking for these two lines in the output:
```
NVIDIA_L4_GPUS    1.0  0.0
GPUS_ALL_REGIONS  1.0  0.0
```

If either limit is `0.0`:

1. Console → IAM & Admin → Quotas & System Limits.
2. Filter for the failing metric (`NVIDIA L4 GPUs` filtered to `us-central1`, or `GPUs (all regions)` global).
3. Tick the row → "Edit Quotas" → request `1`.
4. Justification: "Research project — QLoRA fine-tuning of Whisper for multi-dialect Arabic ASR. Single L4, expected ~30 GPU-hours total."
5. Approval typically lands in 1–24 hours.

You cannot proceed to `deploy/02_gcp_training.md` until both quotas are at least `1`.

## Step 7 — Set up a budget alert

```bash
# Find your billing account id
gcloud billing accounts list --format="value(name)"
```

Then in the Console: Billing → Budgets & alerts → Create Budget → name "whisper-arabic" → scope to project `brain-tumor-73704` → amount `$150` → alerts at 50% / 90% / 100% by email.

This is the safety net against forgetting to stop a `g2-standard-16` over a weekend.

## What you have after this runbook

- `gcloud config list` shows your account, project, region, zone all set.
- `gcloud billing projects describe ...` shows `billingEnabled=True`.
- Seven APIs enabled.
- L4 quota approved (or request submitted).
- A budget alert wired up.

Next step: `deploy/01_dataset_acquisition.md` (data prep) — that runs entirely against HF Hub and GCS, no GPU quota required.
