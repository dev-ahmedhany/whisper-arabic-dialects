# 04 — Hetzner CX53 Benchmark (AMD EPYC, production validation)

A Hetzner CX53 is a 16 vCPU AMD EPYC 32 GB instance at ~$0.043/hr. It is the production-deployment target — commodity low-cost cloud — and the cross-platform comparison vs GCP Sapphire Rapids is itself a paper finding.

Total expected cost: ~$2 for the entire benchmark.

## Prerequisites

- Hetzner Cloud account and API token: https://console.hetzner.cloud
- `hcloud` CLI installed and configured (`brew install hcloud` then `hcloud context create whisper`).
- The benchmark Docker image already published per `deploy/03_gcp_benchmark.md` Step 0.

## Step 1 — Provision a CX53

```bash
hcloud server create \
  --name whisper-bench-hetzner \
  --type cx53 \
  --image ubuntu-24.04 \
  --location nbg1 \
  --ssh-key "$(hcloud ssh-key list -o noheader -o columns=name | head -n 1)"
```

Get the IP:

```bash
HOST=$(hcloud server ip whisper-bench-hetzner)
ssh root@"$HOST"
```

## Step 2 — Install Docker

```bash
apt-get update
apt-get install -y docker.io
docker pull <your-image-ref>
```

## Step 3 — Pull test sets and CT2 model variants

If your HF Hub models are public, pulling at runtime via the harness is simplest. Otherwise sync from GCS:

```bash
mkdir -p ~/whisper-arabic && cd ~/whisper-arabic

# Option A: HF Hub (preferred — keeps Hetzner ↔ GCP transfer at zero)
mkdir -p test_sets
hf download dev-ahmedhany/casablanca-test-sets --repo-type dataset --local-dir test_sets   # if you mirror them

# Option B: GCS
apt-get install -y google-cloud-sdk
gcloud auth login
gsutil -m cp -r gs://your-project-whisper-arabic-data/test_sets/ ./
gsutil -m cp -r gs://your-project-whisper-arabic-data/ct2/ ./models/
```

## Step 4 — Run a smart subset (don't replay the whole matrix)

We replay only the production-relevant cells from `configs/benchmark_matrix_pareto.yaml`:

```bash
RUNS_DIR=$(pwd)/runs && mkdir -p "$RUNS_DIR"

docker run --rm \
  --cpus=16 \
  -v "$RUNS_DIR":/app/runs \
  -v "$(pwd)/test_sets":/app/test_sets \
  -v "$(pwd)/models":/app/models \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  <your-image-ref> \
  --config configs/benchmark_matrix_pareto.yaml \
  --platform-label hetzner-cx53 \
  --include-models ft-turbo ft-large-v3
```

Then a thread-scaling sweep with the CX53's full 16 vCPU:

```bash
# Edit configs/benchmark_matrix_threads.yaml first to add 16 to cpu_threads
docker run --rm --cpus=16 ... \
  --config configs/benchmark_matrix_threads.yaml \
  --platform-label hetzner-cx53
```

## Step 5 — Sync results back to your laptop

```bash
scp -r root@"$HOST":~/whisper-arabic/runs ./runs-hetzner
```

## Step 6 — DESTROY the instance

```bash
hcloud server delete whisper-bench-hetzner
```

Hetzner bills hourly — leaving the box up for a forgotten weekend is the only way to overshoot the $2 budget.

## Verification

```bash
cat runs-hetzner/results.jsonl >> runs/results.jsonl
python -m src.build_results_tables
```

Table 5 (cross-platform) should now populate. The interesting numbers are RTF ratios per cell and the resulting cost-per-hour-of-audio: any RTF degradation up to 9.3× still leaves CX53 cheaper than `c3-standard-8` per transcribed hour.
