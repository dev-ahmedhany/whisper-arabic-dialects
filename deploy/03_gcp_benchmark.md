# 03 — GCP CPU Benchmark (`c3-standard-8`, Intel Sapphire Rapids)

CPU-only inference benchmarking on the reproducibility platform: a Sapphire Rapids `c3-standard-8` (8 vCPU, 32 GB) — the platform anyone with a GCP account can spin up to verify the paper's numbers.

## Prerequisites

- `deploy/00_gcp_bootstrap.md` complete, datasets in GCS per `deploy/01_dataset_acquisition.md`.
- For Phase 3b (fine-tuned models): CT2 model variants pushed to GCS or HF Hub (after Phase 2).
- For **Phase 3a (zero-shot baselines, no training required)**: nothing else — `faster_whisper.WhisperModel("openai/whisper-large-v3-turbo")` auto-pulls pre-converted CT2 weights from HF Hub on first use.

## Phase 3a vs 3b

The benchmark matrix splits naturally into two halves so you can fill the paper's
zero-shot rows in parallel with training:

| Sub-phase | Models | When to run | Tables it fills |
|---|---|---|---|
| **Phase 3a** | `zero-shot-large-v3`, `zero-shot-turbo` | Anytime after Phase 1 | ZS rows of Table 1 (and §4 Zero-Shot Baselines prose) |
| **Phase 3b** | `ft-turbo`, `ft-large-v3` | After Phase 2 + CT2 conversion | FT rows of Table 1; all of Tables 2, 3, 4 |

Same `c3-standard-8` instance, same image, same harness — just `--include-models` filters which rows of `runs/results.jsonl` get added.

## Step 0 — Build and publish the benchmark image

Two options depending on whether you want local Docker activity:

```bash
# Option A — Cloud Build (server-side; preferred — no local Docker, no upload bandwidth)
PROJECT=$(gcloud config get-value project)
gcloud artifacts repositories create whisper \
  --repository-format=docker \
  --location=us-central1 \
  --description="whisper-arabic benchmark images" 2>/dev/null || true
gcloud builds submit \
  --tag "us-central1-docker.pkg.dev/$PROJECT/whisper/bench:latest" .
```

```bash
# Option B — local Docker push
docker build -t whisper-arabic-bench:latest .
gcloud auth configure-docker us-central1-docker.pkg.dev
PROJECT=$(gcloud config get-value project)
docker tag whisper-arabic-bench:latest \
  us-central1-docker.pkg.dev/$PROJECT/whisper/bench:latest
docker push us-central1-docker.pkg.dev/$PROJECT/whisper/bench:latest
```

## Step 1 — Provision `c3-standard-8`

```bash
gcloud compute instances create whisper-bench-gcp \
  --zone=us-central1-a \
  --machine-type=c3-standard-8 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --scopes=cloud-platform
```

Cost: ~$0.40/hr. `--scopes=cloud-platform` lets the VM pull from GCS and Artifact Registry without separate auth. 100 GB disk is plenty: 3.5 GB datasets + ~10 GB Docker layers + OS leaves ample headroom and stays well under the 510 GB SSD quota.

## Step 2 — Connect, install Docker, pull image

```bash
gcloud compute ssh "$INSTANCE" --zone="$ZONE"
```

On the instance:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER"
exit
gcloud compute ssh "$INSTANCE" --zone="$ZONE"   # re-login for group to take effect

docker pull <your-image-ref>
```

## Step 3 — Pull test sets and CT2 model variants

```bash
sudo apt-get install -y google-cloud-sdk
gcloud auth login
mkdir -p ~/whisper-arabic && cd ~/whisper-arabic

gsutil -m cp -r gs://your-project-whisper-arabic-data/test_sets/ ./
mkdir -p models
gsutil -m cp -r gs://your-project-whisper-arabic-data/ct2/whisper-faster-turbo-ar-* models/
gsutil -m cp -r gs://your-project-whisper-arabic-data/ct2/whisper-faster-large-v3-ar-* models/
```

(Or pull CT2 variants directly from HF Hub — the harness accepts HF model ids too.)

## Step 4 — Run the benchmark matrices

The Docker image's entrypoint is `scripts.run_benchmark_matrix`. Mount results, test sets, and the HF cache:

```bash
RUNS_DIR=$(pwd)/runs && mkdir -p "$RUNS_DIR"

docker run --rm \
  --cpus=8 \
  -v "$RUNS_DIR":/app/runs \
  -v "$(pwd)/test_sets":/app/test_sets \
  -v "$(pwd)/models":/app/models \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  <your-image-ref> \
  --config configs/benchmark_matrix_quality.yaml \
  --platform-label gcp-c3-standard-8

# Repeat for the other matrices
docker run --rm --cpus=8 ... --config configs/benchmark_matrix_speed.yaml --platform-label gcp-c3-standard-8
docker run --rm --cpus=8 ... --config configs/benchmark_matrix_pareto.yaml --platform-label gcp-c3-standard-8
docker run --rm --cpus=8 ... --config configs/benchmark_matrix_threads.yaml --platform-label gcp-c3-standard-8
```

The quality+pareto matrices for both fine-tuned models is roughly 200–300 cells, ~5–15 min per cell — figure on a few days of wall-clock if running serially. Run at night with `nohup`/`tmux`.

## Step 5 — Pull results back

```bash
gcloud compute scp --recurse \
  "$INSTANCE":~/whisper-arabic/runs ./runs-gcp \
  --zone="$ZONE"
```

## Step 6 — Stop the instance

```bash
gcloud compute instances stop "$INSTANCE" --zone="$ZONE"
# or delete entirely once you're sure you're done:
gcloud compute instances delete "$INSTANCE" --zone="$ZONE"
```

## Verification

After download, append to the local `runs/results.jsonl` and rebuild paper tables:

```bash
cat runs-gcp/results.jsonl >> runs/results.jsonl
python -m src.build_results_tables
```

Open `paper/paper.md` — Tables 1–4 should now contain real numbers.
