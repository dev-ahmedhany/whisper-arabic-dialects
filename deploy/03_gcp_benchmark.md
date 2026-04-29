# 03 — GCP CPU Benchmark (`c3-standard-8`, Intel Sapphire Rapids)

CPU-only inference benchmarking on the reproducibility platform: a Sapphire Rapids `c3-standard-8` (8 vCPU, 32 GB) — the platform anyone with a GCP account can spin up to verify the paper's numbers.

## Prerequisites

- GCP project, datasets and CT2 model variants on GCS or HF Hub.
- `gcloud` authenticated.
- Image `whisper-arabic-bench:latest` published to a registry your project can pull from (Artifact Registry or Docker Hub) — see Step 0.

## Step 0 — Build and publish the benchmark image (one-time, from your laptop)

```bash
docker build -t whisper-arabic-bench:latest .

# Option A — Docker Hub
docker tag whisper-arabic-bench:latest <dockerhub-user>/whisper-arabic-bench:latest
docker push <dockerhub-user>/whisper-arabic-bench:latest

# Option B — GCP Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
docker tag whisper-arabic-bench:latest \
  us-central1-docker.pkg.dev/$(gcloud config get-value project)/whisper/bench:latest
docker push us-central1-docker.pkg.dev/$(gcloud config get-value project)/whisper/bench:latest
```

## Step 1 — Provision `c3-standard-8`

```bash
PROJECT=$(gcloud config get-value project)
ZONE=us-central1-a
INSTANCE=whisper-bench-gcp

gcloud compute instances create "$INSTANCE" \
  --zone="$ZONE" \
  --machine-type=c3-standard-8 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --metadata=enable-oslogin=TRUE
```

Cost: ~$0.40/hr.

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
