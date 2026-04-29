# 04 — Hetzner Cost-per-Audio-Minute Benchmark

Two Hetzner classes are benchmarked alongside GCP c3-standard-8:

| Tier | Instance | Cores | RAM | $/hr | Purpose |
|---|---|---|---|---|---|
| **Cheapest x86** | `cx23` | 2 (shared, AMD) | 4 GB | $0.008 | low-end cost-per-minute floor |
| **Cheapest ARM** | `cax11` | 2 (shared, Ampere Altra) | 4 GB | $0.009 | ARM cost vs x86 at same RAM tier |
| **Production x86** | `cx53` | 16 (shared, AMD Zen) | 32 GB | $0.043 | full benchmark replay vs GCP |

§A covers the **cheap x86 + ARM cost sweep** (the §3.10 cost-per-min finding).
§B covers the original **CX53 cross-platform replay** (validates GCP WER).

The CX23/CAX11 sweep is ~$0.01 total. The CX53 replay is ~$2.

## Prerequisites

- Hetzner Cloud account and API token: https://console.hetzner.cloud
- `hcloud` CLI installed and configured (`brew install hcloud` then `hcloud context create whisper-arabic`).
- An SSH key registered with Hetzner (`hcloud ssh-key list`).
- Local copy of the MSA test set + audio (or rsync from a GCP bench host — see §A Step 1.5).

## §A — Cheap-tier cost-per-minute sweep (cx23 + cax11)

### Step A1 — Provision both servers

```bash
hcloud server create --name whisper-bench-cx23  --type cx23  --image ubuntu-24.04 --location nbg1 --ssh-key laptop
hcloud server create --name whisper-bench-cax11 --type cax11 --image ubuntu-24.04 --location nbg1 --ssh-key laptop
hcloud server list
```

Note the public IPv4 of each — referred to as `$IP_CX23` and `$IP_CAX11` below.

### Step A1.5 — Stage the MSA test data (audio is in private GCS)

The dataset bucket isn't world-readable, so package the MSA-only subset on a host that already has it (e.g. `whisper-bench-cpu` from `deploy/03_gcp_benchmark.md`) and push to Hetzner via your laptop:

```bash
gcloud compute ssh whisper-bench-cpu --zone=us-central1-a --tunnel-through-iap \
  --command="cd ~/whisper-arabic-dialects && tar czf /tmp/hetzner_payload.tgz \
             test_sets/test_msa_fleurs_msa_test.jsonl audio/fleurs_ar/"
gcloud compute scp whisper-bench-cpu:/tmp/hetzner_payload.tgz /tmp/hetzner_payload.tgz \
  --zone=us-central1-a --tunnel-through-iap

# 117 MB of audio + JSONL — ~30 s upload to nbg1
scp -i ~/.ssh/id_ed25519 /tmp/hetzner_payload.tgz root@$IP_CX23:/root/
scp -i ~/.ssh/id_ed25519 /tmp/hetzner_payload.tgz root@$IP_CAX11:/root/
```

### Step A2 — Run the bootstrap + sweep on each server

The same script `hetzner_run.sh` works for both (parameterized by platform label). It:

1. Installs `python3-pip python3-venv git ffmpeg libsndfile1`
2. Clones the repo and installs `pip install -e . faster-whisper jiwer scipy psutil pyyaml tqdm`
3. Untars `/root/hetzner_payload.tgz` and rewrites the absolute audio paths in the test JSONL to be relative
4. Runs `python -m src.eval_harness` for each model in the sweep with `--compute-type int8 --beam-size 1 --cpu-threads 2`

```bash
# script body lives inline in deploy/_assets/hetzner_run.sh of this repo, or copy from /tmp.
scp /tmp/hetzner_run.sh root@$IP_CX23:/root/run.sh
ssh root@$IP_CX23 "chmod +x /root/run.sh && nohup /root/run.sh hetzner-cx23 \
  'tiny small turbo' > /root/run_outer.log 2>&1 & disown"

scp /tmp/hetzner_run.sh root@$IP_CAX11:/root/run.sh
ssh root@$IP_CAX11 "chmod +x /root/run.sh && nohup /root/run.sh hetzner-cax11 \
  'tiny small turbo' > /root/run_outer.log 2>&1 & disown"
```

`--cpu-threads 2` matches the actual core count of cx23/cax11. `large-v3` is **not** in the sweep — its CT2 int8 peak is 3.7 GB, which is too tight for the 4 GB box (no margin for OS/io). If you want it, upgrade to a `cx33`/`cax21`.

### Step A3 — Sync results and tear down

```bash
scp root@$IP_CX23:~/whisper-arabic-dialects/runs/results.jsonl  /tmp/cx23_results.jsonl
scp root@$IP_CAX11:~/whisper-arabic-dialects/runs/results.jsonl /tmp/cax11_results.jsonl

cat runs/results.jsonl /tmp/cx23_results.jsonl /tmp/cax11_results.jsonl | sort -u > /tmp/merged.jsonl
mv /tmp/merged.jsonl runs/results.jsonl

python -m src.cost_per_min   # regenerates paper/COST.md with measured cells

hcloud server delete whisper-bench-cx23 whisper-bench-cax11
```

The `--measured` flag of `cost_per_min` will now drop the *(proj)* tags from the cx23/cax11 columns.

## §B — Production-replay sweep on CX53 (original plan)

A Hetzner CX53 is a 16 vCPU AMD EPYC 32 GB instance at ~$0.043/hr. It is the production-deployment target — commodity low-cost cloud — and the cross-platform comparison vs GCP Sapphire Rapids is itself a paper finding.

Total expected cost: ~$2 for the entire benchmark.

### Prerequisites (CX53)

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
