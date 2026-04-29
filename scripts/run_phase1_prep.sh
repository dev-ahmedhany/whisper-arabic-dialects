#!/usr/bin/env bash
# Phase 1 — pull all seven datasets from HuggingFace Hub, build dialect-balanced splits.
#
# Designed to run on the data-prep VM provisioned in deploy/01_dataset_acquisition.md Step 0
# (e2-standard-4, Debian 12, 500 GB disk). Datasets are pulled in parallel groups sized to
# the VM's 4 vCPU and ~15 MB/s practical CDN bandwidth.
#
# Sample caps per dataset are set to oversample the target hours in configs/dataset_mix.yaml;
# src.data_prep then takes whatever is needed to hit the per-dialect target. Drop the caps
# (delete --max-samples lines) for a final run once the pipeline is trusted.
#
# Usage on the VM:
#   cd ~/whisper-arabic-dialects
#   nohup bash scripts/run_phase1_prep.sh > logs/phase1.log 2>&1 &
#   tail -f logs/phase1.log

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

started=$(date +%s)
echo "=== Phase 1 dataset prep STARTED at $(date -Iseconds) ==="

wait_group () {
  local name="$1"; shift
  local pid rc=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      echo "[$(date +%H:%M:%S)] $name: pid $pid exited non-zero"
      rc=1
    fi
  done
  if [[ $rc -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] $name FAILED — see logs/ for details"
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] $name done"
}

# -----------------------------------------------------------------------------
# Group A — Common Voice (CV18 ar mirror) + MASC in parallel
# -----------------------------------------------------------------------------
echo ">>> Group A starting: Common Voice 18 ar + MASC"

python -m scripts.prepare_common_voice --split train --max-samples 8000 \
  --out test_sets/common_voice_18_ar_train.jsonl \
  --audio-dir audio/common_voice_18_ar \
  > logs/01_common_voice.log 2>&1 &
A1=$!

python -m scripts.prepare_masc --split train --max-samples 8000 \
  --out test_sets/masc_levantine_train.jsonl \
  --audio-dir audio/masc \
  > logs/02_masc.log 2>&1 &
A2=$!

wait_group "Group A" "$A1" "$A2"

# -----------------------------------------------------------------------------
# Group B — MGB-3 + MGB-5 in parallel
# -----------------------------------------------------------------------------
echo ">>> Group B starting: MGB-3 + MGB-5"

python -m scripts.prepare_mgb3 --split train --max-samples 6000 \
  --out test_sets/mgb3_egyptian_train.jsonl \
  --audio-dir audio/mgb3 \
  > logs/04_mgb3.log 2>&1 &
B1=$!

python -m scripts.prepare_mgb5 --split train --max-samples 6000 \
  --out test_sets/mgb5_moroccan_train.jsonl \
  --audio-dir audio/mgb5 \
  > logs/05_mgb5.log 2>&1 &
B2=$!

wait_group "Group B" "$B1" "$B2"

# -----------------------------------------------------------------------------
# Group C — FLEURS + Casablanca (small held-out test sets, sequential)
# -----------------------------------------------------------------------------
echo ">>> Group C starting: FLEURS + Casablanca"

python -m scripts.prepare_fleurs --split test \
  --out test_sets/fleurs_msa_test.jsonl \
  --audio-dir audio/fleurs_ar \
  > logs/06_fleurs.log 2>&1

python -m scripts.prepare_casablanca --split test \
  --out-dir test_sets \
  --audio-dir audio/casablanca \
  --max-per-dialect 500 \
  > logs/07_casablanca.log 2>&1

echo "[$(date +%H:%M:%S)] Group C done"

# -----------------------------------------------------------------------------
# Group D — assemble dialect-balanced train/val splits
# -----------------------------------------------------------------------------
echo ">>> Group D starting: src.data_prep"

python -m src.data_prep --config configs/dataset_mix.yaml --output-dir test_sets \
  > logs/08_data_prep.log 2>&1

echo "[$(date +%H:%M:%S)] Group D done"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
elapsed=$(( $(date +%s) - started ))
echo "=== Phase 1 prep DONE at $(date -Iseconds) (elapsed: ${elapsed}s) ==="
echo ""
echo "--- per-dataset row counts ---"
for f in test_sets/*.jsonl; do
  printf '%8d  %s\n' "$(wc -l < "$f")" "$f"
done

echo ""
echo "--- split_summary.json ---"
cat test_sets/split_summary.json

echo ""
echo "--- disk usage ---"
du -sh audio/ test_sets/

echo ""
echo "Next: run scripts/run_phase1_upload.sh to push test_sets/ + audio/ to GCS."
