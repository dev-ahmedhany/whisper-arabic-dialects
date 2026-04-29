#!/usr/bin/env bash
# 5h subset / 1 epoch sanity check before committing to the full training run.
# Run this on the GCP L4 instance after datasets are prepared.
#
# Expected wall-clock on L4: ~30–45 min.
# Expected outcome: val WER drops measurably from baseline; loss curve trends down.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f test_sets/train.jsonl ]]; then
  echo "test_sets/train.jsonl missing — run scripts/prepare_*.py + python -m src.data_prep first."
  exit 1
fi

# Slice 5h-equivalent rows for the sanity train
python -c "
import json, random
random.seed(42)
rows = [json.loads(l) for l in open('test_sets/train.jsonl')]
random.shuffle(rows)
target_s = 5 * 3600
acc = 0
sub = []
for r in rows:
    if acc >= target_s: break
    sub.append(r); acc += float(r['duration_s'])
with open('test_sets/train_sanity.jsonl', 'w') as f:
    for r in sub: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'sanity train: {len(sub)} rows, {acc/3600:.2f}h')
"

python -m src.train \
  --config configs/train_turbo.yaml \
  --num-train-epochs 1 \
  --output-dir checkpoints/sanity-turbo
