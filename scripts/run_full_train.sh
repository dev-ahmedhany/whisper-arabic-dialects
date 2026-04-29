#!/usr/bin/env bash
# Full 3-epoch training run on the L4 instance. Expected wall-clock: ~12–18h for turbo,
# ~24–30h for large-v3. Use `nohup` or `tmux` so the run survives SSH disconnects.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${1:-turbo}"
case "$MODEL" in
  turbo)    CONFIG="configs/train_turbo.yaml" ;;
  large-v3) CONFIG="configs/train_large_v3.yaml" ;;
  medium)   CONFIG="configs/train_medium.yaml" ;;
  small)    CONFIG="configs/train_small.yaml" ;;
  *)
    echo "usage: $0 [turbo|large-v3|medium|small]"
    exit 1
    ;;
esac

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY not set — pass --no-wandb or set the env var first."
  exit 1
fi

echo "Starting full training: $MODEL ($CONFIG)"
python -m src.train --config "$CONFIG"

echo "Done. Checkpoint at the output_dir defined in $CONFIG"
echo "Next: convert with: python -m src.convert_ct2 --base-model <model_id> --lora-dir <output_dir> --output-prefix whisper-faster-${MODEL}-ar"
