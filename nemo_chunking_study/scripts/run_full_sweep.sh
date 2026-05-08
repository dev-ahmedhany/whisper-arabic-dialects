#!/bin/bash
# Full sweep: chunk_ms ∈ [2..30] s, overlap ∈ {0, 500, 1000} ms,
# silence_trim ∈ {off, on}, on the sherpa-onnx RNNT path AND the
# NeMo PyTorch CTC path. ~10 configs × 3 overlaps × 2 trim × 2
# backends = ~120 configs (some skipped where overlap≥chunk). Total
# ~3-4 hours on e2-standard-16.
#
# Run as a GCE startup-script (not via SSH). The script self-contains
# everything from package install to result upload.

exec > >(tee -a /tmp/sweep.log) 2>&1
set -e
echo "=== sweep start: $(date -u) ==="

apt-get update -qq
apt-get install -y -qq python3-pip python3-venv ffmpeg curl

python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --quiet --upgrade pip wheel

echo "--- pip install ---"
# The PyTorch CTC backend needs nemo_toolkit which is a heavy install
# (drags PyTorch + transformers + lhotse + librosa + …). The sherpa-only
# subset is much lighter. Caller passes BACKENDS env var.
pip install --quiet sherpa-onnx huggingface_hub datasets jiwer soundfile numpy psutil onnx torchcodec
if echo "${BACKENDS:-sherpa_rnnt}" | grep -q nemo_pytorch_ctc; then
  pip install --quiet "nemo_toolkit[asr]"
fi

set +x
HF_TOKEN=$(curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token)
export HF_TOKEN

echo "--- pull RNNT bundle via curl (idempotent) ---"
mkdir -p /tmp/rnnt
for f in encoder.onnx decoder.onnx joiner.onnx tokens.txt silero_vad.onnx; do
  if [ ! -s "/tmp/rnnt/$f" ]; then
    curl -sLf --max-time 600 -o "/tmp/rnnt/$f" \
      -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/dev-ahmedhany/stt_ar_fastconformer_hybrid_large_pcd_v1.0-sherpa/resolve/main/$f"
  fi
done

# Sherpa-onnx wants vocab_size etc. on every transducer ONNX, not just
# the encoder. Patch decoder/joiner if needed.
python3 - <<'PYEOF'
import onnx
src = onnx.load('/tmp/rnnt/encoder.onnx')
src_meta = {p.key: p.value for p in src.metadata_props}
for f in ['/tmp/rnnt/decoder.onnx', '/tmp/rnnt/joiner.onnx']:
    m = onnx.load(f)
    have = {p.key for p in m.metadata_props}
    for k, v in src_meta.items():
        if k not in have:
            p = m.metadata_props.add(); p.key = k; p.value = v
    onnx.save(m, f)
print('metadata stamped')
PYEOF

echo "--- pull bench script ---"
curl -sLf -o /tmp/run_sweep.py \
  https://raw.githubusercontent.com/dev-ahmedhany/whisper-arabic-dialects/main/nemo_chunking_study/scripts/run_sweep.py 2>/dev/null \
  || cat <<'PYEOF' > /tmp/run_sweep.py
# Fallback: caller is expected to scp run_sweep.py before reset.
# This is filled in by the deploy script when it adds the metadata.
PYEOF

echo "--- run sweep ---"
mkdir -p /tmp/results
python3 /tmp/run_sweep.py \
  --model-dir /tmp/rnnt \
  --backends "${BACKENDS:-sherpa_rnnt}" \
  --out /tmp/results/sweep_results.jsonl

echo "--- upload results to GCS bucket if configured ---"
if [ -n "${RESULTS_BUCKET:-}" ]; then
  gsutil cp /tmp/results/sweep_results.jsonl "gs://$RESULTS_BUCKET/sweep_$(date -u +%Y%m%dT%H%M%SZ).jsonl" || true
fi

echo "=== sweep DONE: $(date -u) ==="
touch /tmp/sweep.done
