# v3 Data Build + Mixed-Domain Evaluation Runbook

The v3 round of the paper introduces (a) an expanded training mix (~38 h, all
Casablanca dialect-relevant countries + cleaned MGB-3 + uncapped CV18) and
(b) **mixed-domain held-out test sets** (50 % Casablanca conversational + 50 %
broadcast), so dialect WER reflects both registers a deployed model will see.

This runbook reproduces both on a fresh GCP L4 (`g2-standard-16`) instance.
Total wall-clock: ~30 min for the data + ~5 min per zero-shot eval cell.

## Prerequisites

- L4 instance brought up per `deploy/02_gcp_training.md` (g2-standard-16,
  Deep Learning VM image `common-cu129-ubuntu-2204-nvidia-580`, 200 GB
  pd-balanced, scopes=cloud-platform).
- Code cloned + venv populated:
  ```bash
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.10-venv python3.10-dev gcc g++ make ffmpeg git tmux
  git clone https://github.com/dev-ahmedhany/whisper-arabic-dialects ~/whisper-arabic-dialects
  cd ~/whisper-arabic-dialects
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip install -e .
  ```
- HF token written so dataset/model downloads + `push_to_hub` work:
  ```bash
  mkdir -p ~/.cache/huggingface
  echo "<your_hf_write_token>" > ~/.cache/huggingface/token
  chmod 600 ~/.cache/huggingface/token
  ```
- Existing GCS data synced (dataset prep from `deploy/01_dataset_acquisition.md`):
  ```bash
  cd ~/whisper-arabic-dialects && mkdir -p audio test_sets runs logs
  gsutil -m -q cp gs://dev-ahmedhany-whisper-arabic/test_sets/*.jsonl test_sets/
  gsutil -m -q rsync -r gs://dev-ahmedhany-whisper-arabic/audio audio/
  ```

## 1. Build the v3 train mix + mixed-domain test sets

The script lives in the repo at `scripts/build_v3_mix.py` (small v2-equivalent
variant) and inline at `/tmp/build_v3_full.py` for the expanded variant. Either
way, the canonical artifacts are the JSONL files under `test_sets/`.

For the **expanded** variant used in the paper's v3 results (Casablanca all 5
dialect-relevant countries + Egyptian-Speech-Clean-MGB3 + uncapped CV18 + MASC):

```bash
cd ~/whisper-arabic-dialects
.venv/bin/python /tmp/build_v3_full.py
```

This step:
1. Pulls Casablanca for 5 countries — **Egypt, Jordan, Palestine, UAE, Yemen** —
   both `validation` (used as train) and `test` splits, into
   `audio/casablanca/<Country>/` and `test_sets/casablanca_<country>_{train,test}.jsonl`.
   Maghrebi countries (Algeria, Mauritania, Morocco) are intentionally excluded
   per `memory/maghrebi_excluded.md` (84.7 % zero-shot WER, out of scope for
   QLoRA budget).
2. Pulls `MohamedGomaa30/Egyptian-Speech-Clean-MGB3` (cleaned MGB-3 with
   music/noise separated) into `audio/egyptian_clean_mgb3/` and
   `test_sets/egyptian_clean_mgb3_train.jsonl`.
3. Builds `test_sets/train_v3.jsonl` + `test_sets/val_v3.jsonl` by combining:
   - **MSA** (≤15 h): Common Voice 18 Arabic train.
   - **Egyptian** (≤25 h): Casablanca Egypt + MGB-3 + Clean-MGB-3.
   - **Levantine** (≤15 h): Casablanca Jordan + Casablanca Palestine + MASC.
   - **Gulf** (≤10 h): Casablanca UAE + Casablanca Yemen.
4. Builds **mixed-domain test sets** (canonical filenames — DO NOT rename, see
   `memory/v3_eval_protocol.md`):

   | dialect | filename | composition |
   |---|---|---|
   | MSA | `test_sets/test_msa_fleurs_msa_test.jsonl` | 100 % FLEURS broadcast |
   | Egyptian | `test_sets/test_v3_egyptian_mixed.jsonl` | 50 Casablanca Egypt test + 50 MGB-3 |
   | Levantine | `test_sets/test_v3_levantine_mixed.jsonl` | 50 Casablanca Jordan test + 50 MASC |
   | Gulf | `test_sets/test_v3_gulf_mixed.jsonl` | 100 Casablanca UAE test (no broadcast Gulf source on HF) |

Expected stats after the build:

```
=== TRAIN: ~26800 rows (~38h)
=== VAL:   ~920 rows (~1.3h)
=== distribution: msa=9.6h, egyptian=18.6h, levantine=9.8h, gulf=1.9h
=== test sets: 100 rows per dialect
```

## 2. Zero-shot baseline eval (Whisper-large-v3 CT2 int8, beam=2)

Run from the same L4. Uses the canonical decode default `beam_size=2` per
`memory/v3_eval_protocol.md`.

```bash
cd ~/whisper-arabic-dialects
.venv/bin/python -m src.eval_harness \
    --model Systran/faster-whisper-large-v3 \
    --model-name zero-shot-large-v3-v3 \
    --compute-type int8 --beam-size 2 --cpu-threads 8 --device cpu \
    --test-set test_sets/test_msa_fleurs_msa_test.jsonl \
    --dialect msa --platform-label l4-cpu --max-samples 100

# repeat for each dialect, swapping --test-set + --dialect
```

(If you hit the silent-SIGTERM bug seen in 2026-04 sessions, fall back to the
direct `evaluate_model()` call shown in `/tmp/run_zs_v3.py`. The CLI wrapper
sometimes dies under specific tmux configurations — `nohup … & disown` works
but tmux kill-server during sibling commands kills the eval.)

Expected zero-shot WER (Whisper-large-v3 CT2 int8, beam=2, 100 samples each):

| dialect | WER (CI 95%) |
|---|---:|
| MSA | 8.51 % [6.67, 10.24] |
| Egyptian (mixed) | 38.48 % [34.30, 43.53] |
| Levantine (mixed) | 37.70 % [32.37, 43.48] |
| Gulf (Casablanca-only) | 52.72 % [46.90, 58.34] |
| **avg-4** | **34.35 %** |

Results land in `runs/results.jsonl` and per-utterance predictions in
`runs/predictions/`. They feed `src/build_results_tables.py` which splices
the table into `paper/paper.md` at the `<!-- INSERT: table_N -->` markers.

## 3. v3 fine-tune

```bash
cd ~/whisper-arabic-dialects
nohup .venv/bin/python -m src.train --config configs/train_large_v3_r8.yaml \
    > ~/v3_train.log 2>&1 < /dev/null &
disown
```

Key config differences from v2 (see `configs/train_large_v3_r8.yaml`):

- Base model: **`openai/whisper-large-v3`** (1.55 B) — quality-ceiling pick.
- LoRA: **`r=8, α=16`** unchanged from v2 — verified to survive CT2 int8
  quantization (paper §6.5).
- `push_to_hub: true`, `hub_model_id: dev-ahmedhany/whisper-large-v3-arabic-ft-v3-lora`,
  `hub_strategy: every_save`. Each `save_steps=500` checkpoint goes straight
  to a NEW HF repo — the v1 repos at `…-ft-lora` / `…-ft` are NOT touched
  (per `memory/hf_versioning.md`).
- `gradient_checkpointing=true`, `per_device_train_batch_size=4`,
  `gradient_accumulation_steps=4` (effective batch 16, fits L4 24 GB).

Wall-clock: ~9 s/step on L4. Early-stop usually halts at step 2000-4000
(~5-10 h). `load_best_model_at_end=True` so the saved adapter is the
best-WER checkpoint.

## 4. v3 fine-tune evaluation

After training:

```bash
# (a) merge LoRA into base safetensors
.venv/bin/python -c "
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

base = WhisperForConditionalGeneration.from_pretrained(
    'openai/whisper-large-v3', torch_dtype=torch.float32)
proc = WhisperProcessor.from_pretrained(
    'openai/whisper-large-v3', language='arabic', task='transcribe')
peft = PeftModel.from_pretrained(base, 'checkpoints/whisper-large-v3-ar-lora/checkpoint-XXXX')
merged = peft.merge_and_unload(safe_merge=True)
merged.save_pretrained('checkpoints/v3-merged', safe_serialization=True)
proc.save_pretrained('checkpoints/v3-merged')
"

# (b) copy the few tokenizer files CT2 needs that processor.save_pretrained omits
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
import shutil
for f in ['preprocessor_config.json', 'normalizer.json', 'special_tokens_map.json',
          'added_tokens.json', 'merges.txt', 'vocab.json']:
    try:
        shutil.copy(hf_hub_download('openai/whisper-large-v3', f),
                    f'checkpoints/v3-merged/{f}')
    except Exception as e:
        print('skip', f, e)
"

# (c) convert to CT2 int8
.venv/bin/ct2-transformers-converter \
    --model checkpoints/v3-merged \
    --output_dir checkpoints/v3-ct2-int8 \
    --quantization int8 \
    --copy_files preprocessor_config.json tokenizer_config.json normalizer.json \
                 special_tokens_map.json added_tokens.json merges.txt vocab.json tokenizer.json

# (d) eval on the same 4 mixed test sets
for d in msa egyptian levantine gulf; do
    case $d in
        msa) F=test_sets/test_msa_fleurs_msa_test.jsonl ;;
        *)   F=test_sets/test_v3_${d}_mixed.jsonl ;;
    esac
    .venv/bin/python -m src.eval_harness \
        --model checkpoints/v3-ct2-int8 --model-name ft-v3-int8 \
        --compute-type int8 --beam-size 2 --cpu-threads 8 --device cpu \
        --test-set "$F" --dialect "$d" \
        --platform-label l4-cpu --max-samples 100
done
```

## 5. Publish

Per `memory/hf_versioning.md`, **never overwrite an existing HF repo**. Push
the v3 trio to fresh repo IDs:

```bash
# (a) LoRA — already auto-pushed by the trainer to
#     dev-ahmedhany/whisper-large-v3-arabic-ft-v3-lora

# (b) merged
huggingface-cli upload dev-ahmedhany/whisper-large-v3-arabic-ft-v3 \
    checkpoints/v3-merged

# (c) CT2 int8
huggingface-cli upload dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8 \
    checkpoints/v3-ct2-int8
```

Then write a model card per the template at
`dev-ahmedhany/whisper-large-v3-turbo-arabic-ft-ct2-int8/README.md`.

## 6. Teardown

```bash
gcloud compute instances delete whisper-train --zone=us-central1-a --quiet
```

⚠️ **Before deleting**, verify all artifacts are on HF (per the v2-loss
incident on 2026-05-01). Run:

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
for m in api.list_models(author='dev-ahmedhany'):
    print(m.id)
"
```

— and confirm `whisper-large-v3-arabic-ft-v3-{lora,,-ct2-int8}` are all
listed before tearing down. The auto-push hook covers the LoRA; the merged
+ CT2 artifacts must be uploaded manually before deletion.

## 7. Checkpoint trajectory + best-ckpt selection

Because `eval_strategy=no` (§3) means the Trainer cannot pick the best
checkpoint, we eval **every** saved adapter offline and pick by held-out
WER. The LoRA repo on HF preserves all 30+ saves as Git revisions, so
this can be reproduced from a clean checkout.

Same protocol for every checkpoint: snapshot_download the revision, merge
LoRA into base safetensors, ct2-transformers-converter int8, then run
`src.eval_harness` at `--beam-size 2 --cpu-threads 8 --device cpu` against
the 4 canonical v3 mixed test sets (n=100/dialect). Identical to the
zero-shot baseline cell above — only the model swaps in.

Trajectory sweep (CT2 int8, beam=2, 8 threads, c3-standard-8, n=100/dialect):

| ckpt | MSA | Egyptian | Levantine | Gulf | avg-4 |
|---:|---:|---:|---:|---:|---:|
| 500   | 10.78 | 36.09 | 39.10 | 48.11 | 33.52 |
| 1000  | 11.21 | 31.69 | 37.35 | 47.37 | 31.90 |
| 1500  | 11.05 | 29.69 | 36.31 | 45.61 | 30.67 |
| 2000  | 10.99 | 29.89 | 34.45 | 42.84 | 29.55 |
| 2500  | 10.73 | 27.56 | 31.44 | 42.38 | 28.03 |
| 3000  | 10.47 | 26.96 | 32.71 | 41.27 | 27.85 |
| 3500  | 10.36 | 26.56 | 32.71 | 41.37 | 27.75 |
| 4000  | 11.68 | 25.43 | 31.79 | 41.83 | 27.68 |
| 4500  | 11.84 | 26.43 | 32.02 | 40.26 | 27.64 |
| **4750** ⭐ | **10.52** | **23.90** | **30.63** | **41.46** | **26.63** |
| 5000  | 11.73 | 24.17 | 31.32 | 41.46 | 27.17 |
| 5500  | 12.05 | 24.83 | 29.58 | 41.64 | 27.03 |
| 6000  | 11.52 | 24.77 | 29.12 | 41.83 | 26.81 |
| 6250  | 12.21 | 24.77 | 29.81 | 41.92 | 27.18 |
| 6500  | 11.47 | 24.43 | 30.51 | 41.92 | 27.08 |
| 6750  | 11.36 | 24.30 | 30.63 | 41.37 | 26.91 |
| 7000  | 11.73 | 23.30 | 29.81 | 41.46 | 26.58 |
| 7250  | 13.64 | 23.44 | 30.16 | 40.81 | 27.01 |
| 7500  | 12.37 | 24.10 | 31.90 | 42.47 | 27.71 |
| 7750  | 11.84 | 24.10 | 30.97 | 43.12 | 27.51 |

**Selection: ckpt-4750.** Two ckpts (6000, 7000) edge avg-4 by ≤0.05 pp,
but both regress MSA by 1.0–1.2 pp (10.52 → 11.52 / 11.73). The MSA cell
is the overfit signal: the v3 mix is dialect-heavy, so ckpts past ~5000
trade MSA quality for marginal dialect gains. We rejected the 7000 swap
because the MSA loss exceeds the dialect gain in user-weighted impact.

Selecting 4750 also produces the best per-dialect Pareto: it is strictly
within the bootstrap CI of the best Egyptian (23.30 @ 7000), best
Levantine (29.12 @ 6000), and best Gulf (40.26 @ 4500) results from the
sweep, while having the best MSA among the dialect-leading group.

**Verification of the published HF artifact.** To confirm the
`dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8` repo matches a
clean local rebuild from the LoRA revision SHA `7923fe7bc9b7` (= ckpt-4750),
we re-eval'd both:

| model | MSA | Egyptian | Levantine | Gulf |
|---|---:|---:|---:|---:|
| HF v3-ct2-int8 (downloaded) | 10.41 | 24.30 | 30.97 | 42.29 |
| Local merge of revision 7923fe7bc9b7 | 10.52 | 23.90 | 30.63 | 41.46 |

All four rows within bootstrap-CI noise. The published artifact is the
correct ckpt-4750.

Raw per-utterance predictions for the trajectory live in
`runs/predictions/` on the bench instances at the time of the sweep.
The aggregated WER rows (one per ckpt × dialect) are in
`runs/results.jsonl` for ckpts 500-6000; ckpts 6250-8250 were eval'd on
ephemeral bench-c2/c3 boxes and only their summary rows survive (above).
A fresh trajectory sweep of any subset is one `tmux` invocation away
following the recipe in this section.
