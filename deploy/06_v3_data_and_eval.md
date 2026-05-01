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
