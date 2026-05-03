# whisper-arabic-dialects

Production-aware multi-dialect Arabic ASR via parameter-efficient fine-tuning of Whisper variants. Code, paper, and reproducible benchmarking pipeline accompanying the paper *Production-Aware Fine-Tuning of Whisper Variants for Multi-Dialect Arabic ASR: A Cross-Platform CPU Inference Study*.

The paper draft is in [`paper/paper.md`](paper/paper.md). Deployment runbooks for GCP training and CPU benchmarking on GCP + Hetzner are in [`deploy/`](deploy/).

## Released artifacts

**v3 large-v3 fine-tune** (paper headline, Apr 2026):
- 🤗 [LoRA adapter](https://huggingface.co/dev-ahmedhany/whisper-large-v3-arabic-ft-v3-lora) — every save during training (30+ Git revisions for trajectory analysis)
- 🤗 [Merged HF model](https://huggingface.co/dev-ahmedhany/whisper-large-v3-arabic-ft-v3) (~3 GB safetensors, transformers-ready)
- 🤗 [CTranslate2 int8](https://huggingface.co/dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8) (~1.6 GB, production-deployable on CPU)

**v2 turbo fine-tune** (smaller/faster, paper §6.5):
- 🤗 [LoRA adapter](https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft-lora), [Merged](https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft), [CT2 int8](https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft-ct2-int8)

**Other:**
- 🤗 [Interactive demo](https://huggingface.co/spaces/dev-ahmedhany/whisper-arabic-dialects) — Gradio Space, side-by-side vs zero-shot
- 🌐 [Project case study](https://hany.dev/case-studies/whisper-arabic-dialects)

## Headline results — v3 large-v3 fine-tune (clean held-out test, May 2026)

> **Note:** Earlier headline numbers in this README were inflated by a test-set contamination (96 of 728 utterances overlapped with the training pool — 50 MGB-3-train rows reused as Egyptian test, 46 MASC-train rows as Levantine test). Those numbers have been removed. The table below uses **only Casablanca *test* split rows** (built by [`scripts/build_v3_test_clean.py`](scripts/build_v3_test_clean.py), which asserts no overlap against the training JSONL before writing). Postmortem: [`deploy/07_contamination_postmortem.md`](deploy/07_contamination_postmortem.md).

Clean held-out 4-dialect test (n=100/dialect, all conversational Casablanca-test for dialects, FLEURS-test for MSA). Same decoding config across rows: **CT2 int8, beam=2, 8 threads, c3-standard-8** (Sapphire Rapids).

| Dialect | Test source | ZS Whisper-large-v3 | **FT-v3-ckpt-4750 (this work)** | Δ |
|---|---|---:|---:|---:|
| MSA | FLEURS test | **8.35%** | 10.84% | +2.49 |
| Egyptian | Casablanca Egypt test | 50.38% | **40.00%** | **−10.38 pp** ✅ |
| Levantine | Casablanca Jordan test | 37.77% | **30.24%** | **−7.53 pp** ✅ |
| Gulf | Casablanca UAE test | 49.48% | **43.35%** | **−6.13 pp** ✅ |
| **avg-4** | | 36.50% | **31.11%** | **−5.39 pp** ✅ |

**FT-v3 beats zero-shot Whisper-large-v3 by 5.39 pp average** on the contamination-free test. Double-digit gain on Egyptian (−10.38), strong on Levantine (−7.53) and Gulf (−6.13). MSA loses 2.49 pp — known dialect-vs-MSA tradeoff that v4 (planned) addresses with an MSA-rebalanced 74 h+ mix.

### Cross-family comparison (same clean test, May 2026)

| Model | Params | Backend / quant | MSA | Egyptian | Levantine | Gulf | avg-4 |
|---|---:|---|---:|---:|---:|---:|---:|
| ZS Whisper-large-v3-turbo | 0.81B | CT2 int8, c3-standard-8 | 10.20 | 52.09 | 41.49 | 52.81 | 39.15 |
| ZS Whisper-large-v3 | 1.55B | CT2 int8, c3-standard-8 | 8.35 | 50.38 | 37.77 | 49.48 | 36.50 |
| Qwen3-ASR-1.7B | 1.7B | BF16, L4 GPU | 8.67 | 57.53 | 39.73 | 48.54 | 38.62 |
| **FT-v3-ckpt-4750 (this work)** | 1.55B | CT2 int8, c3-standard-8 | 10.84 | **40.00** | **30.24** | **43.35** | **31.11** |

FT-v3-ckpt-4750 is the lowest-WER row on every dialect except MSA. Reproduction in [`deploy/06_v3_data_and_eval.md`](deploy/06_v3_data_and_eval.md) and [`deploy/07_contamination_postmortem.md`](deploy/07_contamination_postmortem.md).

The v2-turbo numbers (paper §6.5) used the same now-deprecated mixed test set and need re-evaluation on the clean splits — those rows have been removed pending re-test.

### Dialects scoped out

- **Maghrebi (Moroccan/Algerian/Tunisian)** — excluded from training and reporting. Whisper has insufficient Maghrebi Arabic in pretraining (84.7% zero-shot WER at large-v3 int8); QLoRA cannot bring it within range of other dialects in this training budget. Paper §3.7.

## Quickstart

```bash
# Local sanity (no GPU, no datasets needed):
pip install -r requirements-bench.txt -e .
pytest tests/

# Tiny zero-shot smoke run (downloads ~50 FLEURS samples):
python -m scripts.run_zero_shot_baseline --tiny
```

## Repo map

| Path | Purpose |
|---|---|
| `src/normalization.py` | Arabic text normalizer, applied identically to every reference and hypothesis. Versioned (`NORMALIZER_VERSION`). |
| `src/eval_harness.py` | Single eval entry point. Loads any `faster_whisper.WhisperModel`, runs CPU inference, logs WER + CER + RTF + memory + bootstrap CI to JSONL. |
| `src/bootstrap_ci.py` | Bootstrap WER/CER confidence intervals (n=1000, seed=42). |
| `src/significance.py` | Wilcoxon signed-rank test on per-utterance WERs. |
| `src/data_prep.py` | Dialect-balanced train/val/test JSONL builder. |
| `src/train.py` | QLoRA fine-tuning recipe (NF4 + r=32 + bf16). |
| `src/convert_ct2.py` | Merge LoRA → CT2 sweep across {fp32, fp16, int8_fp32, int8_fp16, int8}. |
| `src/build_results_tables.py` | Aggregates `runs/results.jsonl` into the six paper tables and updates `paper/paper.md` in place. |
| `scripts/prepare_*.py` | Per-dataset JSONL builders (Common Voice 18 ar, FLEURS, Casablanca, MASC, MGB-3, MGB-5). |
| `scripts/run_zero_shot_baseline.py` | Drives the eval harness across the zero-shot baseline matrix. |
| `scripts/run_benchmark_matrix.py` | Drives the eval harness across a config-defined matrix for any model. |
| `configs/*.yaml` | Training hyperparameters and benchmark axis definitions. |
| `Dockerfile` | CPU benchmark image — same binary on GCP and Hetzner. |
| `paper/paper.md` | The paper draft. Tables filled by `build_results_tables.py`. |

## Reproducing the paper results

Follow the runbooks in order. Every command used to produce the paper's results is captured in one of these — no improvised steps.

1. **`deploy/00_gcp_bootstrap.md`** — install `gcloud`, authenticate, link billing, enable APIs, confirm L4 quota, set up budget alerts.
2. **`deploy/01_dataset_acquisition.md`** — pull the seven Arabic datasets from HuggingFace Hub and assemble the dialect-balanced splits.
3. **`deploy/02_gcp_training.md`** — provision the L4 training instance and run QLoRA on turbo (then large-v3).
4. **`deploy/03_gcp_benchmark.md`** — provision the `c3-standard-8` CPU instance and run the benchmark matrix.
5. **`deploy/04_hetzner_benchmark.md`** — provision a Hetzner CX53 and replay the benchmark for cross-platform validation.
6. **`deploy/05_artifacts_publishing.md`** — push models to HF Hub, make the W&B project public, push code.

Approximate compute spend: ~$120 on GCP (4 fine-tuning runs across small / medium / turbo / large-v3 + benchmarking sweep) plus ~$2 on Hetzner (cross-platform validation).

## Citing

If you use this code, the released models, or results from the accompanying paper, please cite:

```bibtex
@misc{hany2026whisperarabic,
  title        = {Production-Aware Fine-Tuning of Whisper Variants for Multi-Dialect
                  Arabic ASR: A Cross-Platform CPU Inference Study},
  author       = {Hany, Ahmed},
  year         = {2026},
  howpublished = {Preprint, arXiv},
}
```

A `CITATION.cff` is also provided for the GitHub "Cite this repository" button.

## License

Code: Apache-2.0 (`LICENSE`). Paper text under `paper/`: CC-BY-4.0.
