# whisper-arabic-dialects

Production-aware multi-dialect Arabic ASR via parameter-efficient fine-tuning of Whisper variants. Code, paper, and reproducible benchmarking pipeline accompanying the paper *Production-Aware Fine-Tuning of Whisper Variants for Multi-Dialect Arabic ASR: A Cross-Platform CPU Inference Study*.

The paper draft is in [`paper/paper.md`](paper/paper.md). Deployment runbooks for GCP training and CPU benchmarking on GCP + Hetzner are in [`deploy/`](deploy/).

## Released artifacts

- 🤗 **LoRA adapter** (~111 MB, for further FT): https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft-lora
- 🤗 **Merged HF model** (~3 GB, ready to use with `transformers`): https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft
- 🤗 **Interactive demo** (Gradio Space, side-by-side vs zero-shot): https://huggingface.co/spaces/dev-ahmedhany/whisper-arabic-dialects
- 🌐 **Project case study**: https://hany.dev/case-studies/whisper-arabic-dialects

## Headline results — v2 fine-tune on mixed-domain test sets

Held-out 4-dialect test (n=100/dialect). Egyptian + Levantine are **50% Casablanca conversational + 50% broadcast** (MGB-3, MASC) — designed to reflect both registers a deployed model will see. Gulf is 100% Casablanca (no public broadcast Gulf source); MSA is 100% FLEURS broadcast. Same exact recordings + decoding config used for both rows. Both models: CT2 int8, beam=2, 8 threads, c3-standard-8.

| Dialect | Test composition | Zero-shot turbo | **v2-ft turbo (this work)** | Δ |
|---|---|---:|---:|---:|
| MSA | FLEURS broadcast | 10.20% | 11.42% | +1.22 |
| Egyptian | 50 Casablanca + 50 MGB-3 | 44.61% | **36.09%** | **−8.52 pp** ✅ |
| Levantine | 50 Casablanca + 50 MASC | 41.53% | **40.49%** | −1.04 (tied) |
| Gulf | Casablanca UAE | 59.00% | **53.92%** | **−5.08 pp** ✅ |
| **avg-4** | | **38.84%** | **35.48%** | **−3.35 pp** ✅ |

**v2-ft beats zero-shot turbo by 3.35 pp average** on the mixed-domain test, with dominant gains on Egyptian (−8.52) and Gulf (−5.08). The earlier Casablanca-only test only showed a 0.72 pp average lift; the mixed-domain test reveals the model's real advantage on dialect-diverse traffic.

The v3 large-v3 fine-tune (in training, see [`configs/train_large_v3_r8.yaml`](configs/train_large_v3_r8.yaml)) targets the next quality ceiling and pushes every checkpoint to a fresh HF repo (`dev-ahmedhany/whisper-large-v3-arabic-ft-v3-lora`). See [`deploy/06_v3_data_and_eval.md`](deploy/06_v3_data_and_eval.md) for the full reproduction.

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
