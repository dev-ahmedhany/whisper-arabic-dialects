# whisper-arabic-dialects

Production-aware multi-dialect Arabic ASR via parameter-efficient fine-tuning of Whisper variants. Code, paper, and reproducible benchmarking pipeline accompanying the paper *Production-Aware Fine-Tuning of Whisper Variants for Multi-Dialect Arabic ASR: A Cross-Platform CPU Inference Study*.

The paper draft is in [`paper/paper.md`](paper/paper.md). Deployment runbooks for GCP training and CPU benchmarking on GCP + Hetzner are in [`deploy/`](deploy/).

## Released artifacts

- 🤗 **LoRA adapter** (~111 MB, for further FT): https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft-lora
- 🤗 **Merged HF model** (~3 GB, ready to use with `transformers`): https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft
- 🤗 **Interactive demo** (Gradio Space, side-by-side vs zero-shot): https://huggingface.co/spaces/dev-ahmedhany/whisper-arabic-dialects
- 🌐 **Project case study**: https://hany.dev/case-studies/whisper-arabic-dialects

## Headline results

Held-out test sets (n=100 per dialect, beam=1, deterministic Arabic normalizer in `src/normalization.py`).
Zero-shot is `openai/whisper-large-v3-turbo` (CT2 int8). Fine-tuned is this work, evaluated as PEFT bf16 GPU.

| Dialect | Test source | Zero-shot WER | **Fine-tuned WER** | Δ |
|---|---|---:|---:|---:|
| MSA | FLEURS Arabic | 10.4% | 11.5% | +1.1 pp ⚠️ |
| Egyptian | Casablanca | 65.0% | 62.7% | **−2.3 pp** ✅ |
| Levantine | Casablanca | 40.3% | 51.9% | +11.6 pp ⚠️ |
| Gulf | Casablanca | 61.1% | 58.6% | **−2.5 pp** ✅ |
| **avg (4 dialects)** | | **44.2%** | **46.2%** | +2.0 pp |

Maghrebi is excluded (84.7% zero-shot WER — Whisper has insufficient Moroccan/Algerian Arabic in pretraining for QLoRA to recover within budget; see paper §3 / §6.1).

The val WER during training reaches **33.10%** on a held-out slice of the training-source distribution. The held-out test WER above is higher because the test sets are from **different sources** than training (FLEURS vs Common Voice for MSA, Casablanca vs MGB-3 / MASC for dialects), exposing domain mismatch — itself a paper finding (§6.1).

The v1 fine-tune is **not a strict improvement** over zero-shot. It trades a ~1 pp MSA regression and a substantial Levantine regression for −2.5 pp on Egyptian and Gulf. A v2 with lower-rank LoRA (r=8 α=16) and Casablanca train splits is in progress to address the Levantine regression and the int8 quantization sensitivity (paper §6.2).

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
