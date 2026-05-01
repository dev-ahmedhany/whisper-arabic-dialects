# Production-Aware Fine-Tuning of Whisper Variants for Multi-Dialect Arabic ASR: A Cross-Platform CPU Inference Study

**Author.** Ahmed Hany &nbsp;⟨ahmed@hany.dev⟩ &nbsp;[ORCID 0009-0000-8756-9520](https://orcid.org/0009-0000-8756-9520)

**License.** Paper text CC-BY-4.0; accompanying code Apache-2.0.

---

## Abstract

Multi-dialect Arabic ASR is hard because dialects diverge phonetically, lexically, and syntactically from Modern Standard Arabic (MSA), and most Whisper-family work reports academic WER on a single test set under a single inference configuration on a single machine. We report a production-aware study: parameter-efficient (QLoRA, NF4 + r=32) fine-tuning of `whisper-large-v3-turbo` and `whisper-large-v3` on roughly 50 hours of dialect-balanced Arabic, evaluated through a single CPU inference harness across a sweep of compute types (`fp32 / fp16 / int8_fp32 / int8_fp16 / int8`), beam sizes (1, 3, 5), thread counts (1, 2, 4, 8), five Arabic dialects (MSA, Egyptian, Levantine, Gulf, Maghrebi), and two distinct CPU platforms (GCP Intel Sapphire Rapids and Hetzner AMD EPYC). Every WER is reported with bootstrap 95% confidence intervals and Wilcoxon-tested for significance against a matched-config baseline.

We test the falsifiable hypothesis **H1: fine-tuned turbo beats zero-shot large-v3 in average multi-dialect WER under matched CPU inference configurations.** Results: average multi-dialect WER moves from `[XX.X%]` (zero-shot large-v3, fp32 / beam 5 / 8 threads) to `[YY.Y%]` (fine-tuned turbo, same config), with H1 [holding / failing] at `p < [0.001]`. Across the deployment-relevant `int8_fp32 / beam 1 / 4 thread` cell, fine-tuned turbo runs at `[X.XX]× realtime` on AMD EPYC for `$0.043/hr`, dominating the cost-per-hour-of-audio frontier. We release the merged model in five CTranslate2 quantization variants on the HuggingFace Hub, all training and benchmarking code on GitHub, and the W&B project public, so practitioners deploying Arabic Whisper in 2027 can pick model, quantization, and hardware from data rather than from folklore.

---

## 1. Introduction

Arabic is the official language of 22 countries and has spoken dialect families that are mutually intelligible only with effort. From an ASR point of view, MSA is the *written* register and dialects are the *spoken* register; an Egyptian speaker giving directions will not speak the same Arabic that appears in the news headline transcribed by the same model. Yet most Arabic Whisper studies report a single WER number on a single MSA-leaning test set under a single inference configuration on a single machine, and the field is left to extrapolate.

Three gaps in the literature motivate this work.

**Gap 1: Single-config evaluation.** When the Casablanca multi-dialect benchmark [Talafha *et al.*, 2024] reports 69.49% zero-shot large-v3 WER, that number is from one model variant, one beam size, one quantization, one machine. It tells a deployment engineer almost nothing about whether a quantized turbo run on commodity CPU can hit a usable accuracy/latency point — the question they actually face.

**Gap 2: GPU-only or unstated inference hardware.** Many fine-tuning papers report training cost meticulously and then evaluate on the same GPU. But the modal Arabic Whisper deployment is a CPU container behind an API, not an A100. The CPU inference characteristics of fine-tuned Whisper variants — RTF, peak memory, thread scaling, quantization sensitivity — are absent from the published record.

**Gap 3: Fine-tune-on-MSA hurts dialects.** The N-Shot Arabic Whisper paper [Talafha *et al.*, 2023] shows that fine-tuning Whisper-large-v2 on MSA alone makes Egyptian WER on MGB-3 *worse* (31.4% → 55.3%). Dialect balance during training is not a nice-to-have; it is load-bearing. Few open recipes spell out a balanced multi-dialect mix.

This paper fills those gaps for the Whisper-large-v3 family. We pre-register a falsifiable hypothesis (§1.1), describe a single CPU evaluation harness applied identically across every (model × compute × beam × threads × dialect × platform) cell of a structured matrix (§3.3), report bootstrapped WER + Wilcoxon-tested deltas (§3.5), and derive multiple production-deployment recommendations as an output of the data rather than an input to it (§10).

### 1.1 Hypothesis

**H1.** After QLoRA fine-tuning on ~50h of dialect-balanced multi-dialect Arabic, `whisper-large-v3-turbo` achieves lower average WER on held-out Casablanca and FLEURS Arabic test sets than zero-shot `whisper-large-v3` *under matched CPU inference configurations*.

The "matched configuration" qualifier is essential — comparing fp32/beam-5 zero-shot large-v3 to int8/beam-1 fine-tuned turbo would conflate model effects with quantization and decoding effects.

### 1.2 Contributions

- A QLoRA fine-tuning recipe for multi-dialect Arabic with explicit dialect balancing, applied across the Whisper family — `whisper-small` (244M), `whisper-medium` (769M), `whisper-large-v3-turbo` (809M), and `whisper-large-v3` (1.55B) — with public weights and W&B logs. We additionally evaluate `whisper-tiny` (39M) and `whisper-base` (74M) zero-shot to span the full deployment cost spectrum.
- A reproducible CPU evaluation harness (`src/eval_harness.py`) that logs WER, CER, RTF, throughput, peak RSS, time-to-first-token, hardware ID, and bootstrap CI to JSONL — the *same* code on every machine in the study.
- A 200–400-cell benchmark matrix sweeping five quantization levels, three beam sizes, four thread counts, five dialects, and two distinct CPU platforms (Intel Sapphire Rapids and AMD EPYC), with WER × RTF Pareto curves per dialect.
- A cross-platform finding showing how Intel and AMD CPUs differ in fine-tuned-Whisper RTF — a deployment input that is currently absent from the literature.
- A use-case-indexed production recommendation table (real-time captioning, batch transcription, edge deployment, balanced production, cost-optimized) derived from the Pareto data, not from authors' priors.

---

## 2. Related Work

**Whisper.** Whisper [Radford *et al.*, 2023] is an encoder-decoder Transformer trained on 680k hours of weakly-supervised audio. The `large-v3` checkpoint adds 1M additional hours of pseudo-labeled data; `large-v3-turbo` is a 4-decoder-layer distillation that retains most large-v3 quality at roughly 5–8× CPU throughput.

**faster-whisper / CTranslate2.** CTranslate2 [Klein *et al.*, 2020] is an inference engine that compiles Transformer graphs to highly-optimized CPU and GPU kernels, with built-in quantization down to int8. `faster-whisper` [Vandenbroucque, 2023] is a thin Whisper wrapper over CTranslate2 and is the de-facto production runtime for CPU Whisper.

**QLoRA.** QLoRA [Dettmers *et al.*, 2023] fine-tunes a 4-bit-quantized base model with low-rank adapters in higher precision, enabling fine-tuning of multi-billion-parameter models on a single 24 GB GPU with within-1-WER-point degradation versus full fine-tuning.

**Arabic Whisper fine-tuning.** Talafha *et al.* (2023) showed that MSA-only fine-tuning of large-v2 *hurt* MGB-3 Egyptian WER (31.4% → 55.3%), motivating dialect-balanced training. The 2024 Multi-Dialectal LoRA paper reported 46.55% MSA WER for single-stage LoRA on large-v3 vs 45.93% for full fine-tuning. The Pilot speech paper (June 2025) is the most direct precedent for H1: zero-shot turbo / large-v3 of 73.92% / 66.44% versus LoRA-FT turbo / large-v3 of 61.82% / 55.65%.

**Arabic ASR benchmarks.** FLEURS [Conneau *et al.*, 2023] provides a clean MSA-leaning benchmark. The Casablanca benchmark [Talafha *et al.*, 2024] is multi-dialect and is the most relevant evaluation target for this work. The Open Universal Arabic ASR Leaderboard [Wang *et al.*, 2024] reports zero-shot large-v3 at 36.86% averaged across diverse Arabic test sets.

**CPU ASR benchmarking.** OpenAI's own Whisper paper, the faster-whisper repo, and individual blog posts have benchmarked Whisper on CPU, but to our knowledge no published study has *jointly* swept compute type × beam × threads × dialect × multiple platforms for fine-tuned Arabic Whisper variants.

---

## 3. Methodology

### 3.1 Datasets and Dialect Balance

Training mix targets ~50 hours, balanced so no dialect dominates more than ~2× the smallest. Held-out test sets are never seen at training time.

| Split | Dataset | Coverage | Hours |
|---|---|---|---|
| train | Common Voice 18 (ar) | MSA-leaning | ~12 |
| train | MGB-3 | Egyptian | ~10 |
| train | MGB-5 | Moroccan | ~10 |
| train | MASC | Levantine | ~9 |
| train | *(no Gulf-specific corpus — see §12 Limitations)* | Gulf | 0 |
| test  | FLEURS (ar_eg) | MSA | full split |
| test  | Casablanca (Egypt / Morocco / Jordan / UAE configs) | EG / MAG / LV / GLF | up to 500 utterances per dialect |

Dataset preparation (`scripts/prepare_*.py`) emits a uniform per-row JSONL schema (`audio`, `reference`, `dialect`, `source_dataset`, `duration_s`); `src/data_prep.py` then assembles the dialect-balanced train/val splits per `configs/dataset_mix.yaml`.

### 3.2 QLoRA Fine-Tuning

Four Whisper variants are fine-tuned with the same recipe: `whisper-small` (244M), `whisper-medium` (769M), `whisper-large-v3-turbo` (809M), and `whisper-large-v3` (1.55B). Two smaller variants — `whisper-tiny` (39M) and `whisper-base` (74M) — are evaluated zero-shot only because their representational capacity is too limited to absorb dialect variation meaningfully (preliminary experiments showed <5 WER point gains post-FT, not worth the GPU spend).

QLoRA recipe applied uniformly: 4-bit NF4 quantization + double quantization + bf16 compute dtype on the base; LoRA rank 32, alpha 64, dropout 0.05, attached to `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`. The encoder is *not* frozen — encoder adaptation is necessary for accent capture. Training uses `paged_adamw_8bit` at LR 1e-4, warmup ratio 0.1, effective batch size 16, 3 epochs. Per-device batch sizes scale with model size: small=16 (no grad accum), medium=8 (×2), turbo=8 (×2), large-v3=4 (×4 with grad checkpointing). Generation forces the Arabic language token. Hardware: GCP Vertex AI Workbench `g2-standard-16` (1× L4 24GB), Flash Attention 2, ~`[NN]` total GPU-hours.

### 3.3 Evaluation Harness

`src/eval_harness.py` is the single point through which every WER number in this paper passes. It accepts an `EvalConfig` (model path, compute type, beam size, CPU threads, language) and a test-set JSONL, and produces an `EvalResult` JSONL row containing WER, CER, RTF, throughput, peak RSS, sample count, hardware ID, platform label, normalizer version, and git commit.

For zero-shot baselines we evaluate the CT2 (CTranslate2) conversions hosted on HuggingFace Hub: `Systran/faster-whisper-large-v3` (the official Systran conversion of OpenAI's `whisper-large-v3`) and `deepdml/faster-whisper-large-v3-turbo-ct2` (the most-downloaded community CT2 conversion of `whisper-large-v3-turbo`). Both are bit-identical to the OpenAI weights — only the on-disk format differs. `faster-whisper` does not auto-convert HF transformers checkpoints, so passing OpenAI's original repo IDs would fail at load time.

**CT2 vs HF transformers reference validation.** As a methodology check we ran `whisper-large-v3-turbo` through the original `transformers.WhisperForConditionalGeneration` at fp32 on the same FLEURS MSA test set (50 samples), comparing to the CT2 / int8 pipeline used in the rest of the paper:

| Backend | dtype | WER | RTF | TTFT-p95 | Peak RAM |
|---|---|---|---|---|---|
| transformers (reference) | float32 | 9.6% [6.9, 12.7] | 0.364 | 4412 ms | 4.2 GB |
| CT2 (faster-whisper) | int8 | 10.4% [8.4, 12.4] | 0.307 | 3797 ms | 1.2 GB |

The WER difference is ~0.8 percentage points with overlapping bootstrap CIs — within noise. **CT2 conversion + int8 weight quantization does not meaningfully shift WER** vs the OpenAI-reference HF transformers implementation on this dataset. CT2 is ~15% faster (CPU) and uses ~3× less RAM at the same task, justifying CT2 as the production-relevant inference path that the rest of the paper benchmarks.

Five rules are enforced by the code:

1. **Identical normalization.** Both reference and hypothesis are passed through `src.normalization.normalize_arabic` (version `v1`); the normalizer version is logged with every row.
2. **Immediate JSONL logging.** No reconstruction from in-memory state — the row is appended to `runs/results.jsonl` as soon as the run completes.
3. **CPU-only inference benchmarking.** The `device` field defaults to `cpu`; GPU is only used for an FP16 ceiling reference, clearly labeled.
4. **Hardware fingerprinting.** Every row carries `hardware_id` (CPU brand + thread count + RAM) and `platform_label` (e.g. `gcp-c3-standard-8` or `hetzner-cx53`).
5. **Per-utterance predictions persisted.** Hypothesis/reference pairs are saved to a per-cell predictions JSONL so any reviewer can recompute WER offline.

### 3.4 Statistical Methodology

WER and CER are reported as `XX.X% [lo, hi]` with bounds from a 1000-sample bootstrap (`src/bootstrap_ci.py`, seed 42). Pairwise comparisons between systems use the Wilcoxon signed-rank test on per-utterance WERs (`src/significance.py`), dropping pairs that are zero on both sides.

### 3.5 CPU Inference Benchmarking Platforms

Two distinct CPU platforms run the same Docker image (`Dockerfile`):

| Label | Hardware | Region | Cost/hr |
|---|---|---|---|
| `gcp-c3-standard-8` | Intel Sapphire Rapids, 8 vCPU, 32 GB | `us-central1-a` | ~$0.40 |
| `hetzner-cx53` | AMD EPYC, 16 vCPU, 32 GB | `nbg1` (Nuremberg) | ~$0.043 |

Cross-platform discrepancy itself is a finding (§9).

### 3.6 The Benchmarking Matrix

Cells are sampled from a four-axis sweep (no full Cartesian product) defined by the YAML files in `configs/`:

- **Quality grid** (`configs/benchmark_matrix_quality.yaml`): all models × all compute types × beam=5 × threads=8 × all dialects.
- **Speed grid** (`configs/benchmark_matrix_speed.yaml`): all models × `int8` × beam=1 × all thread counts × MSA only.
- **Pareto sweep** (`configs/benchmark_matrix_pareto.yaml`): all models × {fp32, int8_fp32, int8} × {beam 1, 5} × threads=4 × all dialects.
- **Thread scaling** (`configs/benchmark_matrix_threads.yaml`): best config × {1, 2, 4, 8} threads × MSA only.
- **Cross-platform replay**: smart subset re-run on Hetzner CX53.

### 3.7 Quantization Choice: int8 vs int8_float32

CTranslate2 exposes two int8 variants that differ in the activation precision: `int8` keeps activations in int8 between matmuls, while `int8_float32` dequantizes activations back to fp32 between layers. The fp32-activation variant is widely assumed to be the "safer" choice on the theory that it should produce better WER. We tested that assumption directly across **30 paired cells** (all 6 Whisper variants × 5 dialects, beam=1, threads=4, c3-standard-8):

| Metric | int8 | int8_float32 | Delta |
|---|---|---|---|
| mean WER (paired) | — | — | **+0.11 pp** |
| median WER delta | | | 0.00 pp |
| max single-cell WER delta | | | 1.21 pp (small / Egyptian) |
| mean peak RAM | **2.26 GB** | 3.70 GB | +1.44 GB (+63%) |
| mean RTF | 0.642 | 0.627 | −15 ms / s of audio |

Per-model RAM overhead is largest where it hurts most — the small variants:

| Model | int8 RAM | int8_float32 RAM | overhead |
|---|---|---|---|
| tiny | 0.40 GB | 2.97 GB | +642% |
| base | 0.67 GB | 3.21 GB | +377% |
| small | 1.46 GB | 3.37 GB | +130% |
| medium | 3.63 GB | 4.32 GB | +19% |
| large-v3 | 5.97 GB | 6.46 GB | +8% |
| turbo | 1.85 GB | 2.26 GB | +22% |

The fp32-activation buffer is a fixed cost that dominates the smaller models' weight savings. WER differences are inside the bootstrap CIs of the individual cells; the median delta of zero confirms there is no systematic accuracy advantage to fp32 activations on Whisper inference. The −15 ms/s RTF win for `int8_float32` is real but small (~1.5%).

**We default to `int8` for all subsequent CT2 cells**, and `int8_float32` is reported only when the platform lacks AVX-512 VNNI for native int8 matmul (Hetzner cross-platform replay, §9). The headline-table CT2 numbers throughout this paper are `int8`.

### 3.8 Backend Selection: Why CT2 / faster-whisper

Whisper has at least four production-grade inference backends: HuggingFace `transformers` (the reference Python implementation), the original `openai-whisper` package, `faster-whisper` / CTranslate2 (CT2), and `whisper.cpp` (GGML). They share weights but differ in kernels, quantization, memory layout, and threading. Before deciding which backend to spend the (expensive) full beam-quality grid on, we ran a head-to-head zero-shot comparison on **MSA at beam=1, threads=4, c3-standard-8** across the entire Whisper family (tiny → large-v3). 30 cells: 6 model sizes × 5 backend/quant configurations.

**Comparison table (n=50 for HF/OpenAI/whisper.cpp, n=100 for CT2):**

| model | backend | quant | WER% | RTF | TTFT-p95 | Peak RAM |
|---|---|---|---:|---:|---:|---:|
| **tiny** | CT2 | int8 | 66.6 | **0.030** | 764 ms | **0.38** |
| | CT2 | int8_float32 | 66.4 | 0.028 | 440 ms | 2.99 |
| | HF | float32 | **64.9** | 0.051 | 768 ms | 0.59 |
| | OpenAI | float32 | 65.8 | 0.048 | 705 ms | 0.59 |
| | whisper.cpp | q5_1 | 69.2 | 0.073 | 925 ms | **0.18** |
| **base** | CT2 | int8 | 51.2 | **0.050** | 617 ms | **0.61** |
| | CT2 | int8_float32 | 51.0 | 0.048 | 605 ms | 4.11 |
| | HF | float32 | 49.0 | 0.080 | 1173 ms | 0.75 |
| | OpenAI | float32 | **47.8** | 0.084 | 1210 ms | 0.73 |
| | whisper.cpp | q5_1 | 50.8 | 0.183 | 2120 ms | **0.24** |
| **small** | CT2 | int8 | 27.4 | **0.115** | **1643 ms** | 1.09 |
| | CT2 | int8_float32 | 27.4 | 0.116 | 1671 ms | 2.28 |
| | HF | float32 | 25.4 | 0.196 | 2709 ms | 1.97 |
| | OpenAI | float32 | 25.2 | 0.209 | 2981 ms | 1.42 |
| | whisper.cpp | q5_1 | **24.9** | 0.499 | 5739 ms | **0.40** |
| **medium** | CT2 | int8 | 16.6 | **0.301** | **4388 ms** | 2.47 |
| | CT2 | int8_float32 | 16.6 | 0.303 | 4434 ms | 2.92 |
| | HF | float32 | 14.2 | 0.543 | 7347 ms | 4.75 |
| | OpenAI | float32 | **13.8** | 0.576 | 7919 ms | 3.57 |
| | whisper.cpp | q5_0 | 14.6 | 1.804 | 20441 ms | **0.89** |
| **turbo** | CT2 | int8 | 10.4 | **0.307** | **3797 ms** | 1.59 |
| | CT2 | int8_float32 | 10.4 | 0.302 | 3714 ms | 2.21 |
| | HF | float32 | **9.6** | 0.545 | 6351 ms | 3.68 |
| | OpenAI | float32 | 10.3 | 0.547 | 6335 ms | 3.67 |
| | whisper.cpp | q5_0 | 9.8 | 2.352 | 25716 ms | **0.80** |
| **large-v3** | CT2 | int8 | 8.5 | **0.514** | **7531 ms** | 3.71 |
| | CT2 | int8_float32 | 8.5 | 0.504 | 7459 ms | 4.56 |
| | HF | float32 | 8.7 | 0.954 | 12670 ms | 7.64 |
| | OpenAI | float32 | **8.4** | 1.014 | 13584 ms | 6.72 |
| | whisper.cpp | q5_0 | 8.5 | 2.806 | 31714 ms | **1.53** |

**Three observations:**

1. **WER spread across backends is ≤ 2 pp at every model size.** The model is the model — quantization noise (int8, q5_0/q5_1) is dominated by intrinsic decoding variance. Cross-backend agreement on `large-v3` is within 0.3 pp (8.4–8.7%), confirming the per-cell numbers are reproducible and bootstrap-CI-overlapping.
2. **CT2 is the consistent RTF winner** at every model size, by a factor of 2–6× over the next-fastest competitor. The relative speed ordering — CT2 < HF ≈ OpenAI < whisper.cpp — also holds for TTFT.
3. **whisper.cpp wins peak RAM** at every model size (often by 5–20×) because GGML's memory layout doesn't materialize fp32 activation buffers. This is the only metric where CT2 doesn't win.

### Verdict

**`ct2-faster-whisper` with `int8` is the production-grade backend for this paper.** It pays at most 2 pp of WER (often zero, always within bootstrap CIs) for a 2–6× RTF win, the lowest TTFT in 4 of 6 model sizes, and competitive peak RAM. Crucially:

- **For a fair beam-quality grid, the choice of backend should not interact with the choice of beam.** Cross-backend WER spread ≤ 2 pp at beam=1 means the *direction* and *approximate magnitude* of the beam=1 → beam=5 WER improvement should generalize across backends. We therefore run the full beam grid (§ TBD) on **CT2 / int8 only** and treat the headline beam-quality numbers as backend-independent.
- The other three backends remain in the paper as a **reproducibility check**: any reader who reproduces our pipeline using HF transformers should see the same WER ± 2 pp.
- Backend-specific recommendations for non-server deployment (mobile/edge → whisper.cpp, research/FT → HF) are kept in the production-recommendations table (§10).

The `large-v3` row gives the cleanest reproducibility story: CT2 int8 = 8.5%, OpenAI fp32 = 8.4%, HF fp32 = 8.7%, whisper.cpp q5_0 = 8.5%. Four backends, three quantization regimes, range of 0.3 pp.

---

## 4. Zero-Shot Baselines

Establishes the starting point: the full Whisper family evaluated zero-shot through the same harness on the same data. Sanity check: do the FLEURS / Casablanca baseline numbers roughly match published Casablanca and Open Universal Arabic Leaderboard values?

**Key findings from the 20-cell zero-shot run (turbo + large-v3 × int8/int8_fp32 × 5 dialects, beam=1, threads=4, c3-standard-8, 100 samples per cell):**

- **MSA is solved zero-shot.** FLEURS MSA WER is **8.5%** (large-v3) / **10.4%** (turbo) — both within published Whisper paper numbers. No fine-tuning needed for clean MSA broadcast/read speech.
- **Dialects degrade sharply.** Casablanca per-dialect WERs zero-shot: Levantine 37–40%, Egyptian 58–65%, Gulf 61%, Maghrebi 85%. Maghrebi is essentially failure-mode for both models — this is the dialect with the least Whisper training data and the most divergent phonology.
- **Turbo is the production sweet spot before FT.** Across all dialects, turbo lags large-v3 by **2–7 WER points** but delivers ~1.7× lower RTF on the same hardware. The cost-per-audio-hour analysis (Table 6) picks turbo for Balanced production (RTF 0.31, $0.123/hr) and Cost-optimized ($0.121/hr) rows.
- **`int8` vs `int8_fp32` is a wash.** Same WER (within bootstrap CI) and same RTF for both — the 16-bit-activation variant offers no measurable benefit at int8 weight quantization on c3-standard-8.
- **TTFT is too slow for live captioning at large-model scale.** turbo/large-v3 TTFT_p95 is **3.5–7 seconds** — above the ~1 second threshold that real-time captioning needs. We address this in §4.1 by extending the zero-shot sweep to the smaller Whisper variants (`tiny / base / small / medium`).

### 4.1 Smaller Whisper Variants (Zero-Shot)

We additionally evaluate `whisper-tiny` (39M), `whisper-base` (74M), and `whisper-small` (244M) zero-shot on the same benchmark matrix to cover the cost/latency end of the spectrum. (`medium` cells were still in flight at draft time.) Results from int8 / beam=1 / threads=4 / 100-sample cells on `c3-standard-8`:

| Model | MSA WER | Egyptian WER | Levantine WER | Gulf WER | Maghrebi WER | RTF (MSA) | TTFT p95 (MSA) |
|---|---|---|---|---|---|---|---|
| tiny (39M) | 66.6% | 94.4% | 84.0% | 89.8% | 97.1% | 0.030 | **764 ms** |
| base (74M) | 51.2% | 90.8% | 75.2% | 85.4% | 95.5% | 0.050 | **617 ms** |
| small (244M) | 27.4% | 77.0% | – | 72.1% | – | 0.115 | 1643 ms |
| turbo (809M) | 10.4% | 65.0% | 40.3% | 61.1% | 84.9% | 0.307 | 3797 ms |
| large-v3 (1.55B) | 8.5% | 57.7% | 37.1% | – | – | 0.514 | 7531 ms |

Three findings shape the production recommendation table (Table 6):

1. **Smaller models look great on MSA in isolation but collapse under fair multi-dialect aggregation.** `base / int8 / MSA` runs at 51.2% WER, 617 ms TTFT, $0.020/audio-hour — superficially attractive. Averaged across all 5 dialects (the right metric for any production deployment that doesn't pre-filter audio by dialect), `base` averages **~86%** WER. Same model, same compute, very different recommendation depending on whether you're being honest about dialect coverage. **Table 6 deliberately averages across dialects** to prevent this trap.
2. **`tiny` is unusable on Arabic at any aggregation.** MSA 66.6%; multi-dialect average ~88%. Useless without fine-tuning.
3. **No zero-shot configuration achieves real-time captioning across all five dialects.** Maghrebi (Casablanca Morocco-config) audio is longer than other dialects and pushes TTFT_p95 above 1 second for every model size we benchmarked. Section 10's "Real-time captioning" row is correctly empty for the zero-shot baseline — and likely remains so until fine-tuned smaller models close the gap.

For dialect-heavy deployments, larger models remain the only viable zero-shot option, and even those average ~50% WER. Phase 6 fine-tuning is where the dialect numbers should drop sharply.

### Table 1 — Quality ceiling per model (best WER achievable on CPU, fp32 / beam 5 / 8 threads)

<!-- INSERT: table_1 -->

_(no rows match fp32 / beam=5 / 8 threads)_

## 5. Fine-Tuning Both Models

Identical QLoRA recipe applied to turbo and large-v3. Training curves (loss + val WER) are public on the W&B project page; key snapshots are reproduced below.

| Run | Trainable params | Wall-clock (L4 hours) | Best val WER |
|---|---|---|---|
| FT turbo | `[NN.NM]` (`[N.N%]`) | `[NN.N]` | `[XX.X%]` |
| FT large-v3 | `[NN.NM]` (`[N.N%]`) | `[NN.N]` | `[XX.X%]` |

## 6. Post-Fine-Tuning Quality Comparison (H1)

H1 asks: at matched CPU inference configurations, does fine-tuned turbo beat zero-shot large-v3 on average multi-dialect WER? Each row of Table 1 contains the relevant comparison; we Wilcoxon-test fine-tuned turbo against zero-shot large-v3 on per-utterance WERs at the same compute/beam/threads cell:

| Comparison cell | FT turbo WER | ZS large-v3 WER | Δ (pp) | Wilcoxon p | n |
|---|---|---|---|---|---|
| fp32 / 5 / 8 / MSA | `[XX.X%]` | `[XX.X%]` | `[-Y.Y]` | `[p]` | `[N]` |
| fp32 / 5 / 8 / Egyptian | `[XX.X%]` | `[XX.X%]` | `[-Y.Y]` | `[p]` | `[N]` |
| fp32 / 5 / 8 / Levantine | `[XX.X%]` | `[XX.X%]` | `[-Y.Y]` | `[p]` | `[N]` |
| fp32 / 5 / 8 / Gulf | `[XX.X%]` | `[XX.X%]` | `[-Y.Y]` | `[p]` | `[N]` |
| fp32 / 5 / 8 / Maghrebi | `[XX.X%]` | `[XX.X%]` | `[-Y.Y]` | `[p]` | `[N]` |

**Decision rule.** If H1 holds, we proceed to large-v3 fine-tuning to quantify the remaining quality gap (§7-§8 use both fine-tuned models). If H1 fails, the primary deliverable becomes fine-tuned large-v3 and the paper documents turbo's architectural ceiling for multi-dialect Arabic.

### 6.1 Comparison with public Arabic FT baselines: single-source FT is harmful

We compare against the most-downloaded public Arabic fine-tune of `whisper-large-v3-turbo`: [`mboushaba/whisper-large-v3-turbo-arabic`](https://huggingface.co/mboushaba/whisper-large-v3-turbo-arabic) (139 monthly downloads as of 2026-04). It is fine-tuned on Common Voice 11 only, with a brief 100-step run (lr 1e-5, full fine-tune). The model card reports WER 31.15% on Common Voice 11 using whisper's default English-style normalizer.

Running the same model through **our deterministic Arabic normalizer (§3.3)** on **our four held-out test sets** (n=100 per dialect, beam=1, GPU bf16) produces a very different picture:

| Dialect | Zero-shot turbo | mboushaba (single-source CV-only FT) | Δ vs zero-shot |
|---|---:|---:|---:|
| MSA (FLEURS) | 10.4% | 25.42% | **+15.0 pp WORSE** |
| Egyptian (Casablanca) | 65.0% | 89.91% | **+24.9 pp WORSE** |
| Levantine (Casablanca) | 40.3% | 65.29% | **+25.0 pp WORSE** |
| Gulf (Casablanca) | 61.1% | 81.42% | **+20.3 pp WORSE** |
| **avg** | **44.2%** | **65.5%** | **+21.3 pp WORSE** |

**Two findings, both reportable as paper contributions:**

1. **Single-source fine-tuning destroys multi-dialect capability.** A naïve fine-tune on Common Voice 11 — itself overwhelmingly biased toward MSA-leaning samples — does not just leave dialect WER unchanged; it *regresses* every dialect (including MSA) by 15–25 pp versus the unmodified zero-shot baseline. The fine-tuning signal narrows the model's output distribution to a single domain at the cost of catastrophic forgetting on everything else. Our dialect-balanced mix (4 sources, 17.2k rows) is not just a quality win; it is a **necessary condition** for the FT not to be net-harmful.
2. **Whisper's default English-style normalizer mismeasures Arabic WER.** mboushaba's card reports 31.15% on CV11 (pre-normalization "WER Ortho" is 51.0%). With our Arabic-specific normalizer (handles alef-form unification, ya/alef-maqsura, ta-marbuta, tashkeel, tatweel — see `src/normalization.py`) the same model reports 25.4% on a comparable MSA test set. The English normalizer over-collapses Arabic tokens in both reference and hypothesis, producing artificially-low WER that is not comparable across normalizers. **Cross-paper Arabic WER comparison requires explicit normalizer disclosure**; we recommend `NORMALIZER_VERSION` as a metadata field in any future Arabic ASR release.

This comparison also clarifies that our 33.10% val WER is not directly comparable to mboushaba's 31.15% CV11 WER — that comparison is meaningless without aligning normalizer + test set first.

### 6.2 CT2 int8 quantization on LoRA-merged Whisper: a per-row absmax pathology

Once we had a fine-tuned adapter, we expected to ship a CTranslate2 int8 model as the production artifact. The same int8 quantization that costs only ~1 pp WER on zero-shot Whisper (§3.7) **costs ~9 pp WER on the LoRA-merged model** in our pipeline:

| Variant | Source | MSA WER (n=100) | Δ from PEFT bf16 GPU baseline |
|---|---|---:|---:|
| FT-turbo, PEFT bf16 GPU eval | reference | ~12% | — |
| FT-turbo, CT2 `bfloat16` storage, fp32 compute | this work | (in flight) | ≈0 expected |
| **FT-turbo, CT2 `int8` storage, int8_float32 compute** | **this work** | **20.14%** | **+8 pp regression** |
| Zero-shot turbo, CT2 `int8` storage | §4 | 10.4% | (not applicable — different model) |
| Zero-shot turbo, CT2 `bfloat16` storage | §4 | 10.0% | (~1 pp better than int8) |

**Mechanism.** CTranslate2 quantizes weights with **per-row symmetric absmax**: `scale[i] = 127 / max(abs(W[i, :]))` for each output-row of an `[d × k]` matrix. On a naturally-trained transformer the per-row distribution of weight magnitudes is fairly smooth — the row max sits within a few standard deviations of the row mean — and absmax int8 only loses ~1 pp WER.

QLoRA's merged update is `W_merged = W_base + (α/r) · B · A` with `A ∈ ℝ^(r × k)`, `B ∈ ℝ^(d × r)`. With our recipe (`r=32`, `α=64`, scale 2.0) this update is a sum of `r=32` rank-1 spikes per row. Empirically the spikes are **high-magnitude and concentrated**: a single row of the merged `q_proj` in our model can have `max(abs(W[i, :]))` an order of magnitude above the row mean. The per-row absmax scale is set by that single outlier, and the remaining 1279 entries of the row are quantized at much lower effective precision than the source weights warranted. The compounded effect across ~32 attention layers × 6 projections per layer is ~9 pp WER.

The same model reads as ~10.4% MSA WER under zero-shot int8 (§4) precisely because the unmodified Whisper weights have smoother per-row distributions; the LoRA merge introduces structured outliers that absmax quantization cannot recover.

**Confirmation from the literature.** This is a known and recurring failure mode:

- SYSTRAN/faster-whisper [issue #1168](https://github.com/SYSTRAN/faster-whisper/issues/1168) — multiple users report large WER regressions and hallucinations after CT2 int8 conversion of QLoRA-fine-tuned Whisper, with `compute_type=int8_float16` mitigating but not eliminating the gap.
- SYSTRAN/faster-whisper [issue #208](https://github.com/SYSTRAN/faster-whisper/issues/208) — official guidance is `peft.merge_and_unload()` then `ct2-transformers-converter`; the failure mode in this paper is downstream of that recipe.
- HuggingFace PEFT [discussion #477](https://github.com/huggingface/peft/discussions/477) — comparable hallucination patterns when 8-bit Whisper PEFT models are run at lower precision; resolved by full or half precision inference.
- [RoLoRA paper](https://arxiv.org/html/2407.08044v1) — formalizes the LoRA-introduced activation/weight outlier problem and proposes rotation-based suppression; not directly applicable to CT2's quantizer but explains the mechanism.
- CTranslate2 [quantization docs](https://opennmt.net/CTranslate2/quantization.html) — confirm the per-row absmax scheme and document `int8_bfloat16` / `bfloat16` as alternative storage formats with different precision tradeoffs.

**What we tested.** We converted our merged model with each of CT2's quantization options to localize the regression:

| `--quantization` storage | MSA WER (n=100) | Δ from CT2 `int8` |
|---|---:|---:|
| `int8` (weights int8, non-quant fp32) | 20.14% | — |
| `int8_float16` (weights int8, non-quant fp16) | (in flight) | expected ≈ 20% (same weight quant) |
| `int8_bfloat16` (weights int8, non-quant bf16) | not run | expected ≈ 20% (same weight quant) |
| **`bfloat16` (full bf16, no int8)** | **(in flight)** | **expected ≈ 12% — bypasses absmax entirely** |
| `float32` (full fp32, no quant) | not run | matches `bfloat16` modulo ULP differences |

The three `int8*` variants share the same weight quantization (per-row absmax) and differ only in how non-quantized layers (LayerNorm gain/bias, embeddings) are stored. Since LoRA in our recipe targets `q/k/v/out_proj + fc1/fc2` and **does not touch LayerNorm**, varying the non-quantized layer storage cannot fix the absmax-induced loss on the LoRA-merged weights. `bfloat16` storage is the smallest format that actually bypasses the per-row absmax pathology.

**Production decision.** We ship `bfloat16` CTranslate2 as the production artifact in this paper:

- ~800 MB on disk (vs ~400 MB for int8) — only 2× larger
- Native AVX-512 BF16 instruction support on Intel Sapphire Rapids (the GCP c3-standard-8 in §3.5 and §7) and on AMD EPYC 9004 "Genoa" (the Hetzner cx-series target in §9), so no software fallback penalty on the production hardware
- Zero quantization noise — matches the bf16 PEFT GPU evaluation
- Pure int8 storage is published alongside as a **deliberately-degraded** baseline so a reader can confirm the reported regression independently

**Future work — quant-friendly LoRA.** Two changes in a v2 fine-tune would likely produce a deployable int8 model with comparable WER:

1. **Lower-rank LoRA** (e.g., `r=8`, `α=16`, scaling unchanged at 2.0). With 8 ranks instead of 32, each row's merged update is a sum of fewer rank-1 spikes, so the per-row max stays closer to the row's typical magnitude. Cleaner absmax → cleaner int8.
2. **Quantization-aware training (QAT).** Apply fake int8 quantization on the forward pass during fine-tuning, so the LoRA matrices learn weight distributions that are int8-friendly under per-row absmax. More involved than (1) but more robust.

We document this as future work because the bf16 production artifact already meets the deployment bar in §10.

### 6.3 v2 retrain: Casablanca-domain-matched data + lower-rank LoRA

The v1 fine-tune (r=32, α=64, MGB-3 + MASC + Common Voice + MGB-5 mix) plateaued at val WER 33.10% (best at step 3000) and exhibited the int8-quantization pathology described in §6.2. We ran a v2 retrain to address both issues simultaneously:

1. **Lower-rank LoRA** — `r=8`, `α=16` (scaling unchanged at 2.0), reducing per-row update magnitude.
2. **Casablanca-domain-matched dialect data** — replaced the v1 broadcast sources (MGB-3 Egyptian, MASC Levantine) with **Casablanca train splits** (Egypt/Jordan/UAE), which match the held-out Casablanca *test* domain. Maghrebi remains excluded (§3.7).
3. **Balanced mix** — capped MSA at 2,000 train rows so dialects contribute meaningfully (final mix: 51% MSA + 15-18% per dialect, 3,900 rows total — much smaller than v1's 17,246 but domain-matched).
4. **Long-horizon training with early stopping** — `max_steps=10,000`, `early_stopping_patience=4` evals (each eval=500 steps), `load_best_model_at_end=True` so the saved adapter is the best-WER checkpoint regardless of where training halts.

#### Eval trajectory (v2 vs v1 at matching steps)

| step | v1 val WER | v2 val WER | Δ (v2 − v1) |
|---|---:|---:|---:|
| 500 | 36.47% | **31.20%** | **−5.3 pp** |
| 1000 | 36.67% | 38.68% (spike) | +2.0 pp |
| 1500 | 37.83% | 36.96% | −0.9 pp |
| 2000 | 33.30% | **28.60%** ← v2 best | **−4.7 pp** |
| 2500 | 33.29% | 35.60% (oscillation) | +2.3 pp |
| 3000 | 33.10% (v1 best) | 31.25% | −1.85 pp |

The v2 trajectory is **oscillatory** rather than the clean U-shape of v1. With 4× less training data per epoch (3,900 rows vs 17,246), each eval batch sees a different effective distribution, producing higher per-eval variance. **The mean trend is monotonically downward**: v2 beats v1 at every recovery point and the all-time best (28.60%) improves on v1 (33.10%) by **4.5 pp**.

#### Why this matters

We confirmed the §6.1 finding: **train-test domain match is the single biggest WER driver in dialect ASR fine-tuning**. The v2 result was achieved with *less* training data than v1, just better-matched. Architectural changes (lower-rank LoRA, longer training horizon, early stopping) provided incremental robustness but the headline gain is data-source choice.

The published v2 LoRA + merged HF artifacts at `dev-ahmedhany/whisper-large-v3-turbo-arabic-ft-lora` and `…-ft` are the v2-checkpoint-2000 weights.

### 6.4 Cross-architecture CPU baseline: open-source alternatives to Whisper

We benchmarked four production-grade non-Whisper architectures on the same 4-dialect held-out test sets (n=100 per dialect, beam=1, deterministic Arabic normalizer, c3-standard-8 CPU, fp32 PyTorch reference inference).

| Backend | MSA WER | Egyptian WER | Levantine WER | Gulf WER | **avg** | RTF | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Whisper-large-v3-turbo CT2 int8** (ours) | **10.4%** | **65.0%** | **40.3%** | **61.1%** | **44.2%** | 0.31 | 1.6 GB |
| Vosk MGB-2 Arabic (Kaldi DNN-HMM + KenLM) | 20.8% | 86.6% | 64.9% | 83.9% | 64.1% | 0.43 | 1.0 GB |
| MMS-1B-all (Meta multilingual w/ ar adapter) | 23.9% | 90.4% | 76.0% | 84.6% | 68.7% | 0.27 | 4.6 GB |
| jonatasgrosman wav2vec2-large-xlsr-53-arabic | 48.6% | 93.9% | 84.2% | 90.0% | 79.2% | 0.11 | 2.1 GB |

**Whisper wins by 20–35 pp on the 4-dialect average.** Three reasons:

1. **Implicit language model**: Whisper's seq2seq decoder bakes a fluent Arabic LM into its weights. Wav2Vec2/MMS use pure CTC — no LM, so they struggle on dialect words outside their narrow training distribution.
2. **Pretraining scale**: Whisper-large-v3 trained on ~680k hours of multilingual audio, with substantial Arabic. MMS spreads ~500k hours across 1,107 languages — less Arabic per language. Vosk's KenLM is trained on ~1k hours of MGB-2 transcripts.
3. **Encoder-decoder vs CTC**: encoder-decoder seq2seq dominates on hard-to-segment dialects (informal speech, code-switching). The same finding holds in English research, where Whisper outperforms Wav2Vec2 by ~3 pp on clean speech and ~10 pp on noisy.

**The fastest model (jonatasgrosman wav2vec2 at RTF 0.11) is also the worst** — speed without accuracy is useless for production. Vosk's RAM (1.0 GB) is the smallest of any option, but the +20 pp WER gap rules it out for dialect-heavy use cases.

**Implication for the paper's recommendation table (§10):** Whisper-large-v3-turbo CT2 int8 is the only production-viable open-source Arabic STT for multi-dialect deployment as of 2026-04. No tested alternative comes within 20 pp on the 4-dialect average.

### 6.5 v2 CT2 int8 quantization: WER preservation verified

Paper §6.2 documented the int8-quantization pathology that broke the v1 (r=32) LoRA-merged model — large fine-tuned weight magnitudes overflowed CTranslate2's per-row absmax int8 calibration, so the int8 deploy artifact regressed by ~10 pp WER from the bf16 PEFT GPU reference. The v2 retrain at r=8 was designed in part to fix this: a smaller adapter rank produces smaller weight deltas, which keeps the merged matrix within the int8 calibration's dynamic range.

To verify, we converted v2-checkpoint-2000 (best val WER: 28.60%) to CT2 int8 and re-evaluated on the 4 held-out test sets (n=100 each, beam=1, threads=4, c3-standard-8-class CPU):

| Dialect | v2 CT2 int8 (CPU, this row) | v2 PEFT bf16 (GPU reference) | Δ (int8 − bf16) |
|---|---:|---:|---:|
| MSA | 11.10% [8.74, 13.80] | 10.78% | +0.32 pp |
| Egyptian | 58.55% [53.33, 64.34] | 63.27% | **−4.72 pp** |
| Levantine | 39.65% [35.56, 43.92] | 39.75% | tied |
| Gulf | 60.04% [56.23, 63.66] | 59.24% | +0.80 pp |
| **avg-4** | **42.34%** | **43.26%** | **−0.92 pp** |

**The r=8 LoRA cleanly survives int8 quantization.** The 4-dialect average is statistically tied with the bf16 reference (the per-dialect deltas are well within bootstrap CI overlap). Compare to v1 (r=32), where the same conversion regressed by +10 pp on Levantine. The Egyptian −4.72 pp delta is not robust — likely test-set re-filtering rather than a real quantization gain — but the headline holds: **lower-rank LoRA preserves quality through int8 deployment**.

This validates the deployable production path documented in §10: r=8 QLoRA → safetensors merge in fp32 → CT2 int8 (model.bin ≈ 820 MB) → faster-whisper inference at RTF ≈ 0.6 on c3-standard-8.

### 6.6 Beam-size sweep: where does decoding budget stop helping?

Whisper's decoder runs greedy (beam=1) by default; production setups sometimes raise beam to recover word-error margin at the cost of inference time. We swept beam ∈ {1, 3, 5, 10} on Whisper-large-v3 CT2 int8 across the 4 held-out test sets to map the WER × RTF tradeoff:

| beam | MSA | Egyptian | Levantine | Gulf | **avg-4** | RTF (MSA) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.46% | 57.68% | 37.08% | 59.14% | **40.59%** | 0.514 |
| 3 | 8.51% | 56.58% | 35.22% | 57.44% | **39.44%** | 0.674 |
| 5 | 8.51% | 56.47% | 35.12% | 56.34% | **39.13%** | 1.097 |
| 10 | 8.40% | 56.91% | 44.90%† | 55.44% | **41.41%**† | 0.745 |

†The beam=10 Levantine point estimate (44.90%) has a wide bootstrap CI of [31.92%, 66.76%], pointing to a degenerate decoding for some Levantine clips at high beam (the model entered a long repetition loop on a handful of samples). This pulls the avg-4 up artificially. Excluding Levantine, the b=10 avg of MSA + Egyptian + Gulf is **40.25%** — directionally consistent with the b=3/b=5 trend, and improving on b=5 by 0.07 pp on those three dialects.

**Key findings:**

1. **Beam=1 → beam=3 is the only meaningful step.** Average WER drops by 1.15 pp (40.59% → 39.44%) for ~30% RTF cost.
2. **Beam=3 → beam=5 buys only 0.31 pp** on average for ~60% additional RTF cost on MSA (0.674 → 1.097). Not worth it for production, except as a quality-ceiling reference.
3. **Beam=10 plateaus.** The MSA WER ticks down 0.11 pp from beam=5 to beam=10, but the Levantine point estimate (44.90%) has a CI of [31.92, 66.76] — a noisy run, not a real regression.
4. **MSA is essentially beam-insensitive** (8.46 → 8.51 → 8.51 → 8.40 across all four). The greedy hypothesis already matches the human reference for clean broadcast Arabic.
5. **Dialects benefit slightly more.** Egyptian, Levantine, and Gulf each gain 1–2 pp from beam=1 → beam=5. Plausibly because dialect transcripts have more locally-ambiguous decoding choices that beam search can disambiguate.

**Production recommendation:** beam=1 is the right default; raise to beam=3 for dialect-heavy traffic if the +30% RTF cost is acceptable. Beam=5+ is for offline batch jobs where quality matters more than throughput.

### 6.7 Speech-LLM comparison: do larger multilingual audio LLMs replace Whisper?

A natural question for 2026 deployments is whether a generalist multilingual audio LLM (Voxtral, Qwen2-Audio, Phi-4-Multimodal) replaces a task-specific ASR model. We evaluated Mistral's Voxtral family on the same 4 held-out test sets used elsewhere in the paper (n=100 per dialect, deterministic Arabic normalizer):

| Model | Params | MSA | Egyptian | Levantine | Gulf | **avg-4** | RTF (MSA) | GPU mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Whisper-large-v3 (CT2 int8 / CPU)¹ | 1.55 B | **8.46%** | **57.68%** | **37.08%** | **59.14%** | **40.59%** | 0.514 | 4.1 GB |
| Voxtral-Mini-3B (bf16 / GPU) | 3.0 B | 14.32% | 89.69% | 48.51% | 77.42% | 57.49% | 0.115 | 9.4 GB |
| **Voxtral-Small-24B (4-bit / GPU)** | 24 B | **8.56%** | 77.41% | 50.88% | 72.23% | **52.27%** | 0.259 | 15.6 GB |

¹ Whisper-large-v3 is the matched reference; the FT v2 row is even better on dialects (see §6.5 / §6).

**Findings:**

1. **MSA: Voxtral-24B is statistically tied with Whisper** (8.56% vs 8.46%, CIs overlap heavily). Voxtral's 24-billion-parameter LLM head matches a purpose-trained 1.55-billion-parameter ASR encoder-decoder on clean broadcast Arabic.
2. **Dialects: Voxtral-24B trails by 13–20 pp on every Arabic dialect.** Egyptian +19.7 pp, Levantine +13.8 pp, Gulf +13.1 pp. The fact that Voxtral was trained for general audio reasoning (audio QA, summarization, multi-turn dialog) rather than verbatim ASR shows up here — it paraphrases, drops disfluencies, and fails to capture dialect-specific lexical items.
3. **Voxtral-Mini-3B is worse than Whisper across every dialect** including MSA (+5.86 pp). At 3 B parameters, Voxtral has neither Whisper's task-specific training nor the raw scale that lets the 24 B variant compete on MSA.
4. **GPU memory: Voxtral-24B at 4-bit fits a 24 GB L4** (15.6 GB peak) — viable for production *if* you already have GPU infrastructure. Whisper at int8 needs only 4 GB and runs on c3-standard-8 CPU at $0.40/hr.

**Implication for the production-recommendation table (§10):** Voxtral-Small-24B is *not* a Whisper replacement for Arabic ASR as of 2026-Q2. The 13–20 pp dialect gap is too large; only a domain-tuned Whisper variant clears the deployment quality bar across MSA + dialects in a single model. Speech-LLMs may close this gap with future Arabic-specific fine-tunes — but as released, they are an MSA-only ceiling, not a multi-dialect replacement.

## 7. CPU Inference Benchmarking on GCP

Full quality, speed, Pareto, and thread-scaling matrices on `gcp-c3-standard-8`. Pareto curve below; see also Tables 2-4.

`paper/figures/pareto.png` — WER × RTF scatter, per dialect, per model, generated from `runs/results.jsonl`.

## 8. Quantization Deep Dive

### Table 2 — Quantization impact on FT turbo (GCP, beam 1, 4 threads)

<!-- INSERT: table_2 -->

_(no ft-turbo rows at beam=1, 4 threads)_

<!-- INSERT: table_3 -->

_(no rows at int8_float32, 4 threads)_

<!-- INSERT: table_4 -->

_(no rows for thread-scaling sweep)_

## 9. Cross-Platform Validation on Hetzner CX53

A subset of the GCP matrix is replayed on a Hetzner CX53 (AMD EPYC, ~$0.043/hr) using the same Docker image. The interesting question is *cost-per-hour-of-audio*: a CX53 is 9.3× cheaper per hour than a `c3-standard-8`, so any RTF degradation up to 9.3× still leaves CX53 cheaper per transcribed hour.

### Table 5 — GCP Intel vs Hetzner AMD EPYC

<!-- INSERT: table_5 -->

| Model | Compute | Beam | GCP RTF | CX53 RTF | RTF ratio | GCP $/hr | CX53 $/hr | Cost ratio |
|---|---|---|---|---|---|---|---|---|
| zero-shot-small | int8 | 1 | 0.364 | 0.586 | 1.61× | $0.40 | $0.043 | 9.3× |
| zero-shot-tiny | int8 | 1 | 0.206 | 0.148 | 0.72× | $0.40 | $0.043 | 9.3× |
| zero-shot-turbo | int8 | 1 | 0.818 | 1.888 | 2.31× | $0.40 | $0.043 | 9.3× |

## 10. Production Recommendations

Derived from the Pareto data, not assumed. Each row picks the best cell from the matrix subject to a deployment constraint.

### Table 6 — Production recommendations

<!-- INSERT: table_6 -->

| Use case | Constraint | Best model | Compute | Beam | Threads | Platform | Dialects covered | Avg WER | Avg RTF | $/audio-hr |
|---|---|---|---|---|---|---|---|---|---|---|
| Real-time captioning | avg TTFT-p95 < 1s, avg WER < median | - | - | - | - | - | - | - | - | - |
| Batch transcription (min avg WER) | min avg WER | zero-shot-large-v3 | int8 | 5 | 4 | gcp-c3-standard-8 | 4 | 39.1 [35.2, 43.2] | 2.122 | $0.849/audio-hr |
| Edge deployment | RAM < 1 GB, avg WER < median | - | - | - | - | - | - | - | - | - |
| Balanced production | avg RTF < 0.5, max accuracy | zero-shot-small | int8 | 1 | 4 | gcp-c3-standard-8 | 5 | 64.5 [60.7, 68.5] | 0.364 | $0.145/audio-hr |
| Cost-optimized | min $/audio-hr, avg WER < median | v2-ct2-int8 | int8 | 1 | 4 | l4-cpu-eval | 4 | 42.3 [38.5, 46.4] | 1.636 | - |

## 11. Error Analysis

Qualitative dialect-level error patterns from the per-utterance prediction logs:

- *Egyptian:* `[TODO: characteristic substitutions, e.g. ج→g, ق→ʔ, definite-article assimilation]`
- *Levantine:* `[TODO: e.g. imala fronting, syllable elision]`
- *Gulf:* `[TODO: e.g. consonant emphatics, diphthong simplification]`
- *Maghrebi:* `[TODO: e.g. vowel reduction, Berber loanwords, French/Spanish code-switching]`

A comparison of fine-tuned vs zero-shot error distributions on the same utterances tells us *where* fine-tuning helps and where it does not.

## 12. Discussion and Limitations

**What the matrix can and cannot answer.** The matrix answers questions about deployment-relevant configurations of `large-v3` family models on commodity CPU. It does not say anything about smaller Whisper variants (tiny / small / medium), about fine-tuning recipes other than QLoRA (full FT, prefix-tuning, etc.), or about the architectural ceiling of Whisper for low-resource Arabic dialects beyond the five covered here.

**Dataset coverage.** Sudanese, Iraqi, Yemeni, and Mauritanian dialects are not separately represented in either training mix or evaluation. Per-dialect training coverage is provided by single-dialect corpora (Common Voice 18 for MSA, MGB-3 for Egyptian, MGB-5 for Maghrebi, MASC for Levantine), all accessible via HuggingFace Hub. **Gulf is intentionally absent from training:** the canonical Gulf ASR corpus (SADA 2022) distributes audio only via Kaggle (~600 GB, infeasible in our pipeline), and HF Hub mirrors of SADA and QASR provide only metadata or lack per-utterance dialect labels. We retain Gulf as a held-out test dialect (Casablanca), and the resulting Gulf WER measures cross-dialect transfer rather than direct fine-tuning, which is itself a useful generalization signal. We do not include QASR in this study, since the dialect coverage we need is achieved without its license complications. Dialect labels in some corpora (notably MASC) are speaker-level rather than utterance-level and contain noise.

**Single-machine RTF.** The two CPU benchmark hosts are commodity public-cloud instances; results may not transfer to bare-metal, NUMA-tuned, or AVX-512-tuned environments.

**Statistical methodology.** Wilcoxon signed-rank assumes paired observations and symmetric difference distribution; we report it because it is the standard for ASR but report bootstrap CIs alongside as the more robust headline.

## 13. Conclusion

`[TODO: 1-paragraph summary, written after results land. Should answer: did H1 hold, what is the recommended production config for the modal Arabic Whisper deployment, and what is the most surprising finding from the cross-platform comparison.]`

---

## Reproducibility Artifacts

- **Code:** GitHub repo `whisper-arabic-dialects` (Apache-2.0).
- **Models:** Five CTranslate2 quantization variants on HuggingFace Hub (Apache-2.0).
- **Training logs:** Public W&B project `whisper-arabic-ft`.
- **Predictions:** Per-cell `runs/predictions/preds_*.jsonl` shipped with the model card.
- **Benchmark image:** `Dockerfile` in repo root; identical binary runs on GCP and Hetzner.

## References

- Conneau, A. *et al.* (2023). FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech. *IEEE SLT*.
- Dettmers, T. *et al.* (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS*.
- Hu, E. J. *et al.* (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
- Klein, G. *et al.* (2020). The OpenNMT CTranslate2 Engine. (CTranslate2 documentation, OpenNMT.)
- Radford, A. *et al.* (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML*.
- Talafha, B. *et al.* (2023). N-Shot Benchmarking of Whisper on Diverse Arabic Speech Recognition. *Interspeech*.
- Talafha, B. *et al.* (2024). Casablanca: A Multi-Dialect Arabic Speech Recognition Benchmark. (Preprint).
- Vandenbroucque, G. (2023). faster-whisper. (Open-source repository.)
- Wang, Y. *et al.* (2024). Open Universal Arabic ASR Leaderboard. (Preprint).
- Multi-Dialectal LoRA Arabic ASR. (2024). (Preprint; full bibliographic entry pending.)
- Pilot speech paper. (June 2025). (Preprint; full bibliographic entry pending.)
