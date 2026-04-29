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

_(both platforms required for cross-platform table)_

## 10. Production Recommendations

Derived from the Pareto data, not assumed. Each row picks the best cell from the matrix subject to a deployment constraint.

### Table 6 — Production recommendations

<!-- INSERT: table_6 -->

| Use case | Constraint | Best model | Compute | Beam | Threads | Platform | Dialects covered | Avg WER | Avg RTF | $/audio-hr |
|---|---|---|---|---|---|---|---|---|---|---|
| Real-time captioning | avg TTFT-p95 < 1s, avg WER < median | - | - | - | - | - | - | - | - | - |
| Batch transcription (min avg WER) | min avg WER | zero-shot-large-v3 | int8 | 1 | 4 | gcp-c3-standard-8 | 5 | 49.4 [45.4, 53.5] | 1.288 | $0.515/audio-hr |
| Edge deployment | RAM < 1 GB, avg WER < median | - | - | - | - | - | - | - | - | - |
| Balanced production | avg RTF < 0.5, max accuracy | zero-shot-small | int8 | 1 | 4 | gcp-c3-standard-8 | 5 | 64.5 [60.7, 68.5] | 0.364 | $0.145/audio-hr |
| Cost-optimized | min $/audio-hr, avg WER < median | zero-shot-turbo | int8 | 1 | 4 | gcp-c3-standard-8 | 5 | 52.3 [48.7, 56.3] | 0.818 | $0.327/audio-hr |

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
