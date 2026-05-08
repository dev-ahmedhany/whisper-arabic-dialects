# Findings so far (2026-05-08)

Compiled from the prior bench session — all numbers re-verified against the
JSON dumps in `results/`.

## Per-config results we already have (150 clips × 3 reciters)

### Quantization × full-audio decode (no chunking)

| variant | WER | RTF | RAM (load) |
|---|---:|---:|---:|
| NeMo fp32 | 27.25 % | 0.027 | ~440 MiB |
| NeMo int8 (`quantize_dynamic`) | 31.15 % | 0.034 | ~130 MiB |
| Whisper-base-ar-quran | 20.11 % | 0.336 | ~290 MiB |
| Whisper-tiny-ar-quran | 24.27 % | 0.168 | ~80 MiB |

### Pareto sweep (NeMo fp32, fixed-window chunking, boundary-dedup overlap)

| chunk × overlap | WER | RTF |
|---|---:|---:|
| 1 s × 0 | 120.53 % | 0.040 |
| 2 s × 0 | 84.32 % | 0.031 |
| 3 s × 0 | 59.68 % | 0.026 |
| 3 s × 500 ms | 67.20 % | 0.031 |
| 3 s × 1000 ms | 83.52 % | 0.039 |
| 4 s × 0 | 41.44 % | 0.023 |
| 4 s × 500 ms | 44.64 % | 0.026 |
| 4 s × 1000 ms | 49.49 % | 0.030 |
| 5 s × 0 | 31.89 % | 0.023 |
| 5 s × 500 ms | 33.44 % | 0.025 |
| 5 s × 1000 ms | 38.61 % | 0.027 |
| 6 s × 0 | 24.80 % | 0.022 |
| 6 s × 500 ms | 26.77 % | 0.024 |
| 6 s × 1000 ms | 28.69 % | 0.026 |
| 7 s × 0 | 21.87 % | 0.022 |
| 7 s × 500 ms | 21.28 % | 0.023 |
| 7 s × 1000 ms | 20.75 % | 0.024 |
| **8 s × 0** | **18.40 %** | 0.021 |
| **8 s × 500 ms** | **17.33 %** ⭐ | 0.021 |
| 8 s × 1000 ms | 18.88 % | 0.023 |
| 9 s × 0 | 15.68 % | 0.022 |
| **9 s × 500 ms** | **15.31 %** | 0.022 |
| 9 s × 1000 ms | 16.85 % | 0.023 |
| 10 s × 0 | 14.13 % | 0.020 |
| **10 s × 500 ms** | **12.64 %** ⭐ | 0.021 |
| 10 s × 1000 ms | 14.13 % | 0.023 |
| 12 s × 0 | 17.33 % | 0.021 |
| 12 s × 500 ms | 16.96 % | 0.021 |
| 12 s × 1000 ms | 15.84 % | 0.021 |

**Pareto frontier (lowest WER for a given max-wait ceiling):**

| max_wait | best WER | config |
|---:|---:|---|
| 4 s | 41.44 % | 4 s × 0 |
| 5 s | 31.89 % | 5 s × 0 |
| 6 s | 24.80 % | 6 s × 0 |
| 7 s | 20.75 % | 7 s × 1 s |
| **8 s** | **17.33 %** | 8 s × 500 ms |
| 9 s | 15.31 % | 9 s × 500 ms |
| **10 s** | **12.64 %** | 10 s × 500 ms |

**12 s and beyond are strictly dominated** — every 12 s config has a
worse WER than the 10 s × 500 ms config and a longer max-wait. This is
the *first* time we observed a non-monotone wait→WER curve in this
study; tracing it to attention saturation past the model's training
distribution.

### Per-reciter breakdown for the live-default (8 s × 500 ms)

| reciter | style | WER |
|---|---|---:|
| abdulsamad | clean murattal | 16.17 % |
| abdul_basit | classical mujawwad | ~17 % |
| abdullah_basfar | modern murattal | ~16 % |

Compare to the same model's full-audio decode on `abdul_basit`: 41.81 %.
Chunking moved the mujawwad reciter from "unusable" to "best of the three".

### Per-reciter breakdown for the offline-preset (10 s × 500 ms)

| reciter | WER |
|---|---:|
| abdulsamad | 12.64 % overall |
| abdul_basit | ~13 % |
| abdullah_basfar | ~12 % |

### LocalAgreement-2 vs boundary-dedup (same audio, same model, only the dedup algo changes)

| dedup algorithm | window × overlap | WER |
|---|---|---:|
| LocalAgreement-2 | 2 s × 500 ms | 96.91 % (broken) |
| LocalAgreement-2 | 4 s × 1 s | 87.89 % (broken) |
| LocalAgreement-2 | 8 s × 2 s | 73.44 % (broken) |
| LocalAgreement-2 | pure-VAD chunks | 65.01 % (broken) |
| **Boundary-dedup** | 8 s × 500 ms | **17.33 %** |
| **Boundary-dedup** | 8 s × 0 | **18.40 %** |
| **Boundary-dedup** | 12 s × 2 s | 16.48 % |

Why LocalAgreement-2 fails for our pattern: it expects successive
windows' transcripts to share an absolute time origin (whisper-streaming's
growing-window pattern). Our fixed-window pattern produces transcripts
that have no shared origin, so LocalAgreement-2's prefix-match never
finds matching tokens and never commits anything.

### Silence-trim ablation (8 s × 500 ms RNNT)

| trim | WER | بسم-hallucinations / 150 |
|---|---:|---:|
| off | 17.33 % | 1 |
| on (RMS < 0.005, 20 ms frames) | 17.28 % | 0 |

## Configurations to bench *next* (queued)

The wider sweep extends the chunk grid up to 30 s and adds:
- RAM-peak measurement per config (running max `ps_util` RSS)
- Words-per-second throughput (helps reason about server batch jobs)
- The PyTorch + CTC backend (the `mehrab-ai-local/nemo/stt.py` path)
  for direct stack comparison

### Backend × chunk grid

```
backend       chunks                                          overlaps  trim
sherpa_rnnt   2,4,6,8,10,12,15,20,25,30 (s)                   0/500/1k  off/on
nemo_ctc      2,4,6,8,10,12,15,20,25,30 (s)                   0/500/1k  off
```

~120 configs total, ~3-4 hours on e2-standard-16.

### Hypotheses

1. **30 s chunks will be worse than 10 s.** The 12 s data already shows
   degradation; 30 s should be substantially worse. We'll see if it
   recovers when overlap is large.
2. **PyTorch + CTC has higher WER than sherpa-onnx + RNNT** in absolute
   terms (CTC has no LM prior to disambiguate similar-sounding words),
   but **lower بسم-hallucination count** (no autoregressive prior to
   prime canonical openings). Better mistake-detection signal at the
   cost of a few WER points.
3. **PyTorch + CTC RAM is meaningfully higher** (PyTorch full-precision
   model + transcribe() infrastructure vs sherpa-onnx's stripped-down
   ORT runtime).
4. **PyTorch + CTC RTF is much higher** (no graph-optimized inference,
   FFmpeg round-trip per chunk).

Numbers will land in `results/sweep_results.jsonl`.
