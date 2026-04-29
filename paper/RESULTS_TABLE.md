# All Benchmark Results (browsing view)

Auto-generated from `runs/results.jsonl` (76 rows). Re-run `python -m src.render_results_md` to refresh.

_WER + CER reported as `point [95% bootstrap CI]` (n=1000). RTF = compute_seconds / audio_seconds (lower is better). TTFT-p95 is 95th-percentile time-to-first-token in milliseconds._

## Schema

Each section header is `backend · model · compute (beam=B, threads=T, platform)`. The table inside breaks that cell down by dialect.

## Per-cell rows

### ct2-faster-whisper · zero-shot-base · int8  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 90.8 [87.0, 94.9] | 50.02 [46.11, 54.72] | 0.461 | 4166 ms | 0.69 GB |
| gulf | 100 | 85.4 [82.5, 88.8] | 41.58 [38.78, 44.92] | 0.268 | 2845 ms | 0.67 GB |
| levantine | 100 | 75.2 [71.2, 79.2] | 32.84 [30.02, 35.93] | 0.149 | 2088 ms | 0.65 GB |
| maghrebi | 100 | 95.5 [93.9, 97.0] | 51.16 [48.71, 53.72] | 0.382 | 3757 ms | 0.75 GB |
| msa | 100 | 51.2 [47.6, 55.1] | 16.51 [14.83, 18.57] | 0.050 | 617 ms | 0.61 GB |

### ct2-faster-whisper · zero-shot-base · int8_float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 91.2 [87.7, 94.8] | 51.13 [46.97, 55.74] | 0.467 | 3898 ms | 2.68 GB |
| gulf | 100 | 85.4 [82.5, 88.5] | 42.07 [39.08, 45.49] | 0.278 | 2989 ms | 4.11 GB |
| levantine | 100 | 74.5 [70.6, 78.3] | 31.99 [29.52, 34.42] | 0.151 | 2526 ms | 2.33 GB |
| maghrebi | 100 | 95.5 [93.5, 97.3] | 50.79 [48.54, 53.16] | 0.391 | 4320 ms | 2.85 GB |
| msa | 100 | 51.0 [47.4, 54.9] | 16.43 [14.78, 18.44] | 0.048 | 605 ms | 4.11 GB |

### ct2-faster-whisper · zero-shot-large-v3 · int8  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 57.7 [52.5, 63.7] | 28.41 [24.03, 33.68] | 1.759 | 6655 ms | 5.95 GB |
| gulf | 100 | 59.1 [54.4, 63.3] | 24.19 [20.74, 28.15] | 1.373 | 7055 ms | 5.86 GB |
| levantine | 100 | 37.1 [33.1, 41.3] | 12.91 [10.84, 15.25] | 1.276 | 5325 ms | 6.05 GB |
| maghrebi | 100 | 84.7 [80.6, 88.6] | 44.09 [39.92, 48.19] | 1.518 | 12781 ms | 6.30 GB |
| msa | 100 | 8.5 [6.6, 10.4] | 2.50 [1.87, 3.16] | 0.514 | 7531 ms | 3.71 GB |

### ct2-faster-whisper · zero-shot-large-v3 · int8_float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 57.7 [52.4, 63.7] | 28.41 [24.03, 33.81] | 1.713 | 6719 ms | 6.46 GB |
| gulf | 100 | 59.6 [54.7, 64.0] | 24.57 [21.01, 28.60] | 1.272 | 7337 ms | 6.43 GB |
| levantine | 100 | 37.3 [33.3, 41.5] | 13.08 [10.95, 15.43] | 1.294 | 5354 ms | 6.42 GB |
| maghrebi | 100 | 84.6 [80.7, 88.4] | 43.76 [39.84, 47.59] | 1.482 | 11077 ms | 6.41 GB |
| msa | 100 | 8.5 [6.6, 10.4] | 2.50 [1.87, 3.16] | 0.504 | 7459 ms | 4.56 GB |

### ct2-faster-whisper · zero-shot-medium · int8  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 66.0 [60.5, 72.8] | 32.24 [27.51, 38.17] | 1.483 | 14752 ms | 3.59 GB |
| gulf | 100 | 59.8 [55.8, 63.9] | 25.45 [22.34, 29.00] | 1.236 | 9875 ms | 3.78 GB |
| levantine | 100 | 44.6 [40.1, 49.1] | 15.40 [13.31, 17.76] | 0.623 | 3163 ms | 3.20 GB |
| maghrebi | 100 | 86.4 [83.4, 89.4] | 43.84 [40.52, 47.66] | 0.922 | 7459 ms | 5.12 GB |
| msa | 100 | 16.6 [14.3, 19.5] | 4.34 [3.58, 5.20] | 0.301 | 4388 ms | 2.47 GB |

### ct2-faster-whisper · zero-shot-medium · int8_float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 65.7 [60.2, 72.4] | 32.54 [27.71, 38.40] | 1.425 | 13488 ms | 4.09 GB |
| gulf | 100 | 60.2 [56.1, 64.4] | 25.32 [22.42, 28.56] | 1.191 | 9976 ms | 4.23 GB |
| levantine | 100 | 44.6 [40.1, 49.1] | 15.40 [13.31, 17.76] | 0.608 | 3099 ms | 4.88 GB |
| maghrebi | 100 | 86.5 [83.5, 89.6] | 43.90 [40.60, 47.72] | 0.894 | 7376 ms | 5.49 GB |
| msa | 100 | 16.6 [14.3, 19.5] | 4.34 [3.58, 5.20] | 0.303 | 4434 ms | 2.92 GB |

### ct2-faster-whisper · zero-shot-small · int8  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 77.0 [72.0, 82.4] | 38.81 [34.45, 43.68] | 0.514 | 4374 ms | 1.49 GB |
| gulf | 100 | 72.1 [68.0, 76.3] | 30.48 [27.74, 33.74] | 0.367 | 3582 ms | 1.35 GB |
| levantine | 100 | 56.8 [52.8, 61.2] | 21.73 [19.13, 24.23] | 0.270 | 1704 ms | 1.35 GB |
| maghrebi | 100 | 89.1 [86.5, 91.8] | 44.14 [40.93, 47.57] | 0.553 | 6828 ms | 2.04 GB |
| msa | 100 | 27.4 [24.3, 31.0] | 7.95 [6.74, 9.33] | 0.115 | 1643 ms | 1.09 GB |

### ct2-faster-whisper · zero-shot-small · int8_float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 78.2 [73.1, 83.7] | 39.51 [35.12, 44.69] | 0.541 | 4370 ms | 4.49 GB |
| gulf | 100 | 72.8 [68.7, 77.0] | 31.13 [28.17, 34.74] | 0.365 | 3344 ms | 2.73 GB |
| levantine | 100 | 56.6 [52.6, 61.0] | 21.79 [19.21, 24.32] | 0.273 | 1667 ms | 2.96 GB |
| maghrebi | 100 | 88.7 [86.1, 91.3] | 43.84 [40.75, 46.93] | 0.510 | 5956 ms | 4.41 GB |
| msa | 100 | 27.4 [24.3, 31.0] | 7.95 [6.74, 9.33] | 0.116 | 1671 ms | 2.28 GB |

### ct2-faster-whisper · zero-shot-tiny · int8  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 94.4 [92.0, 97.0] | 55.66 [51.61, 59.73] | 0.329 | 2748 ms | 0.37 GB |
| gulf | 100 | 89.8 [87.5, 92.1] | 46.62 [43.88, 49.82] | 0.244 | 2330 ms | 0.43 GB |
| levantine | 100 | 84.0 [80.6, 87.4] | 39.10 [36.55, 41.73] | 0.157 | 1608 ms | 0.37 GB |
| maghrebi | 100 | 97.1 [95.5, 99.2] | 53.69 [51.10, 56.49] | 0.269 | 2188 ms | 0.43 GB |
| msa | 100 | 66.6 [63.2, 70.7] | 24.34 [22.31, 26.53] | 0.030 | 764 ms | 0.38 GB |

### ct2-faster-whisper · zero-shot-tiny · int8_float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 94.1 [91.6, 97.0] | 56.02 [52.53, 60.05] | 0.323 | 2733 ms | 2.37 GB |
| gulf | 100 | 91.0 [88.4, 93.7] | 48.86 [45.63, 52.60] | 0.233 | 2321 ms | 2.89 GB |
| levantine | 100 | 84.9 [81.6, 88.2] | 40.70 [37.44, 44.26] | 0.160 | 1721 ms | 4.48 GB |
| maghrebi | 100 | 96.7 [95.6, 97.8] | 53.40 [51.55, 55.27] | 0.267 | 2212 ms | 2.15 GB |
| msa | 100 | 66.4 [62.9, 70.4] | 24.24 [22.24, 26.39] | 0.028 | 440 ms | 2.99 GB |

### ct2-faster-whisper · zero-shot-turbo · int8  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 65.0 [59.5, 71.7] | 32.31 [28.20, 37.82] | 1.084 | 6138 ms | 1.62 GB |
| gulf | 100 | 61.1 [57.6, 64.9] | 25.43 [22.65, 28.24] | 0.898 | 3785 ms | 2.16 GB |
| levantine | 100 | 40.3 [36.3, 44.4] | 13.02 [11.13, 14.97] | 0.760 | 3444 ms | 1.62 GB |
| maghrebi | 100 | 84.9 [81.6, 88.2] | 44.27 [40.49, 48.38] | 1.039 | 8141 ms | 2.27 GB |
| msa | 100 | 10.4 [8.4, 12.4] | 2.81 [2.19, 3.48] | 0.307 | 3797 ms | 1.59 GB |

### ct2-faster-whisper · zero-shot-turbo · int8_float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| egyptian | 100 | 65.4 [59.9, 72.2] | 32.81 [28.29, 38.42] | 1.039 | 4979 ms | 2.27 GB |
| gulf | 100 | 60.7 [57.2, 64.3] | 25.28 [22.42, 28.27] | 0.854 | 3673 ms | 2.28 GB |
| levantine | 100 | 40.3 [36.3, 44.4] | 13.21 [11.28, 15.30] | 0.778 | 3483 ms | 2.23 GB |
| maghrebi | 100 | 85.6 [82.6, 88.8] | 43.74 [39.81, 47.67] | 0.986 | 6229 ms | 2.33 GB |
| msa | 100 | 10.4 [8.4, 12.4] | 2.81 [2.19, 3.48] | 0.302 | 3714 ms | 2.21 GB |

### hf-transformers · zero-shot-base-hf · float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 49.0 [43.5, 54.0] | 15.66 [13.41, 18.07] | 0.080 | 1173 ms | 0.75 GB |

### hf-transformers · zero-shot-large-v3-hf · float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 8.7 [6.2, 11.4] | 2.39 [1.54, 3.34] | 0.954 | 12670 ms | 7.64 GB |

### hf-transformers · zero-shot-large-v3-turbo-hf · float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 9.6 [6.9, 12.7] | 2.56 [1.62, 3.61] | 0.545 | 6351 ms | 3.68 GB |

### hf-transformers · zero-shot-medium-hf · float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 14.2 [10.9, 17.8] | 4.01 [2.87, 5.44] | 0.543 | 7347 ms | 4.75 GB |

### hf-transformers · zero-shot-small-hf · float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 25.4 [21.6, 29.1] | 6.78 [5.49, 8.12] | 0.196 | 2709 ms | 1.97 GB |

### hf-transformers · zero-shot-tiny-hf · float32  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 64.9 [60.5, 69.3] | 22.42 [20.31, 24.48] | 0.051 | 768 ms | 0.59 GB |

### hf-transformers · zero-shot-turbo-hf · float32  _(beam=1, threads=8, gcp-g2-standard-16-cpu)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 9.6 [6.9, 12.7] | 2.56 [1.62, 3.61] | 0.364 | 4412 ms | 4.11 GB |

### openai-whisper · zero-shot-base-openai · float32  _(beam=1, threads=0, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 47.8 [42.4, 53.0] | 15.73 [13.25, 18.89] | 0.084 | 1210 ms | 0.73 GB |

### openai-whisper · zero-shot-large-v3-openai · float32  _(beam=1, threads=0, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 8.4 [6.1, 10.9] | 2.32 [1.51, 3.25] | 1.014 | 13584 ms | 6.72 GB |

### openai-whisper · zero-shot-large-v3-turbo-openai · float32  _(beam=1, threads=0, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 10.3 [7.3, 13.6] | 2.75 [1.76, 3.83] | 0.547 | 6335 ms | 3.67 GB |

### openai-whisper · zero-shot-medium-openai · float32  _(beam=1, threads=0, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 13.8 [10.6, 17.3] | 3.71 [2.77, 4.78] | 0.576 | 7919 ms | 3.57 GB |

### openai-whisper · zero-shot-small-openai · float32  _(beam=1, threads=0, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 25.2 [21.6, 28.6] | 6.76 [5.54, 8.03] | 0.209 | 2981 ms | 1.42 GB |

### openai-whisper · zero-shot-tiny-openai · float32  _(beam=1, threads=0, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 65.8 [61.6, 70.2] | 22.61 [20.59, 24.59] | 0.048 | 705 ms | 0.59 GB |

### whisper.cpp · zero-shot-large-v3-cpp · q5_0  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 8.5 [6.3, 11.0] | 2.37 [1.54, 3.30] | 2.806 | 31714 ms | 1.53 GB |

### whisper.cpp · zero-shot-large-v3-turbo-cpp · q5_0  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 9.8 [6.9, 12.9] | 2.56 [1.64, 3.61] | 2.352 | 25716 ms | 0.80 GB |

### whisper.cpp · zero-shot-medium-cpp · q5_0  _(beam=1, threads=4, gcp-c3-standard-8)_

| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |
|---|---|---|---|---|---|---|
| msa | 50 | 14.6 [11.3, 18.1] | 3.92 [2.93, 4.98] | 1.804 | 20441 ms | 0.89 GB |

## Summary by (backend × model) — averaged across dialects

| backend | model | n_cells | avg WER | avg RTF | avg TTFT_p95 | avg RAM |
|---|---|---|---|---|---|---|
| ct2-faster-whisper | zero-shot-base | 5 | 79.6% | 0.265 | 2781 ms | 1.94 GB |
| ct2-faster-whisper | zero-shot-large-v3 | 5 | 49.5% | 1.271 | 7729 ms | 5.82 GB |
| ct2-faster-whisper | zero-shot-medium | 5 | 54.7% | 0.899 | 7801 ms | 3.98 GB |
| ct2-faster-whisper | zero-shot-small | 5 | 64.6% | 0.362 | 3514 ms | 2.42 GB |
| ct2-faster-whisper | zero-shot-tiny | 5 | 86.5% | 0.204 | 1907 ms | 1.69 GB |
| ct2-faster-whisper | zero-shot-turbo | 5 | 52.4% | 0.805 | 4738 ms | 2.06 GB |
| hf-transformers | zero-shot-base-hf | 1 | 49.0% | 0.080 | 1173 ms | 0.75 GB |
| hf-transformers | zero-shot-large-v3-hf | 1 | 8.7% | 0.954 | 12670 ms | 7.64 GB |
| hf-transformers | zero-shot-large-v3-turbo-hf | 1 | 9.6% | 0.545 | 6351 ms | 3.68 GB |
| hf-transformers | zero-shot-medium-hf | 1 | 14.2% | 0.543 | 7347 ms | 4.75 GB |
| hf-transformers | zero-shot-small-hf | 1 | 25.4% | 0.196 | 2709 ms | 1.97 GB |
| hf-transformers | zero-shot-tiny-hf | 1 | 64.9% | 0.051 | 768 ms | 0.59 GB |
| hf-transformers | zero-shot-turbo-hf | 1 | 9.6% | 0.364 | 4412 ms | 4.11 GB |
| openai-whisper | zero-shot-base-openai | 1 | 47.8% | 0.084 | 1210 ms | 0.73 GB |
| openai-whisper | zero-shot-large-v3-openai | 1 | 8.4% | 1.014 | 13584 ms | 6.72 GB |
| openai-whisper | zero-shot-large-v3-turbo-openai | 1 | 10.3% | 0.547 | 6335 ms | 3.67 GB |
| openai-whisper | zero-shot-medium-openai | 1 | 13.8% | 0.576 | 7919 ms | 3.57 GB |
| openai-whisper | zero-shot-small-openai | 1 | 25.2% | 0.209 | 2981 ms | 1.42 GB |
| openai-whisper | zero-shot-tiny-openai | 1 | 65.8% | 0.048 | 705 ms | 0.59 GB |
| whisper.cpp | zero-shot-large-v3-cpp | 1 | 8.5% | 2.806 | 31714 ms | 1.53 GB |
| whisper.cpp | zero-shot-large-v3-turbo-cpp | 1 | 9.8% | 2.352 | 25716 ms | 0.80 GB |
| whisper.cpp | zero-shot-medium-cpp | 1 | 14.6% | 1.804 | 20441 ms | 0.89 GB |
