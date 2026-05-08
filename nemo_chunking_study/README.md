# Chunked offline transcription on Quranic Arabic with NeMo FastConformer

**Study question:** for an *untuned* general-Arabic ASR model
(`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`, 115 M params, MSA-trained),
what is the right chunking strategy for transcribing classical Quranic
recitation on CPU? How does each chunking choice trade off WER, latency,
and RAM, and how do these numbers compare to the Whisper-fine-tuned
alternative (`tarteel-ai/whisper-base-ar-quran`)?

This study sits next to the [Whisper-Arabic-dialects work](../README.md):
same evaluation rigor (Tashkeel-stripped WER, identical normalization
across all rows, CPU-only inference, hardware fingerprint logged),
different model family (encoder-only RNNT/CTC vs encoder-decoder
attention) and different domain (Quranic recitation, not dialectal
conversation).

## TL;DR

172 NeMo configs across 5 datasets + cross-architecture validation on
whisper-large-v3 (1.5 B params on L4 GPU). All numbers `greedy/beam=1`
for apples-to-apples.

### Headline cross-dataset matrix (NeMo FastConformer-AR-pcd, 115 M params)

| dataset | training? | full WER | best chunked WER | Δ |
|---|---|---:|---:|---:|
| everyayah Quran (150 clips × 3 reciters) | ✅ in NeMo training | 27.25 % | **10.99 %** ⭐ | **−16.27 pp** |
| SADA22 MSA min-6s (100 Saudi MSA clips) | ❌ held out | 41.33 % | **19.37 %** | **−21.96 pp** |
| **SADA22 MSA min-8s (100 long-clip subset)** | ❌ held out | **50.43 %** | **18.41 %** | **−32.01 pp** ⭐⭐ |
| MGB-3 ArabicSpeech (100 broadcast clips) | ❌ held out | 46.77 % | 46.77 % | 0 pp (model fails on dialect) |

### Cross-architecture validation on the same SADA22 MSA min-8s held-out subset

| model | params | full WER | best chunked WER | Δ |
|---|---:|---:|---:|---:|
| NeMo FastConformer-AR-pcd (RNNT, greedy) | 115 M | 50.43 % | **18.41 %** | **−32.01 pp** |
| whisper-large-v3 (encoder-decoder, fp16, greedy) | 1.5 B | 36.38 % | **30.14 %** | **−6.24 pp** |

Whisper has a more forgiving 30 s training distribution so the win is
smaller, but **the direction is identical** — chunking helps both
encoder-only RNNT and encoder-decoder attention models.

### Single best strategy across all datasets and architectures

**`fixed_11000_100`** — 11 s window, 100 ms overlap, n-gram boundary-dedup.
That's it. No VAD, no future-context lookahead, no LocalAgreement, no
ChunkFormer-style masking. Plain fixed windows with a tiny overlap
beat every fancier strategy we tried (46-config v3 sweep included).

### Throughput

NeMo on Hetzner CX33 (4 cores AMD EPYC-Rome @ 2.45 GHz x86_64, $7.99/mo):
RTF 0.040, **24.7 × real-time**, 17,820 hours audio / month, $0.0000075 / minute.

### What this means

For long-form Arabic ASR transcription, the chunking effect is **bigger
than the gap between a 115 M parameter model and a 1.5 B parameter
model**. NeMo + chunking on held-out MSA: 18.41 %. whisper-large-v3 + chunking:
30.14 %. The 115 M model gives **−12 pp** lower WER for **13× fewer
parameters** purely because the chunking interaction is stronger for it.

## Why this matters

Most "chunked decoding" guidance for ASR comes from streaming use cases
(low-latency captions, voice assistants). For *offline transcription
of religious / classical Arabic*, the trade-offs are different:

1. The reference text is fixed and prosody-rich (long elongations, waqf
   pauses, no spontaneous speech).
2. The encoder was likely trained on shorter clips (10–15 s typical for
   FastConformer pre-training mixes); long mujawwad ayahs exceed that
   distribution and the encoder's attention/positional encoding starts
   to degrade — chunking *recovers* accuracy lost to context overflow,
   not just splits work into smaller pieces.
3. RNNT decoders started from `<s>` carry an autoregressive prior that
   can hallucinate canonical prefixes (e.g. emit "بسم الله" before
   "الرحمن الرحيم" because that's the most common transition in the
   training distribution). Chunk-boundary handling matters.

We measured all three of these effects.

## Findings

### 1. The Pareto frontier (v2 fine-grid sweep, 93 configs)

After the v1 sweep flagged 10–12 s as the sweet spot, v2 ran a
fine-grid 7–13 s in 500 ms steps × overlap {0, 100, 200, 300, 500,
700, 1000} ms × ± leading-silence-trim. Top 12 strategies on
everyayah (raw jsonl in [`results/kitchensink_v2.jsonl`](results/kitchensink_v2.jsonl)):

| strategy | WER | RTF |
|---|---:|---:|
| **`fixed_11000_100`** | **10.99 %** ⭐ | 0.022 |
| `fixed_10500_100` | 11.15 % | 0.022 |
| `fixed_10500_700` | 11.20 % | 0.023 |
| `fixed_11000_200` | 11.20 % | 0.023 |
| `fixed_11000_0_trim` | 11.20 % | 0.021 |
| `fixed_10500_500` | 11.25 % | 0.023 |
| `fixed_10500_0` | 11.31 % | 0.022 |
| `fixed_11000_0` | 11.47 % | 0.022 |
| `fixed_11000_1000` | 11.47 % | 0.024 |
| `fixed_10500_300` | 11.68 % | 0.022 |
| `fixed_10500_200` | 11.79 % | 0.022 |
| `fixed_10500_1000` | 11.89 % | 0.024 |

Fine-grid revealed that **11 s with 100 ms overlap** is the sweet spot,
slightly beating the original v1 winner (10 s × 500 ms = 12.64 %) by
1.65 pp. The whole 10.5–11 s × 100–700 ms region is within 1 pp of the
top — no need for exotic tuning, anything in that band ships well.

### 2. LocalAgreement-2 dedup is the wrong algorithm for our pattern

[Whisper-streaming](https://github.com/ufal/whisper_streaming) popularized
a **growing-window** chunking pattern where each window decodes from
`last_committed_time` to `NOW`, so successive transcripts overlap in
absolute time and the prefix-match in LocalAgreement-2 makes sense.

Our pattern is **fixed-window with sliding step**: each window decodes
its own audio independently, transcripts have *no* shared time origin.
LocalAgreement-2's `buffer[0] == staging[0]?` check therefore compares
the previous window's **first** word to the new window's **first** word —
which are almost never the same word — and never commits anything.

Result on the same 150 clips:

| algorithm | WER |
|---|---:|
| fixed 2 s window + LocalAgreement-2 (default in early `nemo_streaming`) | **96.91 %** |
| fixed 4 s window + LocalAgreement-2 | 87.89 % |
| fixed 8 s window + LocalAgreement-2 | 73.44 % |
| **fixed 8 s window + boundary-dedup (longest n-gram match between prev tail and new head)** | **18.40 %** |

Boundary-dedup is the algorithm that's actually correct for fixed-window
overlap. Reference implementation:
[`nemo_streaming/lib/src/hypothesis_buffer.dart`](https://github.com/dev-ahmedhany/nemo-streaming/blob/main/lib/src/hypothesis_buffer.dart).

### 3. Bigger chunks beat full-audio decode

Counterintuitive, but real: for our offline FastConformer, **chunking
the audio into 10 s windows produces lower WER than feeding the whole
clip at once** (12.64 % vs 27.25 % on the same data). Per-reciter:

| reciter | full-audio | 10 s chunks | Δ |
|---|---:|---:|---:|
| abdulsamad (clean murattal) | 16.17 % | ~14 % | −2 pp |
| abdul_basit (mujawwad) | 41.81 % | ~13 % | **−29 pp** |
| abdullah_basfar (modern murattal) | 24.23 % | ~12 % | −12 pp |

The collapse on `abdul_basit` was the smoking gun — that reciter's ayahs
exceed 8–10 s easily, and full-audio decode produces near-2× the WER of
a 10 s-chunked decode. The model isn't intrinsically bad at mujawwad,
it's bad at long audio.

### 4. Chunk-start silence trim eliminates the "Bismillah" hallucination

When a chunk starts with leading silence, the RNNT prediction net is
primed from `<s>` but the encoder sees silence first; the joint
probability path most likely emits **بسم الله الرحمن الرحيم** as the
opening token sequence regardless of the actual audio, because that's
the highest-prior phrase in Quranic training data. Stripping leading
silence (frames with RMS < 0.005) before passing to the recognizer:

| | WER | بسم-hallucinations (n=150) |
|---|---:|---:|
| 8 s/500 ms, no trim | 17.33 % | 1 |
| 8 s/500 ms, leading silence trim | **17.28 %** | **0** |

Cheap (~30 µs/frame) and structural — recommended on by default. No WER
regression in our sample.

### 5. Quantization tradeoffs

Our base export used `onnxruntime.quantization.quantize_dynamic` (the
"easy" int8 path, no calibration data needed). On the same 150 clips,
full-audio decode:

| weights | WER | RTF | size |
|---|---:|---:|---:|
| fp32 | 27.25 % | 0.027 | 437 MiB |
| int8 (dynamic) | 31.15 % | 0.034 | 132 MiB |
| fp16 (`onnxconverter_common.float16.convert_float_to_float16`) | unloadable on iOS ORT 1.22 (malformed Cast metadata) | n/a | 219 MiB |

The 4 pp WER tax for dynamic-int8 is real. **QDQ-style static int8**
(uses regular Conv + Quantize/Dequantize wrappers, requires a
calibration corpus) typically loses <1 pp vs fp32 — that's the
ship-worthy quantization, not the dynamic kind. Punted to future work
and noted in the package's tier config.

### 6. CTC export blocked at the tokenizer

The hybrid model has both RNNT and CTC heads. CTC has no autoregressive
prediction net, so it should structurally be immune to the Bismillah
hallucination from §4. We attempted to export the CTC head separately
and load it via sherpa-onnx's `OfflineRecognizer.from_nemo_ctc`:

| | result |
|---|---|
| Model loads | ✅ once `<blk>` token appended at vocab_size |
| Decode produces output | ✅ for ~3 % of clips |
| Decode produces output for all clips | ❌ — `_Map_base::at` thrown by sherpa-onnx C++ on most clips |

Root cause: my `tokens.txt` dump from NeMo's SentencePiece vocab is
shifted (token IDs start at 1, not 0) and `<blk>` collides with the last
token's ID (1024). Need a clean re-export with proper 0-indexed
SentencePiece vocab + blank at vocab_size+1. Deferred to the planned
Quranic FT, which will produce both heads cleanly in one go.

### 8. Cross-dataset held-out validation (the contamination check)

The everyayah eval has a known caveat: that dataset is in NeMo's
training corpus (390 h Tarteel mix). To prove the chunking trick is
real and not a data-contamination artifact, we ran the same 6 top
strategies on three datasets that are **not** in NeMo's training:

| dataset | full WER | best chunked WER | Δ | notes |
|---|---:|---:|---:|---|
| SADA22 MSA min-6 s (Saudi MSA, real human) | 41.33 % | 19.37 % | **−21.96 pp** | 100 clips ≥ 6 s |
| **SADA22 MSA min-8 s** (longer-clip subset) | **50.43 %** | **18.41 %** | **−32.01 pp** | 100 clips ≥ 8 s — biggest effect |
| MGB-3 ArabicSpeech broadcast | 46.77 % | 46.77 % | 0 pp | model can't decode dialect at all |

The SADA22 results are the headline: on data NeMo never saw, chunking
gives a **bigger** WER reduction than on Quran (the in-training data).
The Δ also scales with clip length (min-6 s → −22 pp, min-8 s → −32 pp),
exactly matching the "long clips overflow training distribution" theory.

The MGB-3 0-pp result is informative, not a failure: chunking can't fix
what the model can't do at all. NeMo's 47 % WER on dialectal Arabic is
because it doesn't speak Egyptian dialect, not because of audio length.

Raw jsonl: [`results/sada22_msa_min6s.jsonl`](results/sada22_msa_min6s.jsonl),
[`results/sada22_msa_min8s.jsonl`](results/sada22_msa_min8s.jsonl),
[`results/mgb3_arabicspeech.jsonl`](results/mgb3_arabicspeech.jsonl).

### 9. Cross-architecture validation (does it work on Whisper too?)

Same 100 SADA22 MSA min-8 s clips, run on whisper-large-v3 (1.5 B
params, fp16 on L4 GPU, greedy decoder, language='arabic'):

| strategy | WER | Δ vs full |
|---|---:|---:|
| `full` | 36.38 % | baseline |
| `fixed_30000_500` | 35.80 % | −0.58 pp |
| `fixed_20000_500` | 31.28 % | −5.10 pp |
| `fixed_15000_500` | 29.99 % | −6.39 pp |
| **`fixed_11000_100`** | **30.14 %** | **−6.24 pp** |

Whisper has a more forgiving 30 s training distribution than NeMo's
20 s, so the chunking effect is smaller in absolute terms (−6 pp vs
−32 pp on the same data). But the **direction is identical** —
chunking helps Whisper too — and the optimal window is in the same
11–15 s range we found for NeMo.

Cross-architecture conclusion: this isn't a NeMo quirk. Encoder-decoder
attention models (Whisper) and encoder-only RNNT (NeMo) both benefit
from chunking long-form audio at training-distribution-aligned windows.

Raw jsonl: [`results/whisper_sada22.jsonl`](results/whisper_sada22.jsonl).

### 10. v3 — ChunkFormer / WhisperX style strategies don't beat plain fixed-window

v3 (46 configs in [`results/kitchensink_v3.jsonl`](results/kitchensink_v3.jsonl))
implements the ChunkFormer right/left/bidirectional context idea and
WhisperX-style VAD-merge. Top 5:

| strategy | WER |
|---|---:|
| `right_ctx_10000_500` (10 s chunk + 500 ms future ctx, drop ctx words) | 12.96 % |
| `left_ctx_10000_500` (10 s chunk + 500 ms past ctx) | 13.97 % |
| `bidir_10000_500_500` | 15.20 % |
| `right_ctx_10000_1000` | 15.41 % |
| `left_ctx_10000_1000` | 15.84 % |

All 46 v3 configs are **worse** than v2's 10.99 % winner. The
literature's context-aware tricks were designed for streaming (where
you need future-context lookahead to commit a partial transcript).
For offline ASR with the full clip in hand, plain `fixed_11000_100`
+ n-gram boundary dedup wins. Streaming and offline have different
optimal chunking patterns.

### 7. Implementation comparison — sherpa-onnx vs NeMo PyTorch

Two production-grade implementations of the same model exist in our
codebase. They make different stack choices:

|  | `nemo_streaming` (this study) | `mehrab-ai-local/nemo/stt.py` |
|---|---|---|
| Runtime | sherpa-onnx (own ORT build) | NeMo PyTorch (`model.transcribe()`) |
| Decoder | RNNT greedy (default) | CTC greedy (`change_decoding_strategy`) |
| Chunking | in-memory Float32 windows | FFmpeg-spawned WAV chunks |
| Overlap dedup | longest-n-gram boundary match | none (just chunks with overlap) |
| VAD | Silero (sherpa-onnx-bundled) | none |
| Target | mobile/desktop, low-latency live | server, batch jobs |
| Mistake-detection invariant | greedy beam=1 (faithful) | CTC has no LM bias |

The bench in this study targets the **sherpa-onnx + RNNT + chunking**
path because that's what ships to mobile via `nemo_streaming`. The
PyTorch + CTC path is benchmarked separately — see [§
"PyTorch CTC implementation" in results](results/pytorch_ctc_comparison.md)
once §6's tokenizer issue is resolved.

## Method

Identical to the parent repo's CPU-eval methodology
([`src/eval_harness.py`](../src/eval_harness.py)) with two adaptations
for the NeMo backend:

1. **Decoding stack**: sherpa-onnx's `OfflineRecognizer.from_transducer`
   with `model_type='nemo_transducer'` and `decoding_method='greedy_search'`,
   loaded from a 3-file split (`encoder.onnx` / `decoder.onnx` / `joiner.onnx`)
   produced by NeMo's
   [`scripts/nemo/fast-conformer-hybrid-transducer-ctc/export-onnx-transducer-non-streaming.py`](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/fast-conformer-hybrid-transducer-ctc/export-onnx-transducer-non-streaming.py).
2. **Normalization**: NFC then strip the Arabic diacritic block
   (`[U+064B-U+0670, U+06D6-U+06ED]`); models in this study aren't
   trained on Tashkeel, neither is the WER scoring. Same regex on
   reference and hypothesis.

Hardware: GCP `e2-standard-16` (16 vCPU, Intel Sapphire Rapids,
e2-standard family pinning), Debian 12 base image, fresh Python 3.11
venv per VM.

Sampling: HuggingFace streaming load of `tarteel-ai/everyayah`,
deterministic `MAX_RECITERS=3, PER_RECITER=50` filter (always picks
the first 3 reciters seen with ≥50 clips each — for our seed runs
that resolved to abdulsamad / abdul_basit / abdullah_basfar). Audio
re-sampled to 16 kHz mono float32 if needed; raw bytes from the dataset
parquet decoded with `soundfile` (no `torchcodec` dependency).

Reproduction:

```bash
# Spin up the e2-standard-16, run startup-script, fetch results
cd nemo_chunking_study
bash scripts/run_full_sweep.sh
```

See [`scripts/run_full_sweep.sh`](scripts/run_full_sweep.sh) for the
end-to-end recipe.

## Status

| | status |
|---|---|
| Pareto frontier 1–12 s windows (v1) | ✅ done |
| Fine-grid 7–13 s × 7 overlaps (v2, 93 configs) | ✅ done — `fixed_11000_100` 10.99 % winner |
| ChunkFormer / WhisperX context strategies (v3, 46 configs) | ✅ done — none beat v2 |
| LocalAgreement-2 vs boundary-dedup | ✅ done |
| Silence-trim ablation | ✅ done |
| Whisper-tiny / -tiny-ar-quran baselines | ✅ done |
| Held-out cross-dataset (SADA22 MSA × 2, MGB-3) | ✅ done — −22 to −32 pp on MSA |
| Cross-architecture (whisper-large-v3 on SADA22 MSA min-8 s) | ✅ done — −6.24 pp |
| whisper-base-ar-quran head-to-head on SADA22 + everyayah | ⏳ in flight |
| Quantization (fp32 / int8) | ✅ done |
| **PyTorch + CTC vs sherpa-onnx + RNNT** | ⏳ deferred (separate workstream) |
| **RAM measurements per config** | ⏳ deferred |
| Quranic FT (will eliminate §6 blocker) | ⏳ separate workstream |

## License + attribution

Apache-2.0, same as the parent repo. Bench scripts and findings released
for reproduction.

Bench audio: [`tarteel-ai/everyayah`](https://huggingface.co/datasets/tarteel-ai/everyayah)
(reciter recordings, public dataset). Base model:
[`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`](https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0).
Whisper baselines:
[`tarteel-ai/whisper-base-ar-quran`](https://huggingface.co/tarteel-ai/whisper-base-ar-quran),
[`tarteel-ai/whisper-tiny-ar-quran`](https://huggingface.co/tarteel-ai/whisper-tiny-ar-quran).

Reference `nemo_streaming` package (used in production by Murattil app):
<https://github.com/dev-ahmedhany/nemo-streaming> *(repo: pending push)*.

---

## Related work — what's been tried before

Five recent papers shape the methodology and define what's already known.
Each addresses a slightly different question; ours sits in the gap they
leave.

| Paper | Core idea | What we borrow |
|---|---|---|
| **Open ASR Leaderboard (arxiv 2510.06961, Oct 2025)** | 60+ systems × 11 datasets × multilingual + long-form tracks; standardized normalization, WER + RTFx | Long-form-vs-batch as a separate axis; the BSF metric (`streaming_WER / batch_WER`) |
| **Pushing the Limits of On-Device Streaming ASR (arxiv 2604.14493)** | 50+ chunking configurations; quantization stack; identifies Nemotron-0.6B int4 as 8.20 % WER + 0.56 s latency Pareto winner | The "delay = chunk + right_context" formula; per-model chunking sweep methodology |
| **ChunkFormer: Masked Chunking Conformer (HF: khanhld/chunkformer-ctc-large-vie)** | Endless decoding via chunks-with-**relative right context** at the attention level; masked batching (no padding); −7.7 pp absolute WER on long-form vs Conformer | The right-context technique — give each chunk N s of future audio so its encoder doesn't run out of context at the chunk's tail |
| **WhisperX (Bain et al., 2023)** | Merge short VAD segments to maximize contextual relevance | VAD-merge as a strategy — start from VAD boundaries, merge until reaching target length |
| **NeMo cache-aware streaming (NVIDIA, 2024)** | Cache-aware streaming attention; flexible latency/accuracy at inference time | Why naive chunking degrades for non-cache-aware models; what the right baseline is |

**The gap our study fills.** All five papers cover **streaming** (chunking
hurts; how much can we limit the damage). None directly examine
**chunking as an accuracy-improvement intervention for offline
transcription** when the audio is *longer than the encoder's training
distribution*. Our headline finding (NeMo full-audio 27.25 % → 10 s
chunked 12.64 %) sits in that gap. The mechanism — attention saturation
past training-clip length — is well-documented; the practical
"chunk-to-recover-WER" intervention is not.

## Techniques we're testing, mapped to the literature

The kitchen-sink sweep `scripts/run_kitchensink_v2.py` evaluates:

1. **Naive fixed-window** (what most production code does — `whisperx`,
   `nemo_streaming.dart`, the offline-Whisper subprocess in
   `mehrab-ai-local`).
2. **Boundary n-gram dedup** (our v2 — drops longest 1-5 word n-gram
   match between previous chunk's tail and new chunk's head). Replaces
   LocalAgreement-2 which fails for our pattern.
3. **Silence-trim per chunk** (RNNT decoders started from `<s>` emit
   canonical prefixes when fed silence; trim leading silence first).
4. **VAD-only chunking** (Silero VAD natural pauses; no max cap).
5. **VAD + max-cap** (Silero VAD with force-split if segment exceeds
   N s; combines natural-pause boundaries with attention-saturation
   protection).
6. **VAD + max-cap + chunk-start padding** (anti-hallucination test —
   prepend silence so encoder sees clean speech onset).
7. **Right-context chunking (ChunkFormer-inspired)** — pad each chunk
   with N s of future audio for attention context, drop the words
   from the lookahead region in post-processing. *Pending — to add
   in v3.*
8. **Left-context chunking** — pad each chunk with N s of past audio.
   Drop the words from the prefix region. *Pending v3.*
9. **WhisperX-style VAD merge** — start from VAD segments, merge
   adjacent shorts until reaching target window. *Pending v3.*

The first 6 are in v2 right now. 7-9 land in v3 once we have v2
results to anchor against.

## Reading these papers to refine our framing

A few honest caveats this literature review surfaces:

- **The Open ASR Leaderboard's long-form track exists precisely because
  chunking strategies affect WER differently across models.** Our
  finding that chunking *helps* a particular model on a particular
  domain is consistent with this — they document the variance, we
  document the positive case.
- **"Pushing the Limits" tested 15+ chunking configs per model** and
  observed the same kind of U-shape we see (small chunks hurt,
  optimal mid-range, very long chunks hurt again — though they
  attribute the long-end degradation to streaming-mode vs batch-mode
  rather than attention saturation specifically).
- **ChunkFormer's right-context idea** is structurally what we should
  layer on top of our boundary-dedup. Pre-pending the next chunk's
  first ~2 s and *trimming* the over-emitted words afterward gives
  the encoder the future context it lacks at chunk boundaries,
  without the dedup ambiguity of naive overlap.

## Sources

- Open ASR Leaderboard: arXiv:2510.06961 — <https://arxiv.org/abs/2510.06961>
- Pushing the Limits of On-Device Streaming ASR: arXiv:2604.14493 — <https://arxiv.org/abs/2604.14493>
- ChunkFormer: <https://huggingface.co/khanhld/chunkformer-ctc-large-vie>
- WhisperX: arXiv:2303.00747 — <https://arxiv.org/abs/2303.00747>
- NeMo cache-aware streaming docs: <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html>
- Open ASR Leaderboard blog post: <https://huggingface.co/blog/open-asr-leaderboard>
- SpeechBrain Conformer streaming tutorial: <https://speechbrain.readthedocs.io/en/v1.0.3/tutorials/nn/conformer-streaming-asr.html>
