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

On 150 clips × 3 reciters (clean murattal `abdulsamad`, classical
mujawwad `abdul_basit`, modern murattal `abdullah_basfar`) drawn from
[`tarteel-ai/everyayah`](https://huggingface.co/datasets/tarteel-ai/everyayah)
on `e2-standard-16` (16 vCPU Sapphire Rapids):

| approach | WER | RTF | RAM peak | first-text latency |
|---|---:|---:|---:|---:|
| **NeMo fp32 + 8 s chunks + 500 ms overlap + boundary-dedup** | **17.33 %** | **0.021** | ~450 MiB | 8 s worst case, ~300 ms typical (waqf-pause flush) |
| NeMo fp32 + 10 s chunks + 500 ms overlap | **12.64 %** ⭐ | 0.025 | ~450 MiB | 10 s |
| NeMo fp32 full-audio (no chunking) | 27.25 % | 0.027 | ~450 MiB | full audio length |
| `tarteel-ai/whisper-base-ar-quran` full-audio | 20.11 % | 0.336 | ~290 MiB | full audio length |
| `tarteel-ai/whisper-tiny-ar-quran` full-audio | 24.27 % | 0.168 | ~80 MiB | full audio length |
| NeMo int8 (`quantize_dynamic`) full-audio | 31.15 % | 0.034 | ~130 MiB | full audio length |

**Headline result.** A correctly-chunked, otherwise-untuned NeMo model
**beats** Whisper-base-ar-quran (which was fine-tuned on Quran) by
**2.78 pp at live latency (8 s ceiling)** and **7.47 pp at 10 s ceiling**,
on this Quranic eval. Chunking matters more than fine-tuning for this
model–data combination.

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

### 1. The Pareto frontier (window × overlap × WER)

Sweep on the same 150 clips, fp32 weights, 8 s/500 ms boundary-dedup as
the operating point:

![Pareto](figures/pareto.svg) <!-- TODO: emit svg from results/pareto.json -->

| max_wait | best WER | config |
|---:|---:|---|
| 4 s | 41.44 % | 4000_0 |
| 5 s | 31.89 % | 5000_0 |
| 6 s | 24.80 % | 6000_0 |
| 7 s | 20.75 % | 7000_1000 |
| 8 s | 17.33 % | 8000_500 ⭐ live default |
| 9 s | 15.31 % | 9000_500 |
| 10 s | 12.64 % | 10000_500 ⭐ offline preset |
| 12 s | strictly dominated | (every 12 s config lost to 10 s/500 ms) |

Steepest improvement is in the 5→10 s region (~5 pp per +2 s wait);
returns flatten and reverse past 10 s. We ship 8 s/500 ms as the live
default and 10 s/500 ms as an offline preset.

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
| Pareto frontier 1–12 s windows | ✅ done |
| LocalAgreement-2 vs boundary-dedup | ✅ done |
| Whisper-base / -tiny baselines | ✅ done |
| Quantization (fp32 / int8) | ✅ done |
| Silence-trim ablation | ✅ done |
| **Wider sweep, 1–30 s windows** | ⏳ planned (this VM) |
| **PyTorch + CTC vs sherpa-onnx + RNNT** | ⏳ planned (next VM) |
| **RAM measurements per config** | ⏳ planned |
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
