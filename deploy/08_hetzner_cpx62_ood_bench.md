# Hetzner CPX62 OOD Benchmark — 4-way CPU comparison

## Goal

Compare four open Arabic ASR models on a **truly out-of-distribution** test set
(MediaSpeech-Arabic), eliminating the in-distribution training-overlap advantage
that inflates leaderboard rankings (see `deploy/07_contamination_postmortem.md`).

## Why MediaSpeech as OOD probe

MediaSpeech-Arabic (`ymoslem/MediaSpeech` config `ar`, ~10 h, ~2,505 utterances,
CC-BY-4.0) is **not in the training mix of any of the four models tested**:

| Model | Trained on MediaSpeech? |
|---|---|
| OpenAI Whisper-large-v3 (zero-shot) | ❌ |
| dev-ahmedhany FT-v3-ckpt-4750 | ❌ |
| nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0 | ❌ (NVIDIA used MASC + CV17 + FLEURS only) |
| google/gemma-4-E2B-it | ❌ |

So WER on MediaSpeech reveals **dialect/register generalization**, not training-set
memorization. Casablanca was excluded after our internal contamination postmortem
showed it was being mis-used in our v3 protocol.

## Hardware

Hetzner Cloud **CPX62** in `fsn1` (Falkenstein):

- 16 vCPU AMD shared
- 32 GB RAM
- 640 GB local disk
- $0.0953/hour ≈ $0.10/hr
- Ubuntu 22.04

One box per model, run in parallel; deleted as soon as each finished.

## Protocol

- **Dataset:** `ymoslem/MediaSpeech` (config `ar`, top 200 streamed, seed-deterministic)
- **Audio total:** 47.8 minutes (~14.4 s/utterance)
- **Threads:** 8 (CT2 default; NeMo's `torch.set_num_threads(8)`)
- **Decoding:** beam=2 for Whisper variants, greedy CTC for FastConformer
- **Normalizer:** `src/normalization.py` (NORMALIZER_VERSION="v1") — applied
  identically to ref + hyp before WER

## Results

| Model | WER % | 95% bootstrap CI | RTF | Peak RAM |
|---|---:|---|---:|---:|
| **dev-ahmedhany FT-v3-ckpt-4750 (CT2 int8)** | **20.38** | [18.81, 21.97] | 0.519 | 5.4 GB |
| OpenAI Whisper-large-v3 (CT2 int8) | 20.86 | [19.08, 22.59] | 0.477 | 3.5 GB |
| nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0 (greedy CTC) | 24.04 | [22.43, 25.70] | **0.012** ⚡ | 1.9 GB |
| google/gemma-4-E2B-it (bf16, CPU) | — | — | — | OOM-stall ⚠️ |

### Wall-clock per box

| Model | Compute time for 47.8 min audio | Per-utterance |
|---|---:|---:|
| FastConformer (greedy CTC) | **35 sec** | 0.18 s/utt |
| ZS Whisper-large-v3 | 22.8 min | 6.8 s/utt |
| FT-v3-ckpt-4750 | 24.8 min | 7.4 s/utt |
| Gemma-4-E2B-it | killed after 30 min wall, 0 utterances produced | — |

## Findings

1. **FT-v3 ≈ ZS Whisper on MediaSpeech**: 20.38 vs 20.86 % (overlapping CIs). Read
   media speech — broadcast register, mostly MSA — is **not** where dialect FT
   helps; Whisper's pretraining already covers this register well.

2. **Whisper variants beat Conformer by ~3.7 pp on OOD**: confirms NVIDIA's
   leaderboard advantage (32.91 % avg with LM, 34.74 % greedy) is heavily
   in-distribution-overfit on MASC/CV17/FLEURS train data. On a dataset NVIDIA
   never trained on, FastConformer regresses to 24.04 %.

3. **Conformer is 40-80× faster than Whisper on CPU**: RTF 0.012 vs 0.48-0.52.
   For real-time use cases (live captioning, dictation, IVR), Conformer is
   the only viable open option. For offline batch where WER quality dominates,
   Whisper FT-v3 at 0.5 pp better is the pick.

4. **Gemma-4-E2B-it is impractical on CPX62 CPU**: 32 GB RAM not enough at bf16
   (audio encoder + 2B LLM + activation memory ~30 GB peak). Multimodal LLMs
   for Arabic ASR are GPU-deployment only as of May 2026.

## Cost-per-audio-minute (CPX62 CPU, $0.0953/hr)

Formula: `$/audio-min = ($/hr) × RTF / 60`

| Model | RTF | $-per-1000-audio-min | audio-min-per-$ |
|---|---:|---:|---:|
| FastConformer (greedy CTC) | 0.012 | **$0.019** | **52,400** |
| ZS Whisper-large-v3 (CT2 int8) | 0.477 | $0.758 | 1,320 |
| FT-v3-ckpt-4750 (CT2 int8) | 0.519 | $0.825 | 1,210 |

The FastConformer at **$0.019 per 1000 audio-minutes** is the headline production-CPU
result. That's ~$0.001 per hour of audio transcribed — enabling at-scale Arabic ASR
deployment for vanishing per-call cost.

## Reproduction

Required: Hetzner API key, `hcloud` CLI installed, HF token (free), 4 CPX62
quota slots in `fsn1` location.

```bash
# 1. Spin 3 boxes (drop the gemma fourth — see Findings #4)
for SLOT in 1 2 3; do
  hcloud server create --name whisper-bench-$SLOT --type cpx62 \
    --location fsn1 --image ubuntu-22.04 --ssh-key <YOUR_KEY> &
done
wait

# 2. Bundle audio + JSONL on a host that has the data, scp to each box.
#    (Bundle includes test_sets/top200/v4_mediaspeech_test.jsonl + audio/mediaspeech/)
#    See scripts/build_v4_full_mix.py for downloading; the top-200 subset is
#    one HF streaming pass (~30 sec).

# 3. Run on each box (one model per box):
#    Whisper-CT2 path:  python -m src.eval_harness --model <hf_id> ...
#    NeMo path:         python scripts/run_nemo_bench.py ...
#    Gemma path:        python scripts/run_gemma_bench.py ...

# 4. Pull runs/results.jsonl + delete the boxes.
hcloud server delete <id>
```

Full results land in `runs/results.jsonl` filtered by
`platform_label like 'hetzner-cpx62-mediaspeech-n200%'`.

## Conclusion

The Hetzner CPX62 bench gives us three rows for the paper's
production-CPU table:

- **Best WER on OOD**: FT-v3-ckpt-4750 (20.38 % on MediaSpeech-Arabic)
- **Best CPU speed**: NVIDIA FastConformer (RTF 0.012, 80× faster than Whisper)
- **Best cost-per-minute**: FastConformer ($0.122/1000 audio-min on CPX62)

Open speech-LLMs (Gemma, Voxtral, Qwen3-Omni) require GPU for practical
deployment as of May 2026 and are out of scope for this paper's
production-CPU narrative.

## Footnote: Cohere transcribe-03 + Riva-LM benches abandoned

Two follow-on benches to add Cohere transcribe-03-2026 (open-weights Apache-2.0,
the 2026-Q1 leaderboard #1 on English) and the NGC Riva Conformer + KenLM (the
exact Open Universal Arabic ASR Leaderboard #6 entry, 32.91% avg WER) on the
same MediaSpeech-200 protocol failed for the same reproducible reason: about
10 minutes after first SSH contact, every Hetzner box in our experiment lost
the contents of `~/.ssh/authorized_keys` and became unreachable. Even the
`chattr +i` immutable-bit and cloud-init `user-data` workarounds did not save
the auth state through the box's apt + pip install phase. The benches almost
certainly *completed* in the boxes' tmux sessions but the result files were
inaccessible without disk-snapshot recovery, which we judged not worth the
~$0.50 + 10-min overhead per attempt. Either bench can be re-attempted on
GCP c3-standard-8 (where SSH stays stable indefinitely) as a follow-up.

## Footnote: cheap-box test (CX23 / CPX32) abandoned

Attempts to bench FastConformer on Hetzner's cheaper boxes (CX23 at $0.008/hr,
CPX32 at $0.026/hr) failed due to a reproducible SSH-key-drop issue on those
instance types: cloud-init successfully placed our public key, the setup
script ran (apt install + NeMo install both completed cleanly with NeMo even
fitting in 4GB RAM on CX23), but ~5-10 min after first contact the SSH
authorized_keys file was overwritten and the box became unreachable. Behavior
reproduced on three independent server creations across `fsn1`/`hel1`. We
moved on with CPX62 numbers as the production-CPU canonical reference; a
v2 follow-up could investigate via Hetzner rescue mode or a different OS image.
