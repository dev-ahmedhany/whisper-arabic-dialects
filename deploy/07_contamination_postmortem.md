# v3 Test Contamination — Postmortem & Clean-Eval Reproduction

## Summary

The original `test_v3_<dialect>_mixed.jsonl` files contained **96 of 728**
(13.2 %) utterances that also appeared in the v2 (and almost certainly v3)
training mix. The leak was concentrated in the broadcast halves:

| Test set | Clean rows (Casablanca-test) | Leaked rows | Leak source |
|---|---:|---:|---|
| Egyptian | 50 | **50** | `ArabicSpeech/MGB-3` *train* split sampled into test |
| Levantine | 50 | **46** of 50 | `pain/MASC` *train* split sampled into test |
| Gulf | 100 | 0 | clean ✅ |
| MSA | 428 | 0 | clean (FLEURS *test*) ✅ |

`/tmp/build_v3_full.py` (lost when the original training L4 was deleted)
sampled 50 random rows from each broadcast dataset for "test" *without*
removing them from the training pool. Since the JSONL audio paths embed
the source dataset's split name (`pain__MASC_train_*.wav`,
`ArabicSpeech__MGB-3_train_*.wav`), the contamination is grep-detectable
post-hoc.

## Quantified impact on the published headline

Re-evaluating every model on the **clean half only** (Casablanca-test +
FLEURS-test rows, dropping every `_train_` row):

| Model | MSA | Egyptian | Levantine | Gulf | clean avg-4 | original (contam) avg-4 |
|---|---:|---:|---:|---:|---:|---:|
| ZS Whisper-large-v3 (CT2 int8) | 8.51 | 44.15 | 39.66 | 52.72 | **36.26** | 34.35 |
| **FT-v3-ckpt-4750** (CT2 int8) | 10.52 | **33.44** | **30.54** | **41.46** | **28.99** | 26.63 |
| Qwen3-ASR-1.7B (BF16 GPU) | 8.67 | 54.52 | 41.71 | 49.22 | 38.53 | — |

Per-dialect Δ vs ZS large-v3 (clean):

| Dialect | clean Δ | contam Δ | revision |
|---|---:|---:|---|
| MSA | +2.01 | +2.01 | unchanged (was always clean) |
| Egyptian | **−10.71 pp** | −14.58 | smaller win, still double-digit |
| Levantine | **−9.12 pp** | −7.07 | *larger* win — leakage was masking it |
| Gulf | **−11.26 pp** | −11.26 | unchanged (was always clean) |
| **avg-4** | **−7.27 pp** | −7.72 | headline narrative intact |

**Best-checkpoint selection still picks ckpt-4750.** Recomputed every
v3 trajectory checkpoint's clean WER from the existing per-utterance
predictions in `runs/predictions/`; ckpt-4750 stays the best avg-4 pick
at 28.99 % clean (next best ckpt-6000 at 29.02 %).

## Reproducing the clean eval

The clean test sets are built by `scripts/build_v3_test_clean.py` —
takes only Casablanca *test* split rows and asserts no overlap against
any provided train pool before writing.

```bash
# Build (100 rows / dialect, seed=42, asserts no train overlap)
python -m scripts.build_v3_test_clean \
    --casa-egyptian  test_sets/casablanca_egyptian_test.jsonl \
    --casa-levantine test_sets/casablanca_levantine_test.jsonl \
    --casa-gulf      test_sets/casablanca_gulf_test.jsonl \
    --train-pool     test_sets/train_v3.jsonl test_sets/train.jsonl \
    --out-dir        test_sets/ \
    --n-per-dialect  100 --seed 42
# writes: test_sets/test_v3_clean_{egyptian,levantine,gulf}.jsonl
# MSA reuses the existing FLEURS test_msa_fleurs_msa_test.jsonl
```

Eval matrix (CT2 int8, beam=2, 8 threads, c3-standard-8):

```bash
for d in msa egyptian levantine gulf; do
    case $d in
        msa) F=test_sets/test_msa_fleurs_msa_test.jsonl ;;
        *)   F=test_sets/test_v3_clean_${d}.jsonl ;;
    esac
    # ZS baseline
    python -m src.eval_harness \
        --model Systran/faster-whisper-large-v3 \
        --model-name zs-large-v3-clean \
        --compute-type int8 --beam-size 2 --cpu-threads 8 --device cpu \
        --test-set "$F" --dialect "$d" \
        --platform-label c3-standard-8-clean --max-samples 100
    # FT-v3 ckpt-4750
    python -m src.eval_harness \
        --model dev-ahmedhany/whisper-large-v3-arabic-ft-v3-ct2-int8 \
        --model-name ft-v3-ckpt-4750-clean \
        --compute-type int8 --beam-size 2 --cpu-threads 8 --device cpu \
        --test-set "$F" --dialect "$d" \
        --platform-label c3-standard-8-clean --max-samples 100
done
```

## What needs updating downstream

1. **Paper §3 methodology** — disclose the contamination + how it was
   discovered + how the corrected numbers were produced
2. **Paper §6 / Headline tables** — every WER number for FT-v3 / ZS
   large-v3 against the v3 mixed test sets must be replaced with the
   clean-half number (see table above). The narrative ("FT beats ZS by
   ~7 pp avg-4 on dialects, MSA tradeoff") is unchanged
3. **HF model card** for `dev-ahmedhany/whisper-large-v3-arabic-ft-v3*`
   — replace headline WER with clean numbers + add the postmortem link
4. **README headline table** — same as HF card
5. **Future v4 train builds** — `scripts/build_v3_test_clean.py` enforces
   the train-pool overlap assertion. The next training mix MUST hold out
   these test rows before sampling for train.
