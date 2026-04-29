# 05 — Publishing Public Artifacts

Once results are in, the public deliverables (HF Hub model variants, W&B project, GitHub repo, paper) get published. This runbook walks each.

## 1. HuggingFace Hub — model variants

For each fine-tuned model (turbo, large-v3), publish all five CT2 quantization variants plus the merged HF-format checkpoint and the LoRA adapter.

```bash
huggingface-cli login
HF_USER=$(huggingface-cli whoami | head -n1 | awk '{print $1}')

for q in float32 float16 int8_float32 int8_float16 int8; do
  REPO="${HF_USER}/whisper-large-v3-turbo-ar-${q}"
  huggingface-cli upload "$REPO" "whisper-faster-turbo-ar-${q}" . \
    --repo-type model --commit-message "Initial release"
done

# LoRA adapter (small, can keep separate)
huggingface-cli upload \
  "${HF_USER}/whisper-large-v3-turbo-ar-lora" \
  checkpoints/whisper-turbo-ar-lora .

# Merged HF checkpoint (for users who want to re-quantize)
huggingface-cli upload \
  "${HF_USER}/whisper-large-v3-turbo-ar-merged" \
  whisper-merged .
```

### Model card template (paste into each repo's README.md)

```markdown
# whisper-large-v3-turbo-ar-int8_float32

CTranslate2 int8_float32 quantization of QLoRA-fine-tuned `whisper-large-v3-turbo`
for multi-dialect Arabic ASR. Trained on ~50h of dialect-balanced Arabic
(MSA + Egyptian + Levantine + Gulf + Maghrebi).

## Per-dialect WER (FLEURS + Casablanca held-out, beam=1, 4 threads, c3-standard-8)

| Dialect | WER |
|---|---|
| MSA | XX.X% [lo, hi] |
| Egyptian | XX.X% |
| Levantine | XX.X% |
| Gulf | XX.X% |
| Maghrebi | XX.X% |

## Usage

```python
from faster_whisper import WhisperModel

model = WhisperModel("dev-ahmedhany/whisper-large-v3-turbo-ar-int8_float32",
                     device="cpu", compute_type="int8_float32", cpu_threads=4)
segments, info = model.transcribe("audio.wav", language="ar", beam_size=1)
print(" ".join(s.text for s in segments))
```

## Limitations

Sudanese, Iraqi, Yemeni, Mauritanian dialects are not represented in training data
and not benchmarked. WER is not a complete quality measure for downstream tasks
(diarization, timestamps); see paper §11 Error Analysis for qualitative patterns.

## Citation

[BibTeX entry — see whisper-arabic-dialects repo README]

## License

Apache-2.0.
```

## 2. W&B — public project

In W&B → Project Settings → Privacy → set to "Public". Add the project README (paste the abstract from `paper/paper.md`).

## 3. GitHub — code repo

```bash
git init
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin https://github.com/dev-ahmedhany/whisper-arabic-dialects.git
git push -u origin main
```

Then on GitHub: enable Issues, add the Apache-2.0 license badge to README, link the HF Hub model repos and W&B project.

## 4. Paper — arXiv submission

Convert `paper/paper.md` to LaTeX (or write LaTeX from the start). See "TeX vs Markdown for arXiv" appendix in the project README. Submit to `arXiv.org` under `eess.AS` (Audio & Speech) with cross-list to `cs.CL`.

Include in the arXiv source tarball:
- `paper.tex`, `paper.bib`, all figures (`paper/figures/*.pdf`)
- README.md pointer to the GitHub + HF Hub artifacts
- (Don't include large data files.)

## 5. Reproducibility checklist (for the paper appendix)

- [ ] All training runs in W&B project `whisper-arabic-ft` (public).
- [ ] All five CT2 variants on HF Hub with model cards.
- [ ] LoRA adapter on HF Hub.
- [ ] `runs/results.jsonl` and `runs/predictions/` shipped as a HF dataset.
- [ ] `Dockerfile` builds with no external deps beyond pip.
- [ ] `git rev-parse HEAD` of the release commit is logged in every `runs/results.jsonl` row.
- [ ] `NORMALIZER_VERSION` is logged in every row; bump if you change `src/normalization.py`.
