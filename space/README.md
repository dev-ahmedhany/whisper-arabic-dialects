---
title: Whisper Arabic Dialects FT
emoji: 🗣️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Zero-shot vs fine-tuned Whisper on Arabic dialects
---

# Whisper-large-v3-turbo — Arabic 4-dialect Fine-Tune (interactive demo)

Compare the **unmodified Whisper-large-v3-turbo** with our **dialect-balanced QLoRA fine-tune**
on the same audio clip. Trained on MSA + Egyptian + Levantine + Gulf (Maghrebi excluded);
see the [model card](https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft)
for held-out per-dialect WER and the [paper draft](https://github.com/dev-ahmedhany/whisper-arabic-dialects/blob/main/paper/paper.md)
for the full methodology.

This Space runs on **free CPU**, so transcription takes 30–60 seconds for a 10-second clip.
For production deployment use the model directly via `transformers` or `faster-whisper` —
the model card has copy-pasteable usage code.

— [Ahmed Hany](https://hany.dev)
