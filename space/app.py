"""Gradio space for Whisper-large-v3-turbo Arabic FT.

Lets visitors transcribe an Arabic audio clip with both:
- the unmodified zero-shot Whisper-large-v3-turbo baseline
- our 4-dialect QLoRA fine-tune

The side-by-side output makes the FT lift on dialects (and the trade-offs
on MSA / Levantine) directly visible.
"""

import gradio as gr
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16

ZS_MODEL_ID = "openai/whisper-large-v3-turbo"
FT_MODEL_ID = "dev-ahmedhany/whisper-large-v3-turbo-arabic-ft"

print(f"loading processors and models on {DEVICE}/{DTYPE}...")
processor = WhisperProcessor.from_pretrained(ZS_MODEL_ID, language="arabic", task="transcribe")

zs_model = WhisperForConditionalGeneration.from_pretrained(ZS_MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
ft_model = WhisperForConditionalGeneration.from_pretrained(FT_MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

zs_model.eval()
ft_model.eval()


def _transcribe_one(model, audio_array, sr, beam_size=1):
    """Run one model on one audio array. Returns transcription text."""
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)
    if audio_array.max() > 1.0:
        # int16 PCM — normalize to [-1, 1]
        audio_array = audio_array / 32768.0
    if sr != 16000:
        # gradio gives us native sample rate; resample to 16k for Whisper
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        ids = model.generate(
            input_features=input_features,
            language="arabic", task="transcribe",
            max_new_tokens=225, num_beams=beam_size,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def transcribe_both(audio):
    if audio is None:
        return "—", "—"
    sr, arr = audio
    zs_text = _transcribe_one(zs_model, arr, sr)
    ft_text = _transcribe_one(ft_model, arr, sr)
    return zs_text, ft_text


HEADER = """
# Whisper-large-v3-turbo — Arabic 4-dialect Fine-Tune

Side-by-side transcription with the **unmodified Whisper-large-v3-turbo** vs. our **QLoRA fine-tune** on
4 Arabic dialects (MSA + Egyptian + Levantine + Gulf).

The fine-tune is part of the [whisper-arabic-dialects](https://github.com/dev-ahmedhany/whisper-arabic-dialects)
research project. See the [model card](https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft)
for held-out per-dialect WER and known limitations.

> **Tip:** record a short Arabic clip (5–10 seconds) and compare both outputs. Egyptian and Gulf
> samples should show the clearest fine-tuning lift; MSA is roughly tied with the baseline; broadcast-style
> Levantine may be worse on the FT model (paper §6.1 — train-test domain mismatch).
"""

with gr.Blocks(title="Arabic Whisper FT", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)
    audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy",
                        label="Audio (mic or upload, any sample rate)")
    btn = gr.Button("Transcribe with both models", variant="primary")
    with gr.Row():
        zs_out = gr.Textbox(label="Zero-shot Whisper-large-v3-turbo (baseline)",
                             rtl=True, interactive=False, lines=4)
        ft_out = gr.Textbox(label="Fine-tuned (this work)",
                             rtl=True, interactive=False, lines=4)
    btn.click(transcribe_both, inputs=audio_in, outputs=[zs_out, ft_out])
    gr.Markdown(
        "Author: [Ahmed Hany](https://hany.dev) · "
        "[Repo](https://github.com/dev-ahmedhany/whisper-arabic-dialects) · "
        "[Model card](https://huggingface.co/dev-ahmedhany/whisper-large-v3-turbo-arabic-ft)"
    )

if __name__ == "__main__":
    demo.queue(max_size=8).launch()
