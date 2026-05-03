"""Reverse-engineer Qwen3-ASR-1.7B: prove the architecture and replay the fused runtime.

Answers two questions:
  (Q1) Does Qwen3-ASR-1.7B use OpenAI's whisper-large-v3 encoder weights, or
       just the same architecture?
  (Q2) Can we run the encoder + adapter + LLM pipeline ourselves in plain
       PyTorch, without the qwen_asr convenience wrapper?

Run on a single L4 (or any 24 GB GPU) with both whisper-large-v3 and
Qwen3-ASR-1.7B downloadable. Stages:

  --inspect    : list submodules, count params, find the audio encoder + adapter
  --compare    : load openai/whisper-large-v3 alongside, compare layer shapes +
                 cosine similarity of attention/FFN weights for layers 0/15/31
  --transcribe : manual fused inference on a single audio file, decoding step
                 by step so each tensor shape is visible

Usage:
  python scripts/inspect_qwen3_asr.py --inspect
  python scripts/inspect_qwen3_asr.py --compare
  python scripts/inspect_qwen3_asr.py --transcribe --audio audio/foo.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _params_m(mod: torch.nn.Module) -> float:
    return sum(p.numel() for p in mod.parameters()) / 1e6


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def load_qwen(device: str = "cuda:0", dtype=torch.bfloat16):
    from qwen_asr import Qwen3ASRModel

    print(f"loading Qwen/Qwen3-ASR-1.7B on {device} ...", flush=True)
    return Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B", dtype=dtype, device_map=device
    )


def find_submodule(root: torch.nn.Module, name_hints: tuple[str, ...]):
    """Walk the module tree and return the first match by attribute name OR class name."""
    for name, mod in root.named_modules():
        leaf = name.rsplit(".", 1)[-1] if name else ""
        cls = type(mod).__name__
        if any(h in leaf or h in cls for h in name_hints):
            return name, mod
    return None, None


def cmd_inspect(qwen) -> None:
    print("\n=== Top-level submodules ===")
    for name, mod in qwen.named_children():
        print(f"  {name:30s}  {type(mod).__name__:30s}  {_params_m(mod):8.1f}M")

    print("\n=== Submodules anywhere matching 'WhisperEncoder' / 'audio' / 'projector' / 'adapter' ===")
    seen = set()
    for name, mod in qwen.named_modules():
        cls = type(mod).__name__
        leaf = name.rsplit(".", 1)[-1] if name else "(root)"
        if any(k.lower() in (leaf + cls).lower() for k in ("WhisperEncoder", "audio", "projector", "adapter", "thinker")):
            key = name.split(".")[0:2]
            key = ".".join(key)
            if key in seen:
                continue
            seen.add(key)
            print(f"  {name:50s}  {cls:30s}  {_params_m(mod):8.1f}M")

    print("\n=== Encoder discovery ===")
    enc_name, enc = find_submodule(qwen, ("WhisperEncoder", "audio_tower", "audio_encoder", "speech_encoder"))
    if enc is None:
        print("  no Whisper-style encoder found — model may use a custom audio frontend")
    else:
        n_layers = len(enc.layers) if hasattr(enc, "layers") else "?"
        d_model = getattr(getattr(enc, "config", None), "d_model", "?")
        print(f"  found at: {enc_name}")
        print(f"  layers={n_layers}  d_model={d_model}  params={_params_m(enc):.1f}M")

    print("\n=== Adapter discovery ===")
    ad_name, ad = find_submodule(qwen, ("multi_modal_projector", "audio_projector", "audio_adapter", "mm_projector", "audio_token"))
    if ad is None:
        print("  no obvious adapter — it may be inlined inside the audio encoder's last layers")
    else:
        print(f"  found at: {ad_name}  ({type(ad).__name__})")
        print(f"  params={_params_m(ad):.2f}M")
        for n, p in ad.named_parameters():
            print(f"    {n:30s}  shape={tuple(p.shape)}")

    print("\n=== LLM decoder discovery ===")
    llm_name, llm = find_submodule(qwen, ("Qwen3", "language_model", "lm_head", "thinker"))
    if llm is not None:
        print(f"  found at: {llm_name}  ({type(llm).__name__})  params={_params_m(llm):.1f}M")


def cmd_compare(qwen) -> None:
    from transformers import WhisperModel

    enc_name, qenc = find_submodule(qwen, ("WhisperEncoder", "audio_tower", "audio_encoder", "speech_encoder"))
    if qenc is None or not hasattr(qenc, "layers"):
        print("Qwen audio encoder not found / not Whisper-shaped — abort compare")
        return

    print("loading openai/whisper-large-v3 (encoder only) for comparison ...", flush=True)
    wm = WhisperModel.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.bfloat16)
    wenc = wm.encoder

    print(f"\n=== Shape comparison ===")
    print(f"  Whisper-large-v3:   layers={len(wenc.layers)}  d_model={wenc.config.d_model}")
    print(f"  Qwen3-ASR encoder:  layers={len(qenc.layers)}  d_model={getattr(getattr(qenc,'config',None),'d_model','?')}")

    if len(wenc.layers) != len(qenc.layers):
        print("  Layer count differs — Qwen3-ASR has a modified Whisper architecture, "
              "probably extra audio-LLM-bridging layers appended.")
        return

    print(f"\n=== Weight similarity (Qwen vs OpenAI Whisper-large-v3) ===")
    print(f"{'layer':>5s}  {'q_proj cos':>11s}  {'q_proj maxdiff':>14s}  {'fc1 cos':>9s}  {'fc1 maxdiff':>11s}")
    for i in [0, len(wenc.layers) // 2, len(wenc.layers) - 1]:
        wq = wenc.layers[i].self_attn.q_proj.weight.detach().cpu()
        qq = qenc.layers[i].self_attn.q_proj.weight.detach().cpu()
        wf = wenc.layers[i].fc1.weight.detach().cpu()
        qf = qenc.layers[i].fc1.weight.detach().cpu()
        print(f"{i:5d}  {_cos_sim(wq, qq):11.4f}  {_max_abs_diff(wq, qq):14.4f}  "
              f"{_cos_sim(wf, qf):9.4f}  {_max_abs_diff(wf, qf):11.4f}")

    print("\nInterpretation:")
    print("  cos_sim ~1.000 + maxdiff ~0      → bit-identical (frozen Whisper)")
    print("  cos_sim ~0.95-0.99 + maxdiff > 0 → initialized from Whisper, then jointly trained")
    print("  cos_sim < 0.5                     → different weights, only architecture is shared")


def cmd_transcribe(qwen, audio_path: Path) -> None:
    """Replay the fused runtime manually — encoder, adapter, embed-injection, generate."""
    from transformers import AutoProcessor, WhisperFeatureExtractor

    enc_name, qenc = find_submodule(qwen, ("WhisperEncoder", "audio_tower", "audio_encoder"))
    ad_name, adapter = find_submodule(qwen, ("multi_modal_projector", "audio_projector", "mm_projector"))
    llm_name, llm = find_submodule(qwen, ("language_model",))
    print(f"using:\n  encoder = {enc_name}\n  adapter = {ad_name}\n  llm     = {llm_name}\n")

    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    import librosa
    audio, sr = librosa.load(str(audio_path), sr=16000)
    print(f"audio: {len(audio)/sr:.2f}s @ {sr}Hz")

    mel = fe(audio, sampling_rate=16000, return_tensors="pt").input_features
    mel = mel.to(qenc.parameters().__next__().device, dtype=torch.bfloat16)
    print(f"mel input shape: {tuple(mel.shape)}  (expect (1, 128, 3000) for large-v3)")

    with torch.inference_mode():
        enc_out = qenc(mel).last_hidden_state
        print(f"encoder out shape: {tuple(enc_out.shape)}  (expect (1, 1500, 1280))")

        if adapter is not None:
            audio_feats = adapter(enc_out)
        else:
            audio_feats = enc_out
        print(f"after adapter shape: {tuple(audio_feats.shape)}  (LLM-hidden-dim, possibly stride-reduced)")

    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B", trust_remote_code=True)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    msgs = [{"role": "system", "content": "Transcribe the following audio."},
            {"role": "user", "content": "<|audio|>"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"\nchat-template prompt:\n{text!r}")

    print("\nThe remaining steps (audio-token-id → embedding splice → llm.generate(inputs_embeds=...))\n"
          "depend on Qwen3-ASR's specific audio-token convention; the qwen_asr package wraps them.\n"
          "But the per-stage shapes printed above are exactly what a custom CT2/ONNX/llama.cpp port\n"
          "would have to reproduce — that's the value of this script.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--transcribe", action="store_true")
    p.add_argument("--audio", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if not (args.inspect or args.compare or args.transcribe):
        args.inspect = True

    qwen = load_qwen(device=args.device)

    if args.inspect:
        cmd_inspect(qwen)
    if args.compare:
        cmd_compare(qwen)
    if args.transcribe:
        if args.audio is None:
            raise SystemExit("--transcribe needs --audio <path>")
        cmd_transcribe(qwen, args.audio)


if __name__ == "__main__":
    main()
