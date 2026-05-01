"""Benchmark Voxtral (Mistral multilingual audio LLM) on Arabic test sets.

Voxtral comes in two sizes - Mini-3B (production-friendly) and Small-24B (for
quality ceiling testing). The HF model card recommends transformers >= 4.45
with the Voxtral processor; vLLM is faster but not strictly needed for an
N=100 run.

Default loads Mini-3B. Pass --model mistralai/Voxtral-Small-24B-2507 --load-4bit
to test the 24B variant on a 24 GB GPU.

Output: writes WER/CER + bootstrap CIs into runs/results.jsonl using the
same EvalResult schema as src.eval_harness, tagged with backend=voxtral so
downstream scoring scripts can filter.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jiwer
import torch

from src.bootstrap_ci import bootstrap_cer_ci, bootstrap_wer_ci
from src.eval_harness import EvalConfig, EvalResult, get_git_commit, get_hardware_id
from src.normalization import NORMALIZER_VERSION, normalize_arabic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Voxtral-Mini-3B-2507")
    p.add_argument("--model-name", default=None)
    p.add_argument("--test-set", required=True, type=Path)
    p.add_argument("--dialect", required=True)
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-log", default=Path("runs/results.jsonl"), type=Path)
    p.add_argument("--predictions-dir", default=Path("runs/predictions"), type=Path)
    p.add_argument("--platform-label", default="l4-gpu")
    p.add_argument("--prompt", default="Transcribe the Arabic audio verbatim. Reply with only the transcription, no extra text.")
    args = p.parse_args()
    if args.model_name is None:
        args.model_name = args.model.rstrip("/").split("/")[-1].lower().replace("-", "")
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": args.device}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    print(f"loading {args.model} (4bit={args.load_4bit}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = VoxtralForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"loaded; gpu_mem={torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    samples = [json.loads(l) for l in args.test_set.open() if l.strip()][:args.max_samples]
    print(f"n={len(samples)} dialect={args.dialect}", flush=True)

    refs, hyps, ttfts_ms = [], [], []
    total_audio = 0.0
    start = time.perf_counter()
    for i, s in enumerate(samples):
        try:
            t0 = time.perf_counter()
            conversation = [
                {"role": "user", "content": [
                    {"type": "audio", "path": s["audio"]},
                    {"type": "text", "text": args.prompt},
                ]}
            ]
            inputs = processor.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(args.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            ttfts_ms.append((time.perf_counter() - t0) * 1000.0)
            refs.append(normalize_arabic(s["reference"]))
            hyps.append(normalize_arabic(text))
            total_audio += float(s.get("duration_s", 0.0))
            if i % 10 == 0:
                print(f"  [{i}/{len(samples)}] hyp={text[:80]!r}", flush=True)
        except Exception as e:
            print(f"  fail @ {i}: {type(e).__name__}: {e}", flush=True)
    elapsed = time.perf_counter() - start

    if not refs:
        raise RuntimeError("all samples failed")

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    wer_m, wer_lo, wer_hi = bootstrap_wer_ci(refs, hyps, n_bootstrap=1000)
    cer_m, cer_lo, cer_hi = bootstrap_cer_ci(refs, hyps, n_bootstrap=1000)
    import numpy as np
    ttft_arr = np.array(ttfts_ms) if ttfts_ms else np.array([np.nan])
    print(f"=== {args.dialect}: WER={wer:.4f}  CI=[{wer_lo:.4f}, {wer_hi:.4f}]  RTF={elapsed/total_audio:.3f}", flush=True)

    result = EvalResult(
        config=asdict(EvalConfig(
            model_path=args.model, model_name=args.model_name,
            compute_type="4bit" if args.load_4bit else "bfloat16",
            beam_size=1, cpu_threads=0, device=args.device,
        )),
        test_set=str(args.test_set), dialect=args.dialect,
        n_samples=len(refs), n_failed=len(samples) - len(refs),
        total_audio_seconds=total_audio, total_compute_seconds=elapsed,
        rtf=elapsed/total_audio if total_audio>0 else float("nan"),
        throughput_realtime_x=total_audio/elapsed if elapsed>0 else float("nan"),
        peak_memory_mb=float(torch.cuda.max_memory_allocated()) / (1024**2),
        wer=wer, wer_ci_lo=wer_lo, wer_ci_hi=wer_hi,
        cer=cer, cer_ci_lo=cer_lo, cer_ci_hi=cer_hi,
        ttft_ms_mean=float(np.nanmean(ttft_arr)),
        ttft_ms_p50=float(np.nanpercentile(ttft_arr, 50)),
        ttft_ms_p95=float(np.nanpercentile(ttft_arr, 95)),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hardware_id=get_hardware_id(),
        platform_label=args.platform_label,
        normalizer_version=NORMALIZER_VERSION,
        git_commit=get_git_commit(),
        extra={"backend": "voxtral"},
    )
    with args.output_log.open("a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    pred_path = args.predictions_dir / f"preds_{args.model_name}_{args.dialect}_{args.platform_label}.jsonl"
    with pred_path.open("w") as f:
        for s, hyp, ref in zip(samples, hyps, refs):
            f.write(json.dumps({"audio": s["audio"], "reference": ref, "hypothesis": hyp},
                               ensure_ascii=False) + "\n")
    print(f"wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()
