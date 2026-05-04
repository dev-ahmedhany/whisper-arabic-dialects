"""Bench google/gemma-4-E4B-it on Arabic test sets via HF transformers (CPU bf16/fp16).

Gemma-4-E4B-it is a multimodal Gemma-4 with native audio input (~300M
audio encoder + 4.5B effective LLM). Uses AutoModelForMultimodalLM.

Same EvalResult schema as src.eval_harness so rows land in
runs/results.jsonl alongside Whisper rows; tagged backend=gemma-4-E4B-it.

CPU mode default for Hetzner CPX62; pass --device cuda:0 to use GPU.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from dataclasses import asdict
from pathlib import Path

import jiwer
import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.bootstrap_ci import bootstrap_cer_ci, bootstrap_wer_ci
from src.eval_harness import EvalConfig, EvalResult, get_git_commit, get_hardware_id
from src.normalization import NORMALIZER_VERSION, normalize_arabic


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--model-name", default="gemma-4-E4B-4way")
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda:0 — defaults to cpu for Hetzner")
    p.add_argument("--max-samples", type=int, default=25)
    p.add_argument("--platform-label", default="cpu-n25")
    p.add_argument("--output-log", default=Path("runs/results.jsonl"), type=Path)
    p.add_argument("--predictions-dir", default=Path("runs/predictions"), type=Path)
    args = p.parse_args()

    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_dir.mkdir(parents=True, exist_ok=True)

    DATASETS = [
        ("test_sets/top25/v4_casablanca_test_all.jsonl", "casablanca"),
        ("test_sets/top25/v4_masc_clean_test.jsonl", "masc_clean"),
        ("test_sets/top25/v4_masc_noisy_test.jsonl", "masc_noisy"),
        ("test_sets/top25/v4_cv18_test.jsonl", "cv18"),
        ("test_sets/top25/v4_sada_test.jsonl", "sada"),
        ("test_sets/top25/v4_fleurs_test.jsonl", "fleurs"),
    ]

    print(f"loading {args.model} on {args.device} ...", flush=True)
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    proc_obj = psutil.Process()

    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device,
    )
    model.train(False)  # equivalent to .eval(), avoids hook trigger
    print(f"loaded; mem={proc_obj.memory_info().rss/1e9:.1f} GB", flush=True)

    PROMPT = ("Transcribe the following speech segment in its original "
              "language. Only output the transcription, with no newlines.")

    for test_path, dialect in DATASETS:
        tp = Path(test_path)
        if not tp.exists():
            print(f"skip {test_path} - not found", flush=True)
            continue
        samples = [json.loads(l) for l in tp.open() if l.strip()][:args.max_samples]
        print(f"\n=== {dialect}: n={len(samples)} ===", flush=True)

        refs, hyps, ttfts_ms = [], [], []
        total_audio = 0.0
        peak_rss = proc_obj.memory_info().rss
        if "cuda" in args.device:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for i, s in enumerate(samples):
            try:
                t0 = time.perf_counter()
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": s["audio"]},
                        {"type": "text", "text": PROMPT},
                    ],
                }]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True, return_tensors="pt",
                    add_generation_prompt=True,
                ).to(model.device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                text = processor.decode(out[0][input_len:], skip_special_tokens=True)
                # Gemma sometimes returns the chat dict literal as a string —
                # strip the {'role':'assistant','content':'...'} wrapper.
                if hasattr(processor, "parse_response"):
                    try:
                        parsed = processor.parse_response(text)
                        if isinstance(parsed, dict) and "content" in parsed:
                            text = parsed["content"]
                        elif isinstance(parsed, str):
                            text = parsed
                    except Exception:
                        pass
                if isinstance(text, str) and text.lstrip().startswith("{") and "content" in text:
                    import ast
                    try:
                        d = ast.literal_eval(text.strip())
                        if isinstance(d, dict) and "content" in d:
                            text = d["content"]
                    except Exception:
                        pass
                ttfts_ms.append((time.perf_counter() - t0) * 1000.0)
                refs.append(normalize_arabic(s["reference"]))
                hyps.append(normalize_arabic(str(text)))
                total_audio += float(s.get("duration_s", 0.0))
                peak_rss = max(peak_rss, proc_obj.memory_info().rss)
                if i % 5 == 0:
                    print(f"  [{i}/{len(samples)}] hyp={str(text)[:80]!r}", flush=True)
            except Exception as e:
                print(f"  fail @ {i}: {type(e).__name__}: {e}", flush=True)
        elapsed = time.perf_counter() - start

        if not refs:
            continue

        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        wer_m, wer_lo, wer_hi = bootstrap_wer_ci(refs, hyps, n_bootstrap=1000)
        cer_m, cer_lo, cer_hi = bootstrap_cer_ci(refs, hyps, n_bootstrap=1000)
        ttft_arr = np.array(ttfts_ms) if ttfts_ms else np.array([np.nan])
        print(f"=== {dialect}: WER={wer:.4f}  RTF={elapsed/total_audio:.3f}", flush=True)

        if "cuda" in args.device:
            peak_mb = float(torch.cuda.max_memory_allocated()) / (1024**2)
        else:
            peak_mb = peak_rss / (1024**2)

        result = EvalResult(
            config=asdict(EvalConfig(
                model_path=args.model, model_name=args.model_name,
                compute_type=str(dtype).split(".")[-1], beam_size=1,
                cpu_threads=psutil.cpu_count() if args.device == "cpu" else 0,
                device=args.device,
            )),
            test_set=test_path, dialect=dialect,
            n_samples=len(refs), n_failed=len(samples) - len(refs),
            total_audio_seconds=total_audio, total_compute_seconds=elapsed,
            rtf=elapsed/total_audio if total_audio > 0 else float("nan"),
            throughput_realtime_x=total_audio/elapsed if elapsed > 0 else float("nan"),
            peak_memory_mb=peak_mb,
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
            extra={"backend": "gemma-4-E4B-it"},
        )
        with args.output_log.open("a") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        pp = args.predictions_dir / f"preds_{args.model_name}_{dialect}_{args.platform_label}.jsonl"
        with pp.open("w") as f:
            for s, hp, rf in zip(samples, hyps, refs):
                f.write(json.dumps({"audio": s["audio"], "reference": rf, "hypothesis": hp},
                                   ensure_ascii=False) + "\n")
        print(f"wrote {pp}", flush=True)


if __name__ == "__main__":
    main()
