"""Render runs/results.jsonl as a comprehensive markdown table for browsing.

Produces paper/RESULTS_TABLE.md — one row per benchmark cell, grouped by
(backend, model, dialect) for fast visual scanning. Easier to inspect than
the raw JSONL when you want to see what the project has measured so far.

Usage:
    python -m src.render_results_md runs/results.jsonl paper/RESULTS_TABLE.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_jsonl", type=Path, nargs="?", default=Path("runs/results.jsonl"))
    p.add_argument("output_md", type=Path, nargs="?", default=Path("paper/RESULTS_TABLE.md"))
    args = p.parse_args()

    if not args.results_jsonl.exists():
        raise SystemExit(f"{args.results_jsonl} not found")
    df = pd.read_json(args.results_jsonl, lines=True)
    if df.empty:
        raise SystemExit("results.jsonl is empty")

    cfg = pd.json_normalize(df["config"])
    cfg.columns = [f"cfg_{c}" for c in cfg.columns]
    df = pd.concat([df.drop(columns=["config"]), cfg], axis=1)

    # backend column from extra.backend, default 'ct2-faster-whisper'
    def _backend(row):
        ex = row.get("extra")
        if isinstance(ex, dict) and ex.get("backend"):
            return ex["backend"]
        return "ct2-faster-whisper"
    df["backend"] = df.apply(_backend, axis=1)

    df = df.sort_values(["backend", "cfg_model_name", "cfg_compute_type", "dialect"]).reset_index(drop=True)

    lines = [
        "# All Benchmark Results (browsing view)",
        "",
        f"Auto-generated from `runs/results.jsonl` ({len(df)} rows). Re-run `python -m src.render_results_md` to refresh.",
        "",
        f"_Last refreshed at run-time. WER + CER reported as `point [95% bootstrap CI]` (n=1000 samples). RTF is `compute_seconds / audio_seconds` (lower is better). TTFT-p95 is the 95th-percentile time-to-first-token in milliseconds._",
        "",
        "## Schema",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| backend | Inference engine: ct2-faster-whisper / hf-transformers / whisper.cpp / openai-whisper |",
        "| model | Model name as logged |",
        "| compute | Quantization / dtype (int8, int8_float32, float32, q5_0, etc.) |",
        "| beam | Beam size during decode |",
        "| threads | CPU threads given to the engine |",
        "| dialect | Arabic dialect of the test set |",
        "| n | Number of samples in this cell |",
        "| WER | Word Error Rate, point + 95% bootstrap CI |",
        "| CER | Character Error Rate |",
        "| RTF | compute_time / audio_time |",
        "| TTFT_p95 | 95th-percentile time-to-first-token (ms) |",
        "| Peak RAM | MB at peak during inference |",
        "| Platform | hardware label (gcp-c3-standard-8 / hetzner-cx53 / ...) |",
        "",
    ]

    header = ("| backend | model | compute | beam | threads | dialect | n | "
              "WER | CER | RTF | TTFT_p95 | Peak RAM | Platform |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append("## Per-cell rows")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for _, r in df.iterrows():
        wer = f"{r['wer']*100:.1f} [{r['wer_ci_lo']*100:.1f}, {r['wer_ci_hi']*100:.1f}]" if pd.notna(r.get("wer")) else "-"
        cer = f"{r['cer']*100:.2f} [{r['cer_ci_lo']*100:.2f}, {r['cer_ci_hi']*100:.2f}]" if pd.notna(r.get("cer")) else "-"
        rtf = f"{r['rtf']:.3f}" if pd.notna(r.get("rtf")) else "-"
        ttft = f"{r['ttft_ms_p95']:.0f} ms" if pd.notna(r.get("ttft_ms_p95")) else "-"
        ram = f"{r['peak_memory_mb']/1024:.2f} GB" if pd.notna(r.get("peak_memory_mb")) else "-"
        lines.append(
            f"| {r['backend']} | {r['cfg_model_name']} | {r['cfg_compute_type']} | "
            f"{int(r['cfg_beam_size'])} | {int(r['cfg_cpu_threads'])} | {r['dialect']} | "
            f"{int(r['n_samples'])} | {wer} | {cer} | {rtf} | {ttft} | {ram} | "
            f"{r['platform_label']} |"
        )

    # Summary by (backend, model)
    lines += ["", "## Summary by (backend × model) — average across dialects", "",
              "| backend | model | n_cells | avg WER | avg RTF | avg TTFT_p95 | avg RAM |",
              "|---|---|---|---|---|---|---|"]
    grouped = df.groupby(["backend", "cfg_model_name"]).agg(
        n=("dialect", "nunique"),
        avg_wer=("wer", "mean"),
        avg_rtf=("rtf", "mean"),
        avg_ttft=("ttft_ms_p95", "mean"),
        avg_ram=("peak_memory_mb", "mean"),
    ).reset_index()
    for _, r in grouped.iterrows():
        wer_pct = f"{r['avg_wer']*100:.1f}%" if pd.notna(r["avg_wer"]) else "-"
        rtf = f"{r['avg_rtf']:.3f}" if pd.notna(r["avg_rtf"]) else "-"
        ttft = f"{r['avg_ttft']:.0f} ms" if pd.notna(r["avg_ttft"]) else "-"
        ram = f"{r['avg_ram']/1024:.2f} GB" if pd.notna(r["avg_ram"]) else "-"
        lines.append(f"| {r['backend']} | {r['cfg_model_name']} | {int(r['n'])} | {wer_pct} | {rtf} | {ttft} | {ram} |")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {len(df)} rows to {args.output_md}")


if __name__ == "__main__":
    main()
