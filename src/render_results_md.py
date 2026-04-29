"""Render runs/results.jsonl as a compact, grouped markdown view for browsing.

Produces paper/RESULTS_TABLE.md — one sub-section per
(backend, model, compute, beam, threads, platform) group, with a small
per-dialect table inside. Avoids repeating constant columns on every row.

Usage:
    python -m src.render_results_md runs/results.jsonl paper/RESULTS_TABLE.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GROUP_KEYS = [
    "backend",
    "cfg_model_name",
    "cfg_compute_type",
    "cfg_beam_size",
    "cfg_cpu_threads",
    "platform_label",
]


def _backend(row) -> str:
    ex = row.get("extra")
    if isinstance(ex, dict) and ex.get("backend"):
        return ex["backend"]
    return "ct2-faster-whisper"


def _fmt_pct_ci(point, lo, hi, decimals=1) -> str:
    if pd.isna(point):
        return "-"
    return f"{point*100:.{decimals}f} [{lo*100:.{decimals}f}, {hi*100:.{decimals}f}]"


def _fmt_num(v, fmt) -> str:
    return format(v, fmt) if pd.notna(v) else "-"


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
    df["backend"] = df.apply(_backend, axis=1)

    df = df.sort_values(GROUP_KEYS + ["dialect"]).reset_index(drop=True)

    lines = [
        "# All Benchmark Results (browsing view)",
        "",
        f"Auto-generated from `runs/results.jsonl` ({len(df)} rows). "
        "Re-run `python -m src.render_results_md` to refresh.",
        "",
        "_WER + CER reported as `point [95% bootstrap CI]` (n=1000). "
        "RTF = compute_seconds / audio_seconds (lower is better). "
        "TTFT-p95 is 95th-percentile time-to-first-token in milliseconds._",
        "",
        "## Schema",
        "",
        "Each section header is `backend · model · compute (beam=B, threads=T, platform)`. "
        "The table inside breaks that cell down by dialect.",
        "",
    ]

    # Per-cell rows, grouped
    lines.append("## Per-cell rows")
    lines.append("")

    for keys, sub in df.groupby(GROUP_KEYS, sort=False):
        backend, model, compute, beam, threads, platform = keys
        sub = sub.sort_values("dialect")
        lines.append(
            f"### {backend} · {model} · {compute}"
            f"  _(beam={int(beam)}, threads={int(threads)}, {platform})_"
        )
        lines.append("")
        lines.append("| dialect | n | WER | CER | RTF | TTFT_p95 | Peak RAM |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            wer = _fmt_pct_ci(r.get("wer"), r.get("wer_ci_lo"), r.get("wer_ci_hi"), 1)
            cer = _fmt_pct_ci(r.get("cer"), r.get("cer_ci_lo"), r.get("cer_ci_hi"), 2)
            rtf = _fmt_num(r.get("rtf"), ".3f")
            ttft = (
                f"{r['ttft_ms_p95']:.0f} ms"
                if pd.notna(r.get("ttft_ms_p95"))
                else "-"
            )
            ram = (
                f"{r['peak_memory_mb']/1024:.2f} GB"
                if pd.notna(r.get("peak_memory_mb"))
                else "-"
            )
            lines.append(
                f"| {r['dialect']} | {int(r['n_samples'])} | "
                f"{wer} | {cer} | {rtf} | {ttft} | {ram} |"
            )
        lines.append("")

    # Summary by (backend, model)
    lines += [
        "## Summary by (backend × model) — averaged across dialects",
        "",
        "| backend | model | n_cells | avg WER | avg RTF | avg TTFT_p95 | avg RAM |",
        "|---|---|---|---|---|---|---|",
    ]
    grouped = (
        df.groupby(["backend", "cfg_model_name"])
        .agg(
            n=("dialect", "nunique"),
            avg_wer=("wer", "mean"),
            avg_rtf=("rtf", "mean"),
            avg_ttft=("ttft_ms_p95", "mean"),
            avg_ram=("peak_memory_mb", "mean"),
        )
        .reset_index()
    )
    for _, r in grouped.iterrows():
        wer_pct = f"{r['avg_wer']*100:.1f}%" if pd.notna(r["avg_wer"]) else "-"
        rtf = _fmt_num(r["avg_rtf"], ".3f")
        ttft = f"{r['avg_ttft']:.0f} ms" if pd.notna(r["avg_ttft"]) else "-"
        ram = (
            f"{r['avg_ram']/1024:.2f} GB" if pd.notna(r["avg_ram"]) else "-"
        )
        lines.append(
            f"| {r['backend']} | {r['cfg_model_name']} | "
            f"{int(r['n'])} | {wer_pct} | {rtf} | {ttft} | {ram} |"
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {len(df)} rows ({len(grouped)} model lines) to {args.output_md}")


if __name__ == "__main__":
    main()
