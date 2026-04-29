"""Aggregate runs/results.jsonl → six paper tables, then splice them into paper.md.

Marker convention: in paper/paper.md, each table-shaped result section contains a line:

    <!-- INSERT: table_N -->

This script replaces the line plus everything until the next `<!-- INSERT: ... -->`
or the next H2 (`## `) heading with a freshly rendered table. Idempotent — safe to
re-run after every benchmark sweep.

Usage:
    python -m src.build_results_tables runs/results.jsonl paper/paper.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


MARKER_RE = re.compile(r"<!--\s*INSERT:\s*(?P<id>[a-zA-Z0-9_]+)\s*-->")
DIALECTS = ["msa", "egyptian", "levantine", "gulf", "maghrebi"]


def _load(jsonl_path: Path) -> pd.DataFrame:
    if not jsonl_path.exists():
        return pd.DataFrame()
    df = pd.read_json(jsonl_path, lines=True)
    if df.empty:
        return df
    cfg = pd.json_normalize(df["config"])
    cfg.columns = [f"cfg_{c}" for c in cfg.columns]
    return pd.concat([df.drop(columns=["config"]), cfg], axis=1)


def _fmt_wer(row: pd.Series) -> str:
    if pd.isna(row.get("wer")):
        return "-"
    return f"{row['wer']*100:.1f} [{row['wer_ci_lo']*100:.1f}, {row['wer_ci_hi']*100:.1f}]"


def _pivot_dialect_wers(sub: pd.DataFrame) -> dict[str, str]:
    out = {}
    for d in DIALECTS:
        rows = sub[sub["dialect"] == d]
        if rows.empty:
            out[d] = "-"
        else:
            r = rows.iloc[0]
            out[d] = _fmt_wer(r)
    return out


def table_1_quality_ceiling(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data yet)_"
    sub = df[(df["cfg_compute_type"] == "float32") & (df["cfg_beam_size"] == 5) & (df["cfg_cpu_threads"] == 8)]
    if sub.empty:
        return "_(no rows match fp32 / beam=5 / 8 threads)_"
    rows = []
    for model_name, g in sub.groupby("cfg_model_name"):
        avg_wer = g["wer"].mean() * 100
        per_d = _pivot_dialect_wers(g)
        rows.append(
            f"| {model_name} | fp32 | 5 | 8 | {avg_wer:.1f} | "
            f"{per_d['msa']} | {per_d['egyptian']} | {per_d['levantine']} | "
            f"{per_d['gulf']} | {per_d['maghrebi']} |"
        )
    header = (
        "| Model | Compute | Beam | Threads | Avg WER | MSA | Egyptian | Levantine | Gulf | Maghrebi |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    return header + "\n" + "\n".join(rows)


def table_2_quantization(df: pd.DataFrame, model_name: str = "ft-turbo") -> str:
    if df.empty:
        return "_(no data yet)_"
    sub = df[
        (df["cfg_model_name"] == model_name)
        & (df["cfg_beam_size"] == 1)
        & (df["cfg_cpu_threads"] == 4)
    ]
    if sub.empty:
        return f"_(no {model_name} rows at beam=1, 4 threads)_"
    grouped = sub.groupby("cfg_compute_type").agg(
        wer=("wer", "mean"),
        rtf=("rtf", "mean"),
        mem=("peak_memory_mb", "mean"),
    )
    rows = []
    for ct in ["float32", "float16", "int8_float32", "int8_float16", "int8"]:
        if ct not in grouped.index:
            continue
        r = grouped.loc[ct]
        rows.append(f"| {ct} | {r['wer']*100:.1f} | {r['rtf']:.3f} | {r['mem']/1024:.2f} GB |")
    header = "| Compute type | WER avg | RTF | Memory |\n|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


def table_3_beam(df: pd.DataFrame, model_name: str = "ft-turbo") -> str:
    if df.empty:
        return "_(no data yet)_"
    sub = df[
        (df["cfg_model_name"] == model_name)
        & (df["cfg_compute_type"] == "int8_float32")
        & (df["cfg_cpu_threads"] == 4)
    ]
    if sub.empty:
        return "_(no rows at int8_float32, 4 threads)_"
    grouped = sub.groupby("cfg_beam_size").agg(wer=("wer", "mean"), rtf=("rtf", "mean"))
    rows = [f"| {bs} | {r['wer']*100:.1f} | {r['rtf']:.3f} |" for bs, r in grouped.iterrows()]
    header = "| Beam size | WER avg | RTF |\n|---|---|---|"
    return header + "\n" + "\n".join(rows)


def table_4_threads(df: pd.DataFrame, model_name: str = "ft-turbo") -> str:
    if df.empty:
        return "_(no data yet)_"
    sub = df[
        (df["cfg_model_name"] == model_name)
        & (df["cfg_compute_type"] == "int8_float32")
        & (df["cfg_beam_size"] == 1)
        & (df["dialect"] == "msa")
    ]
    if sub.empty:
        return "_(no rows for thread-scaling sweep)_"
    grouped = sub.groupby("cfg_cpu_threads").agg(rtf=("rtf", "mean"))
    if 1 not in grouped.index:
        baseline_rtf = grouped["rtf"].iloc[0]
    else:
        baseline_rtf = grouped.loc[1, "rtf"]
    rows = [
        f"| {t} | {r['rtf']:.3f} | {baseline_rtf / r['rtf']:.2f}× |"
        for t, r in grouped.iterrows()
    ]
    header = "| Threads | RTF | Speedup vs 1 thread |\n|---|---|---|"
    return header + "\n" + "\n".join(rows)


def table_5_cross_platform(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data yet)_"
    pivot = df.pivot_table(
        index=["cfg_model_name", "cfg_compute_type", "cfg_beam_size"],
        columns="platform_label",
        values="rtf",
        aggfunc="mean",
    )
    cols = list(pivot.columns)
    if not any("hetzner" in c.lower() for c in cols) or not any("gcp" in c.lower() for c in cols):
        return "_(both platforms required for cross-platform table)_"
    gcp_col = next(c for c in cols if "gcp" in c.lower())
    hz_col = next(c for c in cols if "hetzner" in c.lower())

    rows = []
    for idx, row in pivot.iterrows():
        gcp_rtf = row.get(gcp_col)
        hz_rtf = row.get(hz_col)
        if pd.isna(gcp_rtf) or pd.isna(hz_rtf):
            continue
        ratio = hz_rtf / gcp_rtf
        model, ct, bs = idx
        rows.append(
            f"| {model} | {ct} | {bs} | {gcp_rtf:.3f} | {hz_rtf:.3f} | {ratio:.2f}× | "
            f"$0.40 | $0.043 | 9.3× |"
        )
    header = (
        "| Model | Compute | Beam | GCP RTF | CX53 RTF | RTF ratio | GCP $/hr | CX53 $/hr | Cost ratio |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    return header + "\n" + "\n".join(rows) if rows else "_(no overlapping rows)_"


# Per-platform hourly rates (USD). Used to derive cost_per_audio_hour = $/hr × RTF
# in Table 6 production recommendations. Add new platforms here as we benchmark them.
PLATFORM_HOURLY_USD = {
    "gcp-c3-standard-8": 0.40,
    "gcp-n2-standard-8": 0.30,
    "hetzner-cx53": 0.043,
    "local": 0.0,
    "dryrun": 0.0,
}


def _add_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hourly_usd"] = df["platform_label"].map(PLATFORM_HOURLY_USD).fillna(0.0)
    # Cost per hour of *audio* transcribed = (machine $/hr) × RTF.
    # Example: at $0.40/hr and RTF=0.5, transcribing 1h of audio costs $0.20.
    df["cost_per_audio_hour"] = df["hourly_usd"] * df["rtf"]
    return df


def _aggregate_across_dialects(df: pd.DataFrame, min_dialects: int = 4) -> pd.DataFrame:
    """Average per-cell metrics across dialects per (model × compute × beam × threads × platform).

    A production deployment serves ALL dialects, not just MSA. Selecting
    Table 6 rows on single-dialect WER (typically MSA, the easiest condition)
    over-promises. Instead we aggregate: group by config, average WER /
    RTF / TTFT / memory across the dialects the group has, then drop groups
    that don't have at least `min_dialects` dialects represented (so we
    don't compare a partial 2-dialect config to a full 5-dialect one).
    """
    keys = ["cfg_model_name", "cfg_compute_type", "cfg_beam_size",
            "cfg_cpu_threads", "platform_label"]
    agg = df.groupby(keys).agg(
        wer=("wer", "mean"),
        wer_ci_lo=("wer_ci_lo", "mean"),
        wer_ci_hi=("wer_ci_hi", "mean"),
        rtf=("rtf", "mean"),
        peak_memory_mb=("peak_memory_mb", "mean"),
        ttft_ms_p95=("ttft_ms_p95", "mean") if "ttft_ms_p95" in df.columns
                    else ("rtf", "mean"),  # fallback if no TTFT
        hourly_usd=("hourly_usd", "first"),
        n_dialects=("dialect", "nunique"),
    ).reset_index()
    agg["cost_per_audio_hour"] = agg["hourly_usd"] * agg["rtf"]
    agg = agg[agg["n_dialects"] >= min_dialects]
    return agg


def table_6_recommendations(df: pd.DataFrame) -> str:
    """Five rows, each picking one config from the matrix per a deployment constraint.

    All metrics are AVERAGED across the dialects in the group — a production
    deployment must serve every dialect, so a recommendation that wins on
    MSA alone but fails on Maghrebi is misleading.
    """
    if df.empty:
        return "_(no data yet)_"
    df = _add_cost_columns(df)
    agg = _aggregate_across_dialects(df, min_dialects=4)
    if agg.empty:
        return "_(no config has been benchmarked across enough dialects yet)_"
    median_wer = float(agg["wer"].median())
    rows = []

    def _row(label: str, sub: pd.DataFrame, constraint: str) -> str:
        if sub.empty:
            return f"| {label} | {constraint} | - | - | - | - | - | - | - | - | - |"
        r = sub.iloc[0]
        cost_str = f"${r['cost_per_audio_hour']:.3f}/audio-hr" if r["hourly_usd"] > 0 else "-"
        wer_str = f"{r['wer']*100:.1f} [{r['wer_ci_lo']*100:.1f}, {r['wer_ci_hi']*100:.1f}]"
        return (
            f"| {label} | {constraint} | {r['cfg_model_name']} | {r['cfg_compute_type']} | "
            f"{int(r['cfg_beam_size'])} | {int(r['cfg_cpu_threads'])} | {r['platform_label']} | "
            f"{int(r['n_dialects'])} | {wer_str} | {r['rtf']:.3f} | {cost_str} |"
        )

    # 1. Real-time captioning: tail latency matters across the whole dialect set.
    realtime = agg[(agg["ttft_ms_p95"] < 1000) & (agg["wer"] < median_wer)] \
        .sort_values("ttft_ms_p95").head(1)
    rows.append(_row("Real-time captioning", realtime, "avg TTFT-p95 < 1s, avg WER < median"))

    # 2. Batch transcription, accuracy is everything.
    batch_min = agg.sort_values("wer").head(1)
    rows.append(_row("Batch transcription (min avg WER)", batch_min, "min avg WER"))

    # 3. Edge: smallest memory footprint at a quality floor.
    edge = agg[(agg["peak_memory_mb"] < 1024) & (agg["wer"] < median_wer)] \
        .sort_values("peak_memory_mb").head(1)
    rows.append(_row("Edge deployment", edge, "RAM < 1 GB, avg WER < median"))

    # 4. Balanced production: best avg WER under a usable latency budget.
    balanced = agg[agg["rtf"] < 0.5].sort_values("wer").head(1)
    rows.append(_row("Balanced production", balanced, "avg RTF < 0.5, max accuracy"))

    # 5. Cost-optimized: minimize $/audio-hour subject to a quality floor.
    cost_opt = agg[agg["wer"] < median_wer].sort_values("cost_per_audio_hour").head(1)
    rows.append(_row("Cost-optimized", cost_opt, "min $/audio-hr, avg WER < median"))

    header = (
        "| Use case | Constraint | Best model | Compute | Beam | Threads | Platform | "
        "Dialects covered | Avg WER | Avg RTF | $/audio-hr |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    return header + "\n" + "\n".join(rows)


TABLE_BUILDERS = {
    "table_1": table_1_quality_ceiling,
    "table_2": table_2_quantization,
    "table_3": table_3_beam,
    "table_4": table_4_threads,
    "table_5": table_5_cross_platform,
    "table_6": table_6_recommendations,
}


def splice_tables(paper_md: str, df: pd.DataFrame) -> str:
    lines = paper_md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = MARKER_RE.match(lines[i].strip())
        if not m:
            out.append(lines[i])
            i += 1
            continue
        marker_id = m.group("id")
        builder = TABLE_BUILDERS.get(marker_id)
        out.append(lines[i])
        if not builder:
            print(f"[warn] unknown marker {marker_id!r}, leaving as-is", file=sys.stderr)
            i += 1
            continue
        rendered = builder(df)
        out.append("")
        out.extend(rendered.splitlines())
        out.append("")
        i += 1
        while i < len(lines):
            stripped = lines[i].strip()
            if MARKER_RE.match(stripped) or stripped.startswith("## "):
                break
            i += 1
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_jsonl", type=Path, nargs="?", default=Path("runs/results.jsonl"))
    p.add_argument("paper_md", type=Path, nargs="?", default=Path("paper/paper.md"))
    args = p.parse_args()

    df = _load(args.results_jsonl)
    if df.empty:
        print(f"[info] {args.results_jsonl} has no rows yet; tables will render as placeholders.")
    else:
        print(f"[info] loaded {len(df)} result rows from {args.results_jsonl}")

    paper = args.paper_md.read_text()
    updated = splice_tables(paper, df)
    args.paper_md.write_text(updated)
    print(f"[ok] tables spliced into {args.paper_md}")


if __name__ == "__main__":
    main()
