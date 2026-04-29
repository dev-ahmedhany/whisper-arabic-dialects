"""Wilcoxon signed-rank test on per-utterance WERs across two systems.

Used for H1 — does fine-tuned turbo significantly beat zero-shot large-v3 at matched
inference configs? Both systems must have transcribed the *same* utterances in the
same order; that's the contract callers must uphold.
"""

from __future__ import annotations

from typing import Sequence

import jiwer
from scipy.stats import wilcoxon

__all__ = ["per_utterance_wers", "wilcoxon_compare"]


def per_utterance_wers(refs: Sequence[str], hyps: Sequence[str]) -> list[float]:
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) length mismatch")
    return [jiwer.wer([r], [h]) for r, h in zip(refs, hyps)]


def wilcoxon_compare(
    wers_a: Sequence[float],
    wers_b: Sequence[float],
) -> dict[str, float | int]:
    if len(wers_a) != len(wers_b):
        raise ValueError(f"wers_a ({len(wers_a)}) and wers_b ({len(wers_b)}) length mismatch")
    pairs = [(a, b) for a, b in zip(wers_a, wers_b) if not (a == 0.0 and b == 0.0)]
    if len(pairs) < 10:
        return {"stat": float("nan"), "p": float("nan"), "n_compared": len(pairs)}
    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    stat, p = wilcoxon(a_vals, b_vals, zero_method="wilcox", alternative="two-sided")
    return {"stat": float(stat), "p": float(p), "n_compared": len(pairs)}
