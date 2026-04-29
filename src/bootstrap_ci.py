"""Bootstrap confidence intervals for WER and CER.

Resamples the (reference, hypothesis) pairs with replacement n times, computes the
metric per sample, returns mean + percentile-based CI bounds. Seeded so identical
inputs always produce identical CIs.
"""

from __future__ import annotations

from typing import Sequence

import jiwer
import numpy as np

__all__ = ["bootstrap_wer_ci", "bootstrap_cer_ci"]


def _bootstrap(
    refs: Sequence[str],
    hyps: Sequence[str],
    metric_fn,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> tuple[float, float, float]:
    if len(refs) != len(hyps):
        raise ValueError(f"refs ({len(refs)}) and hyps ({len(hyps)}) length mismatch")
    if not refs:
        return float("nan"), float("nan"), float("nan")

    n = len(refs)
    rng = np.random.default_rng(seed)
    refs_arr = np.array(refs, dtype=object)
    hyps_arr = np.array(hyps, dtype=object)

    scores = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(list(refs_arr[idx]), list(hyps_arr[idx]))

    alpha = 1.0 - ci
    lo = float(np.percentile(scores, (alpha / 2.0) * 100.0))
    hi = float(np.percentile(scores, (1.0 - alpha / 2.0) * 100.0))
    return float(scores.mean()), lo, hi


def bootstrap_wer_ci(
    refs: Sequence[str],
    hyps: Sequence[str],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    return _bootstrap(refs, hyps, jiwer.wer, n_bootstrap, ci, seed)


def bootstrap_cer_ci(
    refs: Sequence[str],
    hyps: Sequence[str],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    return _bootstrap(refs, hyps, jiwer.cer, n_bootstrap, ci, seed)
