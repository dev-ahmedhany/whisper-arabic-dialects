"""Tests for src.bootstrap_ci."""

from __future__ import annotations

import pytest

from src.bootstrap_ci import bootstrap_cer_ci, bootstrap_wer_ci


def test_perfect_match_zero_wer():
    refs = ["مرحبا بك", "كيف حالك", "شكرا لك"]
    hyps = list(refs)
    mean, lo, hi = bootstrap_wer_ci(refs, hyps, n_bootstrap=200, seed=1)
    assert mean == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_total_mismatch_high_wer():
    refs = ["one two three four"]
    hyps = ["five six seven eight"]
    mean, lo, hi = bootstrap_wer_ci(refs, hyps, n_bootstrap=100, seed=1)
    assert mean == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_ci_brackets_mean():
    refs = [
        "the quick brown fox", "the cat sat on the mat", "hello world",
        "good morning", "she sells seashells", "all your base",
        "the rain in spain", "lorem ipsum dolor sit", "to be or not to be",
        "a journey of a thousand miles",
    ]
    hyps = [r.replace("the", "a") for r in refs]
    mean, lo, hi = bootstrap_wer_ci(refs, hyps, n_bootstrap=500, seed=42)
    assert lo <= mean <= hi
    assert 0.0 < mean < 1.0


def test_deterministic_under_same_seed():
    refs = ["one two three"] * 5
    hyps = ["one four three"] * 5
    a = bootstrap_wer_ci(refs, hyps, n_bootstrap=200, seed=7)
    b = bootstrap_wer_ci(refs, hyps, n_bootstrap=200, seed=7)
    assert a == b


def test_different_seed_different_ci_bounds():
    refs = [f"sample {i} word" for i in range(20)]
    hyps = [f"sample {i} different" for i in range(20)]
    a = bootstrap_wer_ci(refs, hyps, n_bootstrap=200, seed=1)
    b = bootstrap_wer_ci(refs, hyps, n_bootstrap=200, seed=2)
    assert a != b


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_wer_ci(["a"], ["a", "b"])


def test_cer_basic():
    refs = ["abcde"]
    hyps = ["abxde"]
    mean, lo, hi = bootstrap_cer_ci(refs, hyps, n_bootstrap=50, seed=1)
    assert mean == pytest.approx(0.2, abs=1e-9)
    assert lo == pytest.approx(0.2, abs=1e-9)
    assert hi == pytest.approx(0.2, abs=1e-9)


def test_empty_inputs():
    mean, lo, hi = bootstrap_wer_ci([], [], n_bootstrap=100)
    assert mean != mean  # nan
    assert lo != lo
    assert hi != hi
