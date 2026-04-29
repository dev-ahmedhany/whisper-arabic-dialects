"""Tests for src.normalization.

These behaviors are baked into every WER number in the paper. If a test changes,
NORMALIZER_VERSION must bump and all stale results must be re-run.
"""

from __future__ import annotations

from src.normalization import NORMALIZER_VERSION, normalize_arabic


def test_idempotent():
    text = "الْعَالَمُ"
    once = normalize_arabic(text)
    twice = normalize_arabic(once)
    assert once == twice


def test_strips_diacritics():
    assert normalize_arabic("الْعَالَمُ") == "العالم"


def test_unifies_alef():
    assert normalize_arabic("إنسان") == "انسان"
    assert normalize_arabic("أنا") == "انا"
    assert normalize_arabic("آمين") == "امين"


def test_yaa_normalization():
    assert normalize_arabic("على") == "علي"


def test_taa_marbuta():
    assert normalize_arabic("مدرسة") == "مدرسه"


def test_strips_tatweel():
    assert normalize_arabic("كــــتاب") == "كتاب"


def test_punctuation_to_space():
    assert normalize_arabic("مرحبا، كيف الحال؟") == "مرحبا كيف الحال"


def test_collapses_whitespace():
    assert normalize_arabic("  مرحبا   بك   ") == "مرحبا بك"


def test_empty_input():
    assert normalize_arabic("") == ""
    assert normalize_arabic("   ") == ""


def test_full_pipeline():
    raw = "إنّ اللّـهَ غَفُورٌ رَحِيمٌ، يا أخي."
    expected = "ان الله غفور رحيم يا اخي"
    assert normalize_arabic(raw) == expected


def test_version_constant():
    assert isinstance(NORMALIZER_VERSION, str) and NORMALIZER_VERSION
