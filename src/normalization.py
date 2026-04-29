"""Arabic text normalization — applied identically to every reference and hypothesis.

Any change to this function invalidates all previously logged WERs, so the version
string is logged with every result row. If you change behavior, bump the version.
"""

from __future__ import annotations

import re

__all__ = ["normalize_arabic", "NORMALIZER_VERSION"]

NORMALIZER_VERSION = "v1"

_DIACRITICS = re.compile(r"[ً-ٰٟ]")
_ALEF_FORMS = re.compile(r"[إأآٱ]")
_TATWEEL = "ـ"
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = _DIACRITICS.sub("", text)
    text = _ALEF_FORMS.sub("ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace(_TATWEEL, "")
    text = _NON_WORD.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()
