"""Request language detection helpers."""
from __future__ import annotations

import re


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")


def detect_response_language(message: str | None) -> str:
    """Return the response language for model-authored user-facing text.

    TSPilot currently only needs a stable zh/en split. Mixed questions with any
    meaningful CJK text should follow Chinese because Chinese users often include
    English metric names, tickers, SQL keywords, or units inside a Chinese task.
    """

    text = str(message or "")
    cjk_count = len(_CJK_RE.findall(text))
    if cjk_count:
        return "zh"
    return "en" if _ASCII_ALPHA_RE.search(text) else "en"
