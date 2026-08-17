"""Locale-aware instructions shared by every request-scoped LLM prompt."""
from __future__ import annotations

from typing import Literal


PromptLocale = Literal["zh", "en"]


def normalize_prompt_locale(value: str | None) -> PromptLocale:
    """Normalize the request language to one of the supported prompt locales."""

    return "zh" if str(value or "").strip().lower().startswith("zh") else "en"


def prompt_locale_instruction(value: str | None) -> str:
    """Return a native-language control block for request-scoped prompts.

    JSON contracts, code, identifiers and datasource values remain language
    neutral. Only model-authored natural language follows the query locale.
    """

    locale = normalize_prompt_locale(value)
    if locale == "zh":
        return (
            "当前查询语言：简体中文（zh）。\n"
            "所有面向用户或描述性的自然语言内容都必须使用简体中文，包括思考摘要、计划、标题、说明、假设、结论、错误说明与可用性说明。\n"
            "JSON 键、工具名、action 名、代码、数据库标识符、字段名以及原始数据值必须保持原样，不得翻译。\n"
        )
    return (
        "Current query language: English (en).\n"
        "Use English for every user-facing or descriptive natural-language value, including reasoning summaries, plans, titles, explanations, assumptions, conclusions, errors, and availability notes.\n"
        "Keep JSON keys, tool names, action names, code, database identifiers, field names, and original data values unchanged.\n"
    )


def localized_payload_label(value: str | None, *, zh: str, en: str) -> str:
    """Choose a localized label used immediately before a structured payload."""

    return zh if normalize_prompt_locale(value) == "zh" else en
