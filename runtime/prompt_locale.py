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
            "时间展示规则：结构化数据、JSON、引用、定位器与代码中的时间值保持原样；"
            "但在面向用户的自然语言中提到绝对时间时，必须改写为符合中文习惯且仍然精确的表达，"
            "例如将 2023-01-04T23:48:00+00:00 写为“2023年1月4日 23:48（UTC）”。"
            "即使这些自然语言位于 JSON 字符串字段内，本规则也同样适用。"
            "保留问题和证据所需的最小时间精度并明确时区；除非用户明确要求原始格式，"
            "不要在标题、摘要、段落、图表标题或标注文字中直接展示 ISO 8601 字符串。\n"
        )
    return (
        "Current query language: English (en).\n"
        "Use English for every user-facing or descriptive natural-language value, including reasoning summaries, plans, titles, explanations, assumptions, conclusions, errors, and availability notes.\n"
        "Keep JSON keys, tool names, action names, code, database identifiers, field names, and original data values unchanged.\n"
        "Time presentation rule: keep timestamp values unchanged in structured data, JSON, citations, locators, and code; "
        "when an absolute time appears in user-facing prose, render it as precise, natural English, for example "
        "render 2023-01-04T23:48:00+00:00 as 'Jan 4, 2023 at 11:48 PM UTC'. "
        "This rule also applies when that prose is carried in a JSON string field. "
        "Preserve the smallest time precision required by the question and evidence and state the time zone. "
        "Unless the user explicitly asks for the raw format, do not expose ISO 8601 strings directly in titles, "
        "summaries, paragraphs, chart titles, or annotation text.\n"
    )


def localized_payload_label(value: str | None, *, zh: str, en: str) -> str:
    """Choose a localized label used immediately before a structured payload."""

    return zh if normalize_prompt_locale(value) == "zh" else en
