"""Structured request capability profile helpers."""
from __future__ import annotations

import re
from typing import Any


CAPABILITY_ORDER = ("query", "analysis", "forecast", "anomaly")


def build_intent_profile_fallback(message: str) -> dict[str, Any]:
    """Build a minimal capability profile without modeling user-visible facts."""

    capabilities = _infer_requested_capabilities(message)
    return {
        "source": "fallback",
        "primary_goal": message,
        "requested_capabilities": capabilities,
        "required_outputs": _answer_requirements_from_capabilities(capabilities),
        "needs_plan": False,
    }


def normalize_intent_profile(raw: Any, *, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return fallback
    capabilities = raw.get("requested_capabilities")
    if not isinstance(capabilities, list):
        capabilities = fallback.get("requested_capabilities", [])
    capabilities = _normalize_capabilities([str(item) for item in capabilities])
    required_outputs = raw.get("required_outputs")
    if not isinstance(required_outputs, list):
        required_outputs = _answer_requirements_from_capabilities(capabilities)
    return {
        "source": str(raw.get("source") or "llm"),
        "primary_goal": str(raw.get("primary_goal") or fallback.get("primary_goal") or ""),
        "requested_capabilities": capabilities,
        "required_outputs": _normalize_capabilities([str(item) for item in required_outputs], allow_conclusion=True),
        "needs_plan": bool(raw.get("needs_plan", fallback.get("needs_plan", False))),
    }


def apply_intent_profile_to_state(request_state, profile: dict[str, Any]) -> None:
    request_state.intent_profile = profile
    capabilities = list(profile.get("requested_capabilities") or [])
    request_state.requested_capabilities = capabilities


def _infer_requested_capabilities(message: str) -> list[str]:
    normalized = message.lower()
    capabilities = ["query"]
    if any(token in normalized for token in ("预测", "预估", "forecast", "predict", "prediction")):
        capabilities.append("forecast")
    if any(token in normalized for token in ("异常", "离群", "尖峰", "异常点", "outlier", "anomaly", "spike", "dip")):
        capabilities.append("anomaly")
    if _looks_analytical(normalized) and "analysis" not in capabilities:
        capabilities.append("analysis")
    return _normalize_capabilities(capabilities)


def _looks_analytical(message: str) -> bool:
    if any(
        token in message
        for token in (
            "分析",
            "统计",
            "计算",
            "比较",
            "趋势",
            "走势",
            "涨跌",
            "变化",
            "最高",
            "最低",
            "最大",
            "最小",
            "analysis",
            "analyze",
            "calculate",
            "compare",
            "trend",
            "change",
        )
    ):
        return True
    return bool(re.search(r"(?<![A-Za-z0-9_])(max|min)(?![A-Za-z0-9_])", message))


def _answer_requirements_from_capabilities(capabilities: list[str]) -> list[str]:
    requirements = ["conclusion"]
    requirements.extend(item for item in capabilities if item != "query")
    return _dedupe(requirements)


def _normalize_capabilities(values: list[str], *, allow_conclusion: bool = False) -> list[str]:
    aliases = {
        "outlier": "anomaly",
        "prediction": "forecast",
        "predict": "forecast",
        "statistics": "analysis",
        "statistical_summary": "analysis",
    }
    allowed = set(CAPABILITY_ORDER)
    if allow_conclusion:
        allowed.add("conclusion")
    normalized = []
    for value in values:
        item = aliases.get(str(value).strip().lower(), str(value).strip().lower())
        if item in allowed:
            normalized.append(item)
    return _dedupe(normalized)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
