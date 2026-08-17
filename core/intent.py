"""Structured request capability profile helpers."""
from __future__ import annotations

from typing import Any

from core.harness import default_capability_registry


CAPABILITY_ORDER = ("query", "analysis", "forecast", "anomaly", "visualization", "external_knowledge", "skill")


def build_intent_profile_fallback(message: str) -> dict[str, Any]:
    """Build a minimal safe profile when no structured LLM intent is available."""

    capabilities = ["query"]
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


def _answer_requirements_from_capabilities(capabilities: list[str]) -> list[str]:
    requirements = ["conclusion"]
    requirements.extend(item for item in capabilities if item != "query")
    return _dedupe(requirements)


def _normalize_capabilities(values: list[str], *, allow_conclusion: bool = False) -> list[str]:
    registry = default_capability_registry()
    allowed = set(CAPABILITY_ORDER)
    if allow_conclusion:
        allowed.add("conclusion")
        allowed.add("answer")
    normalized = []
    for value in values:
        item = registry.normalize_id(value)
        if item == "answer" and allow_conclusion:
            item = "conclusion"
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
