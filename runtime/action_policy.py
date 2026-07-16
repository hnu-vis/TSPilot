"""Minimal action contract validation.

The ReAct model owns next-step decisions. Runtime validation only rejects
actions that cannot be resolved to any tool.
"""
from __future__ import annotations

from schemas.state import RequestStateModel
from schemas.tool import ToolObservation

VALID_ACTIONS = {
    "todowrite",
    "query_database",
    "insight",
    "forecast",
    "anomaly",
    "rag",
    "skill",
    "format_answer",
}


def validate_action(request_state: RequestStateModel, action_name: str) -> tuple[bool, str | None]:
    if action_name not in VALID_ACTIONS:
        return False, f"Action '{action_name}' is not part of the runtime contract."
    return True, None


def build_policy_observation(
    request_state: RequestStateModel,
    action_name: str,
    reason: str,
) -> ToolObservation:
    return ToolObservation(
        tool_name=action_name,
        success=False,
        summary=reason,
        payload={
            "valid_actions": sorted(VALID_ACTIONS),
            "recovery_hint": "Choose exactly one action from valid_actions and return one JSON object.",
        },
        error=reason,
        payload_truncated=False,
        payload_ref=None,
    )
