"""Runtime action contract validation."""
from __future__ import annotations

from schemas.state import RequestStateModel
from schemas.tool import ToolObservation

VALID_ACTIONS = {
    "todowrite",
    "sql_query",
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
    if action_name == "todowrite" and request_state.todo_list:
        return (
            False,
            "A todo plan already exists. Runtime advances plan status after successful actions; choose the next analysis action or format_answer.",
        )
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
            "recovery_hint": (
                "Choose exactly one allowed next action and return one JSON object. "
                "Do not call todowrite again when a plan already exists; continue with the current evidence gap."
            ),
        },
        error=reason,
        payload_truncated=False,
        payload_ref=None,
    )
