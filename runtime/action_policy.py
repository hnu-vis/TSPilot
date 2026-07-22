"""Runtime action contract validation."""
from __future__ import annotations

from core.completion import evaluate_goal_completion
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
    "terminate",
}

TERMINAL_ACTIONS = {"format_answer", "terminate"}


def validate_action(request_state: RequestStateModel, action_name: str) -> tuple[bool, str | None]:
    if action_name not in VALID_ACTIONS:
        return False, f"Action '{action_name}' is not part of the runtime contract."
    if action_name == "todowrite" and request_state.todo_list:
        return (
            False,
            "A todo plan already exists. Runtime advances plan status after successful actions; choose the next analysis action or terminate.",
        )
    if action_name in TERMINAL_ACTIONS:
        evaluation = evaluate_goal_completion(request_state)
        request_state.completion_state["latest_goal"] = evaluation.model_dump()
        if not evaluation.can_answer:
            return (
                False,
                "Final answer is blocked because the current goal is not complete: "
                + evaluation.reason,
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
            "completion_state": request_state.completion_state,
            "recovery_hint": (
                "Choose exactly one allowed next action and return one JSON object. "
                "Do not call todowrite again when a plan already exists. "
                "Use the latest observation, bounded evidence previews, and artifact refs to decide whether to query, analyze, or terminate with the final answer. "
                "Todo state is progress context; it does not replace the ReAct Thought/Action/Observation loop."
            ),
        },
        error=reason,
        payload_truncated=False,
        payload_ref=None,
    )
