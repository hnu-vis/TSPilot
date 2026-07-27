"""Runtime action contract validation."""
from __future__ import annotations

import re

from core.completion import evaluate_goal_completion, latest_gap_assessment
from schemas.state import RequestStateModel
from schemas.tool import ToolObservation

VALID_ACTIONS = {
    "todowrite",
    "sql_query",
    "code_interpreter",
    "forecast",
    "anomaly",
    "rag",
    "skill",
    "terminate",
}

TERMINAL_ACTIONS = {"terminate"}


def validate_action(
    request_state: RequestStateModel,
    action_name: str,
    action_input: dict | None = None,
    *,
    action_reason: str | None = None,
) -> tuple[bool, str | None]:
    if action_name not in VALID_ACTIONS:
        return False, f"Action '{action_name}' is not part of the runtime contract."
    if action_name == "todowrite" and request_state.todo_list:
        return (
            False,
            "A todo plan already exists. Runtime advances plan status after successful actions; choose the next analysis action or terminate.",
        )
    if action_name != "todowrite" and _requires_initial_todo_plan(request_state):
        return (
            False,
            "This request has an explicit multi-deliverable list. Create the initial todo plan with todowrite before querying or answering.",
        )
    repeat_failure_reason = _repeated_failed_action_without_strategy_change(
        request_state,
        action_name,
        action_reason=action_reason,
    )
    if repeat_failure_reason:
        return False, repeat_failure_reason
    if action_name in TERMINAL_ACTIONS:
        active_todo = next((todo for todo in request_state.todo_list if todo.get("status") == "in_progress"), None)
        if active_todo is not None:
            active_type = str(active_todo.get("task_type") or "").strip().lower()
            if active_type != "answer":
                return (
                    False,
                    "Final answer is blocked because the active todo is not an answer step. "
                    "Assess the previous observation and complete the active todo before terminating.",
                )
        pending_non_answer = [
            todo for todo in request_state.todo_list
            if todo.get("status") != "completed"
            and str(todo.get("task_type") or "").strip().lower() != "answer"
        ]
        if pending_non_answer:
            return (
                False,
                "Final answer is blocked because non-answer todo steps are still incomplete.",
            )
        evaluation = evaluate_goal_completion(request_state)
        request_state.completion_state["latest_goal"] = evaluation.model_dump()
        if not evaluation.can_answer:
            if _terminal_input_explains_unavailable_outputs(request_state, action_input):
                return True, None
            return (
                False,
                "Final answer is blocked because the current goal is not complete: "
                + evaluation.reason,
            )
    return True, None


def _repeated_failed_action_without_strategy_change(
    request_state: RequestStateModel,
    action_name: str,
    *,
    action_reason: str | None,
) -> str | None:
    if action_name in TERMINAL_ACTIONS or action_name == "todowrite":
        return None
    failures: list[ToolObservation] = []
    for observation in reversed(request_state.observations):
        if observation.tool_name != action_name:
            break
        if observation.success:
            return None
        failures.append(observation)
        if len(failures) >= 2:
            break
    if len(failures) < 2:
        return None
    gap = latest_gap_assessment(request_state) or {}
    strategy_text = " ".join(
        str(value).strip()
        for value in (action_reason, gap.get("next_action_reason"))
        if str(value or "").strip()
    )
    if strategy_text:
        return None
    return (
        f"Action '{action_name}' has failed repeatedly. "
        "Assess the latest failure and provide a changed next_action_reason/action_reason before retrying the same tool."
    )


def _terminal_input_explains_unavailable_outputs(
    request_state: RequestStateModel,
    action_input: dict | None,
) -> bool:
    gap = latest_gap_assessment(request_state)
    if not gap or not isinstance(action_input, dict):
        return False
    blocking_items = _gap_blocking_items(gap)
    if not blocking_items:
        return False
    unavailable_outputs = action_input.get("unavailable_outputs")
    unavailable_reason = str(action_input.get("unavailable_reason") or "").strip()
    if not isinstance(unavailable_outputs, list) or not unavailable_reason:
        return False
    unavailable_set = {
        str(item).strip()
        for item in unavailable_outputs
        if str(item).strip()
    }
    return all(item in unavailable_set for item in blocking_items)


def _gap_blocking_items(gap: dict) -> list[str]:
    items: list[str] = []
    values = gap.get("missing")
    if isinstance(values, list):
        items.extend(str(item).strip() for item in values if str(item).strip())
    return items


def _requires_initial_todo_plan(request_state: RequestStateModel) -> bool:
    if request_state.todo_list or request_state.database_context is None:
        return False
    message = str(request_state.message or "")
    numbered_items = re.findall(r"(?<!\d)(?:\d+|[一二三四五六七八九十]+)[\.\、\)]\s*\S", message)
    bullet_items = re.findall(r"(?:^|\n)\s*[-*]\s+\S", message)
    return len(numbered_items) + len(bullet_items) >= 3


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
