"""Runtime action contract validation."""
from __future__ import annotations

from core.completion import current_todo, evaluate_goal_completion, infer_evidence_needs, satisfied_needs
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
    allowed_by_plan, plan_reason = _validate_current_plan_step(request_state, action_name)
    if not allowed_by_plan:
        return False, plan_reason
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


def _validate_current_plan_step(request_state: RequestStateModel, action_name: str) -> tuple[bool, str | None]:
    todo = current_todo(request_state)
    if todo is None:
        return True, None

    task_type = str(todo.get("task_type") or "").strip().lower()
    expected_action = _action_for_task_type(task_type)
    if expected_action is None or expected_action == action_name:
        return True, None

    missing = _missing_current_todo_needs(request_state, todo)
    if action_name == "sql_query" and (_needs_database_query(missing) or _analysis_step_needs_base_evidence(request_state, task_type)):
        return True, None

    return (
        False,
        "Action does not match the active plan step. "
        f"Current task_type is '{task_type}', so choose '{expected_action}' "
        f"or first fill missing query evidence if required. Missing evidence: {missing or 'none'}.",
    )


def _missing_current_todo_needs(request_state: RequestStateModel, todo: dict) -> list[str]:
    needs = list(todo.get("evidence_needed") or infer_evidence_needs(todo) or [])
    satisfied = satisfied_needs(request_state)
    return [need for need in needs if not _need_satisfied_by_state(need, satisfied)]


def _needs_database_query(missing: list[str]) -> bool:
    return any(
        need in {"schema", "sample_rows", "count", "aggregate", "filtered_table", "time_series", "database_evidence"}
        for need in missing
    )


def _analysis_step_needs_base_evidence(request_state: RequestStateModel, task_type: str) -> bool:
    return task_type in {"insight", "anomaly", "forecast"} and request_state.latest_database_evidence is None


def _need_satisfied_by_state(need: str, satisfied: set[str]) -> bool:
    if need in satisfied:
        return True
    if need == "aggregate" and {"count", "filtered_table"} & satisfied:
        return True
    if need == "sample_rows" and {"filtered_table", "time_series"} & satisfied:
        return True
    if need == "database_evidence" and {"schema", "count", "aggregate", "filtered_table", "time_series"} & satisfied:
        return True
    return False


def _action_for_task_type(task_type: str) -> str | None:
    return {
        "plan": "todowrite",
        "query": "sql_query",
        "insight": "insight",
        "anomaly": "anomaly",
        "forecast": "forecast",
        "answer": "terminate",
        "rag": "rag",
        "skill": "skill",
    }.get(task_type)


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
                "When an active todo exists, choose the action matching its task_type unless a database query is still needed to fill missing query evidence."
            ),
        },
        error=reason,
        payload_truncated=False,
        payload_ref=None,
    )
