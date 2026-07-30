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
    capabilities = {
        str(item).strip().lower()
        for item in (request_state.requested_capabilities or [])
        if str(item).strip()
    }
    if action_name == "rag" and request_state.database_context is not None and "rag" not in capabilities:
        return (
            False,
            "RAG is not required by the structured intent profile for this database task. "
            "Use database evidence, analysis, anomaly, forecast, or terminate with grounded caveats.",
        )
    constraints = runtime_action_constraints(request_state)
    required_actions = {
        str(item.get("action") or "").strip()
        for item in constraints.get("required_actions", [])
        if isinstance(item, dict) and str(item.get("action") or "").strip()
    }
    prohibited_actions = {
        str(item).strip()
        for item in constraints.get("prohibited_actions", [])
        if str(item).strip()
    }
    if action_name in prohibited_actions:
        return False, constraints.get("reason") or f"Action '{action_name}' is not allowed in the current state."
    if required_actions and action_name not in required_actions:
        return (
            False,
            (constraints.get("reason") or "The current state requires a different next action.")
            + f" Required actions: {sorted(required_actions)}.",
        )
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
        evaluation = evaluate_goal_completion(request_state)
        request_state.completion_state["latest_goal"] = evaluation.model_dump()
        goal_covered_despite_todos = _terminal_goal_covered_despite_todos(request_state, evaluation)
        unavailable_outputs_explained = (
            not _missing_specialized_tool_output(evaluation.missing_evidence)
            and _terminal_input_explains_unavailable_outputs(request_state, action_input)
        )
        active_todo = next((todo for todo in request_state.todo_list if todo.get("status") == "in_progress"), None)
        if active_todo is not None:
            active_type = str(active_todo.get("task_type") or "").strip().lower()
            if active_type != "answer" and not goal_covered_despite_todos and not unavailable_outputs_explained:
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
        if pending_non_answer and not goal_covered_despite_todos and not unavailable_outputs_explained:
            return (
                False,
                "Final answer is blocked because non-answer todo steps are still incomplete.",
            )
        if not evaluation.can_answer:
            if _missing_specialized_tool_output(evaluation.missing_evidence):
                return (
                    False,
                    "Final answer is blocked because the current goal is not complete: "
                    + evaluation.reason,
                )
            if unavailable_outputs_explained:
                return True, None
            return (
                False,
                "Final answer is blocked because the current goal is not complete: "
                + evaluation.reason,
            )
    return True, None


def _terminal_goal_covered_despite_todos(
    request_state: RequestStateModel,
    evaluation,
) -> bool:
    if not evaluation.can_answer:
        return False
    if not evaluation.answerable_from:
        return False
    gap = latest_gap_assessment(request_state)
    if not gap:
        return False
    if gap.get("can_answer") is not True:
        return False
    return not _gap_blocking_items(gap)


def _missing_specialized_tool_output(missing_evidence: list[str]) -> bool:
    missing = {
        str(item).strip().lower()
        for item in missing_evidence
        if str(item).strip()
    }
    return bool(missing & {"analysis", "forecast", "anomaly"})


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
        latest_goal = request_state.completion_state.get("latest_goal")
        if not isinstance(latest_goal, dict) or not isinstance(action_input, dict):
            return False
        blocking_items = [
            str(item).strip()
            for item in latest_goal.get("missing_evidence", [])
            if str(item).strip()
        ]
    else:
        blocking_items = _gap_blocking_items(gap)
        if not blocking_items:
            latest_goal = request_state.completion_state.get("latest_goal")
            if isinstance(latest_goal, dict):
                blocking_items = [
                    str(item).strip()
                    for item in latest_goal.get("missing_evidence", [])
                    if str(item).strip()
                ]
    if not blocking_items:
        return False
    unavailable_outputs = action_input.get("unavailable_outputs")
    unavailable_reason = str(action_input.get("unavailable_reason") or "").strip()
    if not isinstance(unavailable_outputs, list) or not unavailable_reason:
        return False
    if _latest_database_evidence_is_empty(request_state):
        return any(str(item).strip() for item in unavailable_outputs)
    unavailable_set = {
        str(item).strip()
        for item in unavailable_outputs
        if str(item).strip()
    }
    return all(item in unavailable_set for item in blocking_items)


def _latest_database_evidence_is_empty(request_state: RequestStateModel) -> bool:
    evidence = request_state.latest_database_evidence
    if evidence is None:
        return False
    data = evidence.data if isinstance(evidence.data, dict) else {}
    rows = data.get("rows")
    points = data.get("points")
    if isinstance(rows, list) and len(rows) > 0:
        return False
    if isinstance(points, list) and len(points) > 0:
        return False
    if isinstance(rows, list) or isinstance(points, list):
        return True
    series = data.get("series")
    if isinstance(series, list):
        return all(
            not isinstance(item, dict)
            or not isinstance(item.get("points"), list)
            or len(item.get("points")) == 0
            for item in series
        )
    return "no rows" in str(evidence.summary or "").lower()


def _gap_blocking_items(gap: dict) -> list[str]:
    items: list[str] = []
    values = gap.get("missing")
    if isinstance(values, list):
        items.extend(str(item).strip() for item in values if str(item).strip())
    return items


def _requires_initial_todo_plan(request_state: RequestStateModel) -> bool:
    if request_state.todo_list or request_state.database_context is None:
        return False
    profile = request_state.intent_profile if isinstance(request_state.intent_profile, dict) else {}
    if profile.get("needs_plan") is True:
        return True
    message = str(request_state.message or "")
    numbered_items = re.findall(r"(?<!\d)(?:\d+|[一二三四五六七八九十]+)(?:[\.\、\)]|\s+)\s*\S", message)
    bullet_items = re.findall(r"(?:^|\n)\s*[-*]\s+\S", message)
    return len(numbered_items) + len(bullet_items) >= 3


def runtime_action_constraints(request_state: RequestStateModel) -> dict:
    """Return runtime-owned next-action constraints derived from structured state."""

    if _requires_initial_todo_plan(request_state):
        return {
            "required_actions": [
                {
                    "action": "todowrite",
                    "reason": "The user requested a multi-step deliverable and no todo plan exists.",
                }
            ],
            "prohibited_actions": sorted(VALID_ACTIONS - {"todowrite"}),
            "missing_outputs": ["todo_plan"],
            "reason": "Create the initial todo plan before querying or answering.",
        }

    capabilities = {
        str(item).strip().lower()
        for item in (request_state.requested_capabilities or [])
        if str(item).strip()
    }
    has_database_evidence = not _latest_database_evidence_is_empty(request_state) and request_state.latest_database_evidence is not None
    required: list[dict] = []
    missing: list[str] = []

    if "anomaly" in capabilities and request_state.latest_anomaly is None:
        missing.append("anomaly")
        required.append(
            {
                "action": "anomaly" if has_database_evidence else "sql_query",
                "reason": "Anomaly output is required by the structured intent profile.",
                "input_guidance": {"database_evidence": "latest"} if has_database_evidence else {"constraints": {"evidence_shape": "raw_timeseries"}},
            }
        )
    elif "forecast" in capabilities and not _latest_forecast_is_usable(request_state):
        missing.append("forecast")
        required.append(
            {
                "action": "forecast" if has_database_evidence else "sql_query",
                "reason": "Forecast output is required by the structured intent profile.",
                "input_guidance": {"database_evidence": "latest", "horizon": "derive from user request"} if has_database_evidence else {"constraints": {"evidence_shape": "raw_timeseries"}},
            }
        )

    downstream_analysis = None
    specialized_covered = (
        ("anomaly" in capabilities and request_state.latest_anomaly is not None)
        or ("forecast" in capabilities and _latest_forecast_is_usable(request_state))
    )
    if not specialized_covered:
        downstream_analysis = _latest_query_requests_downstream_analysis(request_state)
    if downstream_analysis and request_state.latest_analysis_id is None:
        required.append(
            {
                "action": "code_interpreter",
                "reason": "Latest database evidence declares uncovered derived outputs for downstream analysis.",
                "input_guidance": {
                    "database_evidence": "latest",
                    "analysis_request": downstream_analysis,
                },
            }
        )
        missing.append("analysis")

    shape_recovery = _latest_query_shape_recovery(request_state)
    if shape_recovery:
        action = "sql_query"
        if has_database_evidence and shape_recovery.get("recommended_downstream_action") == "code_interpreter":
            action = "code_interpreter"
        required.append(
            {
                "action": action,
                "reason": "The previous sql_query failed dialect/query-task shape validation.",
                "input_guidance": shape_recovery,
            }
        )
        missing.append("query_shape_recovery")

    if not required:
        prohibited = []
        if request_state.database_context is not None and "rag" not in capabilities:
            prohibited.append("rag")
        return {
            "required_actions": [],
            "prohibited_actions": prohibited,
            "missing_outputs": [],
            "reason": "No runtime-enforced action constraint is active.",
        }
    prohibited = ["terminate"]
    if request_state.database_context is not None and "rag" not in capabilities:
        prohibited.append("rag")
    return {
        "required_actions": required[:1],
        "prohibited_actions": prohibited,
        "missing_outputs": missing,
        "reason": required[0]["reason"],
    }


def _latest_query_requests_downstream_analysis(request_state: RequestStateModel) -> dict | None:
    evidence = request_state.latest_database_evidence
    if evidence is None:
        return None
    diagnostics = evidence.diagnostics if isinstance(evidence.diagnostics, dict) else {}
    task_coverage = diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {}
    contract = task_coverage.get("query_task_contract") if isinstance(task_coverage.get("query_task_contract"), dict) else None
    if contract is None:
        llm_generation = diagnostics.get("llm_query_generation") if isinstance(diagnostics.get("llm_query_generation"), dict) else {}
        contract = llm_generation.get("query_task_contract") if isinstance(llm_generation.get("query_task_contract"), dict) else None
    if not isinstance(contract, dict):
        return None
    if str(contract.get("downstream_action") or "").strip().lower() != "code_interpreter":
        return None
    missing = task_coverage.get("missing") if isinstance(task_coverage.get("missing"), list) else []
    return {
        "goal": request_state.message,
        "query_task_contract": contract,
        "required_outputs": contract.get("required_outputs") or missing,
        "missing": missing,
    }


def _latest_forecast_is_usable(request_state: RequestStateModel) -> bool:
    forecast = request_state.latest_forecast
    if forecast is None:
        return False
    points = getattr(forecast, "forecast_points", None)
    return isinstance(points, list) and bool(points)


def _latest_query_shape_recovery(request_state: RequestStateModel) -> dict | None:
    latest = request_state.observations[-1] if request_state.observations else None
    if latest is None or latest.tool_name != "sql_query" or latest.success:
        return None
    payload = latest.payload if isinstance(latest.payload, dict) else {}
    error_type = str(payload.get("error_type") or "").strip()
    if error_type != "query_shape_invalid":
        return None
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else payload
    issues = diagnostics.get("query_shape_issues") if isinstance(diagnostics.get("query_shape_issues"), list) else []
    recommended_shape = next(
        (
            str(issue.get("recommended_shape") or "").strip()
            for issue in issues
            if isinstance(issue, dict) and str(issue.get("recommended_shape") or "").strip()
        ),
        None,
    )
    if recommended_shape != "raw_series":
        return None
    return {
        "constraints": {"evidence_shape": "raw_timeseries"},
        "recommended_shape": recommended_shape,
        "recommended_downstream_action": diagnostics.get("recommended_downstream_action"),
        "query_shape_issues": issues,
    }


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
