"""Runtime completion checks for ReAct task and goal progress."""
from __future__ import annotations

from dataclasses import dataclass, field
from schemas.state import RequestStateModel


@dataclass
class CompletionEvaluation:
    completed: bool
    reason: str
    missing_evidence: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    next_action_hint: str | None = None

    def model_dump(self) -> dict:
        return {
            "completed": self.completed,
            "reason": self.reason,
            "missing_evidence": self.missing_evidence,
            "evidence_refs": self.evidence_refs,
            "next_action_hint": self.next_action_hint,
        }


@dataclass
class GoalCompletionEvaluation:
    can_answer: bool
    reason: str
    missing_evidence: list[str] = field(default_factory=list)
    answerable_from: list[str] = field(default_factory=list)
    next_action_hint: str | None = None

    def model_dump(self) -> dict:
        return {
            "can_answer": self.can_answer,
            "reason": self.reason,
            "missing_evidence": self.missing_evidence,
            "answerable_from": self.answerable_from,
            "next_action_hint": self.next_action_hint,
        }


def normalize_todo_for_completion(todo: dict) -> dict:
    """Normalize todo shape without turning it into a runtime evidence contract."""
    normalized = dict(todo)
    task_type = str(normalized.get("task_type") or "").strip().lower()
    if task_type == "plan":
        task_type = "query"
        normalized["task_type"] = task_type
    normalized.pop("evidence_needed", None)
    normalized.setdefault("result_ref", None)
    normalized.setdefault("completion_reason", None)
    return normalized


def evaluate_step_completion(
    request_state: RequestStateModel,
    *,
    tool_name: str,
    full_payload: dict,
    thought: str | None = None,
    action_reason: str | None = None,
) -> CompletionEvaluation:
    """Advance visible todo progress only when the latest action appears to finish it."""
    current = current_todo(request_state)
    if current is None:
        return CompletionEvaluation(False, "No active todo step.", next_action_hint=None)

    refs = evidence_refs_for_payload(full_payload)
    current_index = next((index for index, todo in enumerate(request_state.todo_list) if todo is current), None)
    if current_index is not None:
        history = request_state.completion_state.setdefault("_todo_action_history", {})
        history.setdefault(str(current_index), []).append(tool_name)

    if tool_name in {"todowrite", "format_answer", "terminate"}:
        return CompletionEvaluation(
            False,
            f"Tool '{tool_name}' does not auto-complete ordinary todo steps.",
            evidence_refs=refs,
            next_action_hint="Continue with the next evidence or analysis action.",
        )

    current_task_type = str(current.get("task_type") or "").strip().lower()
    if current_task_type == "generic":
        current_task_type = ""
    if _action_matches_task_type(tool_name, current_task_type):
        return _evaluate_matching_action(
            request_state=request_state,
            current=current,
            tool_name=tool_name,
            full_payload=full_payload,
            refs=refs,
            thought=thought,
            action_reason=action_reason,
        )

    if _signals_transition_to_next_todo(request_state, thought=thought, action_reason=action_reason):
        return CompletionEvaluation(
            True,
            "Agent reasoning indicates the active todo is complete and it is moving to the next todo.",
            evidence_refs=refs,
        )

    return CompletionEvaluation(
        False,
        f"Latest action '{tool_name}' does not satisfy active todo type '{current_task_type or 'unspecified'}'.",
        evidence_refs=refs,
        next_action_hint=_hint_for_task_type(current_task_type),
    )


def evaluate_goal_completion(request_state: RequestStateModel) -> GoalCompletionEvaluation:
    if request_state.database_context is None:
        return GoalCompletionEvaluation(True, "No database context is required for this response.")

    refs = _all_answer_refs(request_state)
    if refs:
        return GoalCompletionEvaluation(True, "Available observations or artifacts exist for the final answer.", answerable_from=refs)

    return GoalCompletionEvaluation(
        can_answer=False,
        reason="No database-backed evidence or derived analysis is available yet.",
        missing_evidence=["database_evidence"],
        next_action_hint="Call sql_query to obtain grounded database evidence first.",
    )


def current_todo(request_state: RequestStateModel) -> dict | None:
    return next((todo for todo in request_state.todo_list if todo.get("status") == "in_progress"), None)


def _action_matches_task_type(tool_name: str, task_type: str) -> bool:
    if not task_type:
        return True
    if task_type == "query":
        return tool_name == "sql_query"
    if task_type == "answer":
        return tool_name in {"format_answer", "terminate"}
    return tool_name == task_type


def _evaluate_matching_action(
    *,
    request_state: RequestStateModel,
    current: dict,
    tool_name: str,
    full_payload: dict,
    refs: list[str],
    thought: str | None,
    action_reason: str | None,
) -> CompletionEvaluation:
    if tool_name == "sql_query":
        coverage = _task_coverage(full_payload)
        if coverage.get("requires_followup") or coverage.get("runtime_requires_followup"):
            missing = _string_items(coverage.get("missing_or_uncertain")) + _string_items(
                coverage.get("runtime_missing_or_uncertain")
            )
            return CompletionEvaluation(
                False,
                "SQL observation reports missing or uncertain task coverage.",
                missing_evidence=missing,
                evidence_refs=refs,
                next_action_hint=coverage.get("next_action_hint") or "Issue a focused sql_query for the missing item.",
            )
        if _query_returned_no_rows(full_payload):
            return CompletionEvaluation(
                False,
                "SQL query returned no rows; keep the current todo active until the empty result is explained or repaired.",
                missing_evidence=["non_empty_query_result"],
                evidence_refs=refs,
                next_action_hint="Validate filters, time range, and field selection with another sql_query or answer later with explicit caveats.",
            )
        if _is_schema_only_query(full_payload) and not _todo_mentions_schema(current):
            return CompletionEvaluation(
                False,
                "Schema-only evidence does not complete a user-visible query todo.",
                missing_evidence=["database_result"],
                evidence_refs=refs,
                next_action_hint="Query the requested rows, points, aggregate, or validation result.",
            )
        return CompletionEvaluation(
            True,
            "SQL observation appears to cover the active query todo.",
            evidence_refs=refs,
        )

    if tool_name in {"insight", "code_interpreter", "anomaly", "forecast", "rag", "skill"}:
        if not refs and not _has_non_empty_payload(full_payload):
            return CompletionEvaluation(
                False,
                f"Tool '{tool_name}' returned no usable artifact for the active todo.",
                missing_evidence=[f"{tool_name}_artifact"],
                next_action_hint=_hint_for_task_type(tool_name),
            )
        return CompletionEvaluation(
            True,
            f"Tool '{tool_name}' produced output for the active todo.",
            evidence_refs=refs,
        )

    if _signals_transition_to_next_todo(request_state, thought=thought, action_reason=action_reason):
        return CompletionEvaluation(
            True,
            "Agent reasoning indicates the active todo is complete.",
            evidence_refs=refs,
        )

    return CompletionEvaluation(
        False,
        "The latest tool output is not enough to complete the active todo.",
        evidence_refs=refs,
        next_action_hint=_hint_for_task_type(str(current.get("task_type") or "")),
    )


def _task_coverage(payload: dict) -> dict:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    coverage = diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {}
    return coverage


def _query_returned_no_rows(payload: dict) -> bool:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    coverage = _task_coverage(payload)
    if "query returned no rows" in _string_items(coverage.get("missing_or_uncertain")):
        return True
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    rows = data.get("rows") if isinstance(data.get("rows"), list) else None
    points = data.get("points") if isinstance(data.get("points"), list) else None
    summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
    row_count = summary_stats.get("rows_count")
    point_count = summary_stats.get("points_count")
    if isinstance(row_count, int) or isinstance(point_count, int):
        return row_count == 0 and (point_count == 0 or point_count is None)
    if points is not None and len(points) > 0:
        return False
    if rows is not None:
        return len(rows) == 0
    if points is not None:
        return len(points) == 0
    return False


def _is_schema_only_query(payload: dict) -> bool:
    return str(payload.get("result_type") or "").lower() in {"schema", "metric_list"}


def _todo_mentions_schema(todo: dict) -> bool:
    text = str(todo.get("content") or "").lower()
    return any(token in text for token in ("schema", "field", "structure", "字段", "结构", "数据源"))


def _has_non_empty_payload(payload: dict) -> bool:
    return any(value not in (None, "", [], {}) for value in payload.values())


def _signals_transition_to_next_todo(
    request_state: RequestStateModel,
    *,
    thought: str | None,
    action_reason: str | None,
) -> bool:
    text = f"{thought or ''} {action_reason or ''}".strip().lower()
    if not text:
        return False
    current_index = next((index for index, todo in enumerate(request_state.todo_list) if todo.get("status") == "in_progress"), None)
    if current_index is None or current_index + 1 >= len(request_state.todo_list):
        return False
    next_todo = str(request_state.todo_list[current_index + 1].get("content") or "").lower()
    transition_markers = (
        "next step",
        "now i need",
        "now i should",
        "continue to",
        "下一步",
        "接下来",
        "然后",
        "继续",
        "转向",
    )
    if not any(marker in text for marker in transition_markers):
        return False
    next_tokens = [token for token in next_todo.replace("，", " ").replace("。", " ").split() if token]
    if not next_tokens:
        return True
    return any(token in text for token in next_tokens[:6])


def _hint_for_task_type(task_type: str) -> str | None:
    return {
        "query": "Call sql_query with the missing filters, fields, aggregation, or time range.",
        "code_interpreter": "Run code_interpreter over the full evidence artifact.",
        "insight": "Run code_interpreter over the full evidence artifact.",
        "anomaly": "Run anomaly after time-series evidence is available.",
        "forecast": "Run forecast after time-series evidence is available.",
        "rag": "Call rag only if external knowledge is required.",
        "skill": "Invoke the requested packaged skill.",
        "answer": "Terminate only after final answer verification can pass.",
    }.get(task_type or "")


def _string_items(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def evidence_refs_for_payload(payload: dict) -> list[str]:
    refs = []
    for key, prefix in (
        ("evidence_id", "evidence"),
        ("analysis_id", "analysis"),
        ("insight_id", "insight"),
        ("forecast_id", "forecast"),
        ("anomaly_id", "anomaly"),
        ("skill_name", "skill"),
    ):
        value = payload.get(key)
        if value:
            refs.append(f"{prefix}:{value}")
    if payload.get("results") is not None:
        refs.append("rag:latest")
    return refs


def _all_answer_refs(request_state: RequestStateModel) -> list[str]:
    refs = []
    if request_state.latest_database_evidence is not None:
        refs.append(f"evidence:{request_state.latest_database_evidence.evidence_id}")
    refs.extend(f"analysis:{analysis_id}" for analysis_id in request_state.analysis_artifacts)
    if request_state.latest_insight is not None:
        refs.append(f"insight:{request_state.latest_insight.insight_id}")
    if request_state.latest_forecast is not None:
        refs.append(f"forecast:{request_state.latest_forecast.forecast_id}")
    if request_state.latest_anomaly is not None:
        refs.append(f"anomaly:{request_state.latest_anomaly.anomaly_id}")
    if request_state.latest_rag:
        refs.append("rag:latest")
    if request_state.latest_skill:
        refs.append(f"skill:{request_state.latest_skill.get('skill_name', 'latest')}")
    return refs
