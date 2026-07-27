"""Runtime completion checks for ReAct task and goal progress."""
from __future__ import annotations

from dataclasses import dataclass, field
from schemas.agent_turn import PreviousObservationAssessment
from schemas.state import RequestStateModel
from schemas.tool import ToolObservation


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


def evaluate_goal_completion(request_state: RequestStateModel) -> GoalCompletionEvaluation:
    if request_state.database_context is None:
        return GoalCompletionEvaluation(True, "No database context is required for this response.")

    refs = _all_answer_refs(request_state)
    missing_required = _missing_required_tool_outputs(request_state)
    if missing_required:
        return GoalCompletionEvaluation(
            can_answer=False,
            reason="Required specialized tool output is missing for the requested analysis.",
            missing_evidence=missing_required,
            answerable_from=refs,
            next_action_hint=_next_action_for_missing_required(missing_required),
        )
    gap = latest_gap_assessment(request_state)
    contract_missing = _missing_contract_outputs(request_state, gap)
    if contract_missing:
        return GoalCompletionEvaluation(
            can_answer=False,
            reason="Task contract required outputs are not fully covered by the latest ReAct gap assessment.",
            missing_evidence=contract_missing,
            answerable_from=refs,
            next_action_hint="Choose the smallest next action that covers the missing task_contract output.",
        )
    if gap is not None:
        gap_missing = _gap_blocking_items(gap)
        can_answer = gap.get("can_answer")
        if can_answer is False or (gap_missing and can_answer is not True):
            return GoalCompletionEvaluation(
                can_answer=False,
                reason="Latest ReAct gap assessment says the user request is not fully covered.",
                missing_evidence=gap_missing or ["gap_assessment_not_answerable"],
                answerable_from=refs,
                next_action_hint=gap.get("next_action_reason") or "Choose the next action that fills the assessed evidence gap.",
            )
        if can_answer is True and refs:
            return GoalCompletionEvaluation(
                True,
                "Latest ReAct gap assessment says available evidence covers the user request.",
                answerable_from=refs,
            )
    if refs:
        return GoalCompletionEvaluation(True, "Available observations or artifacts exist for the final answer.", answerable_from=refs)

    return GoalCompletionEvaluation(
        can_answer=False,
        reason="No database-backed evidence or derived analysis is available yet.",
        missing_evidence=["database_evidence"],
        next_action_hint="Call sql_query to obtain grounded database evidence first.",
    )


def _missing_required_tool_outputs(request_state: RequestStateModel) -> list[str]:
    requirements = {
        str(item).strip().lower()
        for item in (request_state.requested_capabilities or [])
        if str(item).strip()
    }
    missing: list[str] = []
    if "forecast" in requirements and request_state.latest_forecast is None:
        missing.append("forecast")
    if "anomaly" in requirements and request_state.latest_anomaly is None:
        missing.append("anomaly")
    return missing


def _missing_contract_outputs(request_state: RequestStateModel, gap: dict | None) -> list[str]:
    contract = request_state.task_contract
    if contract is None:
        return []
    required = [
        output
        for output in contract.required_outputs
        if output.required
    ]
    if not required:
        return []
    if gap is None:
        return [output.id for output in required]
    if gap.get("can_answer") is not True:
        gap_missing = set(_gap_blocking_items(gap))
        if gap_missing:
            return [
                output.id
                for output in required
                if output.id in gap_missing or output.description in gap_missing
            ] or sorted(gap_missing)
        return [output.id for output in required]
    covered = {
        str(item).strip()
        for item in gap.get("covered", [])
        if str(item).strip()
    }
    missing = {
        str(item).strip()
        for item in gap.get("missing", [])
        if str(item).strip()
    }
    missing_required = []
    for output in required:
        keys = {output.id, output.description}
        if keys & missing:
            missing_required.append(output.id)
            continue
        if not keys & covered:
            missing_required.append(output.id)
    return missing_required


def _next_action_for_missing_required(missing: list[str]) -> str:
    if "forecast" in missing:
        return "Call forecast with the latest time-series evidence before answering."
    if "anomaly" in missing:
        return "Call anomaly with the latest time-series evidence before answering."
    return "Call the missing specialized analysis tool before answering."


def current_todo(request_state: RequestStateModel) -> dict | None:
    return next((todo for todo in request_state.todo_list if todo.get("status") == "in_progress"), None)


def apply_previous_observation_assessment(
    request_state: RequestStateModel,
    assessment: PreviousObservationAssessment | None,
) -> CompletionEvaluation:
    if assessment is not None:
        request_state.completion_state["latest_gap_assessment"] = gap_assessment_payload(assessment)

    current = current_todo(request_state)
    if current is None:
        return CompletionEvaluation(False, "No active todo step.", next_action_hint=None)
    if assessment is None:
        return CompletionEvaluation(False, "No previous observation assessment was provided.", next_action_hint=None)

    current_index = next((index for index, todo in enumerate(request_state.todo_list) if todo is current), None)
    if current_index is None:
        return CompletionEvaluation(False, "Active todo could not be located.", next_action_hint=None)

    previous_observation = request_state.observations[-1] if request_state.observations else None
    reconciled = _reconcile_todos_from_global_assessment(
        request_state,
        assessment,
        current_index=current_index,
        current=current,
        previous_observation=previous_observation,
    )
    if reconciled is not None:
        return reconciled

    if not assessment.completed_active_todo:
        evaluation = CompletionEvaluation(
            False,
            assessment.reason or "LLM assessment did not mark the active todo complete.",
            evidence_refs=assessment.evidence_refs,
            next_action_hint="Continue from the latest observation.",
        )
        request_state.completion_state["latest_step"] = _assessment_state(
            evaluation,
            tool_name=previous_observation.tool_name if previous_observation else None,
            todo_index=current_index,
            todo=current,
            accepted=False,
        )
        return evaluation

    evaluation = _hard_gate_previous_assessment(
        current=current,
        assessment=assessment,
        previous_observation=previous_observation,
    )
    accepted = evaluation.completed
    request_state.completion_state["latest_step"] = _assessment_state(
        evaluation,
        tool_name=previous_observation.tool_name if previous_observation else None,
        todo_index=current_index,
        todo=current,
        accepted=accepted,
    )
    if not accepted:
        return evaluation

    completed_todo = normalize_todo_for_completion(dict(current))
    completed_todo["status"] = "completed"
    completed_todo["result_ref"] = evaluation.evidence_refs[0] if evaluation.evidence_refs else completed_todo.get("result_ref")
    completed_todo["completion_reason"] = evaluation.reason
    request_state.todo_list[current_index] = completed_todo
    next_index = next((index for index, todo in enumerate(request_state.todo_list) if todo.get("status") == "pending"), None)
    if next_index is not None:
        next_todo = dict(request_state.todo_list[next_index])
        next_todo["status"] = "in_progress"
        request_state.todo_list[next_index] = next_todo
        request_state.plan_current_step = next_index + 1
        request_state.planning_complete = False
    else:
        request_state.plan_current_step = len(request_state.todo_list)
        request_state.planning_complete = True
    request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
    return evaluation


def gap_assessment_payload(assessment: PreviousObservationAssessment) -> dict:
    return {
        "completed_active_todo": assessment.completed_active_todo,
        "reason": assessment.reason,
        "evidence_refs": list(assessment.evidence_refs),
        "covered": list(assessment.covered),
        "missing": list(assessment.missing),
        "completed_todos": list(assessment.completed_todos),
        "next_active_todo": assessment.next_active_todo,
        "next_action_reason": assessment.next_action_reason,
        "can_answer": assessment.can_answer,
    }


def latest_gap_assessment(request_state: RequestStateModel) -> dict | None:
    gap = request_state.completion_state.get("latest_gap_assessment")
    return gap if isinstance(gap, dict) else None


def _gap_blocking_items(gap: dict) -> list[str]:
    items: list[str] = []
    values = gap.get("missing")
    if isinstance(values, list):
        items.extend(str(item).strip() for item in values if str(item).strip())
    return items


def _hard_gate_previous_assessment(
    *,
    current: dict,
    assessment: PreviousObservationAssessment,
    previous_observation: ToolObservation | None,
) -> CompletionEvaluation:
    if previous_observation is None:
        return CompletionEvaluation(False, "No previous observation exists for the assessment.", next_action_hint="Run a tool first.")
    if not previous_observation.success:
        return CompletionEvaluation(
            False,
            "Previous observation failed, so it cannot complete the active todo.",
            missing_evidence=["successful_observation"],
            next_action_hint="Repair the failed action or choose another grounded action.",
        )

    tool_name = previous_observation.tool_name
    payload = previous_observation.payload if isinstance(previous_observation.payload, dict) else {}
    current_task_type = str(current.get("task_type") or "").strip().lower()
    if current_task_type in {"generic", "plan"}:
        current_task_type = "" if current_task_type == "generic" else "query"

    if tool_name in {"todowrite"}:
        return CompletionEvaluation(False, "Todo planning observations do not complete business todo steps.")
    if tool_name == "terminate" and current_task_type != "answer":
        return CompletionEvaluation(False, "Terminal observations cannot complete non-answer todo steps.")
    if current_task_type == "answer" and tool_name != "terminate":
        return CompletionEvaluation(False, "Only a terminal final-answer observation can complete an answer todo.")
    if current_task_type and not _action_matches_task_type(tool_name, current_task_type):
        return CompletionEvaluation(
            False,
            f"Previous observation '{tool_name}' does not match active todo type '{current_task_type}'.",
            next_action_hint=_hint_for_task_type(current_task_type),
        )

    refs = assessment.evidence_refs or evidence_refs_for_payload(payload)
    if tool_name == "sql_query":
        if _query_returned_no_rows(payload) and not _assessment_accepts_empty_result(assessment):
            return CompletionEvaluation(
                False,
                "SQL query returned no rows; LLM assessment did not explain why the empty result itself satisfies the todo.",
                missing_evidence=["non_empty_query_result"],
                evidence_refs=refs,
                next_action_hint="Repair the query or explicitly explain why the empty result is the requested outcome.",
            )
        if _is_schema_only_query(payload) and not _todo_mentions_schema(current):
            return CompletionEvaluation(
                False,
                "Schema-only evidence cannot complete a non-schema query todo.",
                missing_evidence=["database_result"],
                evidence_refs=refs,
                next_action_hint="Query the requested rows, aggregate, or validation result.",
            )
        if not refs and not _has_non_empty_payload(payload):
            return CompletionEvaluation(
                False,
                "SQL observation did not provide evidence or a usable payload.",
                missing_evidence=["database_evidence"],
                next_action_hint="Run a grounded sql_query.",
            )
    elif tool_name in {"code_interpreter", "anomaly", "forecast", "rag", "skill"}:
        if not refs and not _has_non_empty_payload(payload):
            return CompletionEvaluation(
                False,
                f"Observation '{tool_name}' did not provide a usable artifact.",
                missing_evidence=[f"{tool_name}_artifact"],
                next_action_hint=_hint_for_task_type(current_task_type or tool_name),
            )
    elif tool_name == "terminate":
        if not _has_final_answer_payload(payload):
            return CompletionEvaluation(
                False,
                "Terminal observation did not provide a usable final answer.",
                missing_evidence=["final_answer"],
                evidence_refs=refs,
                next_action_hint="Assemble the final answer again after evidence is ready.",
            )

    return CompletionEvaluation(
        True,
        assessment.reason or "LLM assessment marked the active todo complete based on the previous observation.",
        evidence_refs=refs,
    )


def _reconcile_todos_from_global_assessment(
    request_state: RequestStateModel,
    assessment: PreviousObservationAssessment,
    *,
    current_index: int,
    current: dict,
    previous_observation: ToolObservation | None,
) -> CompletionEvaluation | None:
    if str(current.get("task_type") or "").strip().lower() == "answer":
        return None
    gap = gap_assessment_payload(assessment)
    if assessment.can_answer is not True:
        return None
    if _gap_blocking_items(gap):
        return None
    if _missing_contract_outputs(request_state, gap):
        return None

    refs = assessment.evidence_refs or _all_answer_refs(request_state)
    invalid_refs = _invalid_evidence_refs(request_state, refs)
    if invalid_refs:
        evaluation = CompletionEvaluation(
            False,
            "LLM progress assessment referenced evidence that does not exist.",
            missing_evidence=invalid_refs,
            evidence_refs=refs,
            next_action_hint="Use only real evidence, analysis, forecast, anomaly, rag, or skill refs from the current context.",
        )
        request_state.completion_state["latest_step"] = _assessment_state(
            evaluation,
            tool_name=previous_observation.tool_name if previous_observation else None,
            todo_index=current_index,
            todo=current,
            accepted=False,
        )
        return evaluation

    completed_any = False
    for index, todo in enumerate(request_state.todo_list):
        if todo.get("status") == "completed":
            continue
        task_type = str(todo.get("task_type") or "").strip().lower()
        if task_type == "answer":
            continue
        completed_todo = normalize_todo_for_completion(dict(todo))
        completed_todo["status"] = "completed"
        completed_todo["result_ref"] = refs[0] if refs else completed_todo.get("result_ref")
        completed_todo["completion_reason"] = assessment.reason or "LLM progress assessment says task contract evidence is covered."
        request_state.todo_list[index] = completed_todo
        completed_any = True

    next_index = _requested_next_active_todo_index(request_state, assessment)
    if next_index is None:
        next_index = next(
            (
                index for index, todo in enumerate(request_state.todo_list)
                if todo.get("status") != "completed"
                and str(todo.get("task_type") or "").strip().lower() == "answer"
            ),
            None,
        )
    if next_index is None:
        next_index = next((index for index, todo in enumerate(request_state.todo_list) if todo.get("status") == "pending"), None)

    if next_index is not None:
        for index, todo in enumerate(request_state.todo_list):
            if todo.get("status") == "in_progress":
                updated = dict(todo)
                updated["status"] = "pending"
                request_state.todo_list[index] = updated
        next_todo = dict(request_state.todo_list[next_index])
        if next_todo.get("status") != "completed":
            next_todo["status"] = "in_progress"
            request_state.todo_list[next_index] = next_todo
            request_state.plan_current_step = next_index + 1
            request_state.planning_complete = False
    else:
        request_state.plan_current_step = len(request_state.todo_list)
        request_state.planning_complete = True

    evaluation = CompletionEvaluation(
        bool(completed_any),
        assessment.reason or "LLM progress assessment reconciled todo progress with covered task contract outputs.",
        evidence_refs=refs,
        next_action_hint=assessment.next_action_reason,
    )
    request_state.completion_state["latest_step"] = _assessment_state(
        evaluation,
        tool_name=previous_observation.tool_name if previous_observation else None,
        todo_index=current_index,
        todo=current,
        accepted=evaluation.completed,
    )
    request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
    return evaluation


def _requested_next_active_todo_index(
    request_state: RequestStateModel,
    assessment: PreviousObservationAssessment,
) -> int | None:
    target = assessment.next_active_todo
    if target is None:
        return None
    if isinstance(target, int):
        return next(
            (
                index for index, todo in enumerate(request_state.todo_list)
                if todo.get("priority") == target or index + 1 == target
            ),
            None,
        )
    normalized_target = str(target).strip().lower()
    if not normalized_target:
        return None
    return next(
        (
            index for index, todo in enumerate(request_state.todo_list)
            if str(todo.get("content") or "").strip().lower() == normalized_target
            or str(todo.get("task_type") or "").strip().lower() == normalized_target
        ),
        None,
    )


def _invalid_evidence_refs(request_state: RequestStateModel, refs: list[str]) -> list[str]:
    if not refs:
        return ["evidence_refs"]
    valid_refs = set(_all_answer_refs(request_state))
    if request_state.latest_database_evidence is not None:
        valid_refs.add(request_state.latest_database_evidence.evidence_id)
    valid_refs.update(request_state.analysis_artifacts.keys())
    valid_refs.update(f"analysis:{analysis_id}" for analysis_id in request_state.analysis_artifacts)
    if request_state.latest_forecast is not None:
        valid_refs.add(request_state.latest_forecast.forecast_id)
    if request_state.latest_anomaly is not None:
        valid_refs.add(request_state.latest_anomaly.anomaly_id)
    for observation in request_state.observations:
        if observation.payload_ref:
            valid_refs.add(observation.payload_ref)
        payload = observation.payload if isinstance(observation.payload, dict) else {}
        valid_refs.update(evidence_refs_for_payload(payload))
    return [
        ref for ref in refs
        if str(ref).strip() and str(ref).strip() not in valid_refs
    ]


def _assessment_state(
    evaluation: CompletionEvaluation,
    *,
    tool_name: str | None,
    todo_index: int,
    todo: dict,
    accepted: bool,
) -> dict:
    return {
        **evaluation.model_dump(),
        "tool_name": tool_name,
        "todo_index": todo_index,
        "todo": todo,
        "assessment_accepted": accepted,
    }


def _action_matches_task_type(tool_name: str, task_type: str) -> bool:
    if not task_type:
        return True
    if task_type == "query":
        return tool_name == "sql_query"
    if task_type == "answer":
        return tool_name == "terminate"
    return tool_name == task_type


def _task_coverage(payload: dict) -> dict:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    coverage = diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {}
    return coverage


def _query_returned_no_rows(payload: dict) -> bool:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    coverage = _task_coverage(payload)
    missing_items = _string_items(coverage.get("missing")) or _string_items(coverage.get("missing_or_uncertain"))
    if "query returned no rows" in missing_items:
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


def _has_final_answer_payload(payload: dict) -> bool:
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return True
    sections = payload.get("sections")
    if isinstance(sections, list) and sections:
        return True
    return False


def _assessment_accepts_empty_result(assessment: PreviousObservationAssessment) -> bool:
    text = str(assessment.reason or "").lower()
    return any(
        token in text
        for token in (
            "empty result",
            "no rows",
            "no matching",
            "no anomalies",
            "无结果",
            "空结果",
            "没有匹配",
            "未发现",
            "无异常",
        )
    )


def _hint_for_task_type(task_type: str) -> str | None:
    return {
        "query": "Call sql_query with the missing filters, fields, aggregation, or time range.",
        "code_interpreter": "Run code_interpreter over the full evidence artifact.",
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
    if request_state.latest_forecast is not None:
        refs.append(f"forecast:{request_state.latest_forecast.forecast_id}")
    if request_state.latest_anomaly is not None:
        refs.append(f"anomaly:{request_state.latest_anomaly.anomaly_id}")
    if request_state.latest_rag:
        refs.append("rag:latest")
    if request_state.latest_skill:
        refs.append(f"skill:{request_state.latest_skill.get('skill_name', 'latest')}")
    return refs
