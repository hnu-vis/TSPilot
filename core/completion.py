"""Runtime completion checks for ReAct task and goal progress."""
from __future__ import annotations

from dataclasses import dataclass, field
from schemas.agent_turn import PreviousObservationAssessment
from schemas.state import RequestStateModel
from schemas.tool import ToolObservation
from core.harness import default_capability_registry


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
    if task_type in {"plan", "list", "todo_list", "todos", "planning"}:
        task_type = "query"
        normalized["task_type"] = task_type
    elif task_type in {"data", "dataset", "timeseries", "time_series", "series", "records"}:
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
    analysis_quality_gaps = _analysis_quality_gaps(request_state)
    if analysis_quality_gaps:
        return GoalCompletionEvaluation(
            can_answer=False,
            reason="Derived analysis is inconsistent with detected anomalies on the same evidence.",
            missing_evidence=analysis_quality_gaps,
            answerable_from=refs,
            next_action_hint=(
                "Rerun code_interpreter on the affected evidence and include transparent outlier treatment "
                "details, or explicitly compute metrics on the anomaly-adjusted series."
            ),
        )
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
            if gap_missing and _gap_missing_is_covered_by_artifacts(request_state, gap_missing):
                return GoalCompletionEvaluation(
                    True,
                    "Verified artifacts cover the latest ReAct gap assessment missing outputs.",
                    answerable_from=refs,
                )
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
    if (
        _requires_specialized_capability(request_state, requirements, "forecast")
        and not _latest_forecast_is_usable(request_state)
        and not _data_facts_cover_specialized_capability(request_state, "forecast")
    ):
        missing.append("forecast")
    if (
        _requires_specialized_capability(request_state, requirements, "anomaly")
        and request_state.latest_anomaly is None
        and not _data_facts_cover_specialized_capability(request_state, "anomaly")
    ):
        missing.append("anomaly")
    if (
        _requires_code_analysis(request_state)
        and request_state.latest_analysis_id is None
        and not _latest_database_evidence_is_empty(request_state)
        and not _data_facts_cover_required_analysis(request_state)
    ):
        missing.append("analysis")
    return missing


def _latest_database_evidence_is_empty(request_state: RequestStateModel) -> bool:
    evidence = request_state.latest_database_evidence
    if evidence is None:
        return False
    data = evidence.data if isinstance(evidence.data, dict) else {}
    row_like_keys = ("rows", "points")
    for key in row_like_keys:
        value = data.get(key)
        if isinstance(value, list):
            return len(value) == 0
    series = data.get("series")
    if isinstance(series, list):
        return all(
            not isinstance(item, dict)
            or not isinstance(item.get("points"), list)
            or len(item.get("points")) == 0
            for item in series
        )
    return "no rows" in str(evidence.summary or "").lower()


def _requires_specialized_capability(
    request_state: RequestStateModel,
    requirements: set[str],
    capability: str,
) -> bool:
    if capability not in requirements:
        return False
    contract = request_state.task_contract
    if contract is None:
        return True
    return any(
        output.required
        and str(output.evidence_kind or "").strip().lower() == capability
        for output in contract.required_outputs
    )


def _requires_code_analysis(request_state: RequestStateModel) -> bool:
    contract = request_state.task_contract
    if contract is None:
        return any(
            str(item).strip().lower() == "analysis"
            for item in (request_state.requested_capabilities or [])
        )
    for output in contract.required_outputs:
        if not output.required:
            continue
        if _contract_output_requires_code_analysis(output) and not _contract_output_is_covered_by_verified_fact(request_state, output):
            return True
    return False


def _data_facts_cover_required_analysis(request_state: RequestStateModel) -> bool:
    """Return whether current verified facts already ground the requested answer.

    Specialized tools should be required because the answer contract has an
    uncovered derived output, not merely because intent classification labeled
    the request as analytical. A database aggregate or point lookup represented
    as a verified DataFact is valid evidence for precise numeric answers.
    """

    contract = request_state.task_contract
    verified = _verified_data_facts(request_state)
    if not verified:
        return False
    if contract is None:
        return True
    required = [output for output in contract.required_outputs if output.required]
    if not required:
        return True
    analysis_outputs = [
        output
        for output in required
        if _contract_output_requires_code_analysis(output)
    ]
    if not analysis_outputs:
        return True
    return all(_contract_output_is_covered_by_verified_fact(request_state, output) for output in analysis_outputs)


def _data_facts_cover_specialized_capability(request_state: RequestStateModel, capability: str) -> bool:
    allowed_sources = {capability.strip().lower()}
    return any(_fact_has_allowed_source(fact, allowed_sources) for fact in _verified_data_facts(request_state))


def _verified_data_facts(request_state: RequestStateModel) -> list:
    facts = list(getattr(getattr(request_state, "fact_set", None), "facts", []) or [])
    verified = []
    for fact in facts:
        if getattr(fact, "status", None) != "verified":
            continue
        if getattr(fact, "fact_type", None) == "data_coverage":
            continue
        if not getattr(fact, "evidence_refs", None):
            continue
        verified.append(fact)
    return verified


def _contract_output_is_covered_by_verified_fact(request_state: RequestStateModel, output) -> bool:
    facts = _verified_data_facts(request_state)
    if not facts:
        return False
    output_keys = _contract_output_keys(output)
    allowed_sources = _contract_allowed_source_types(output)
    for fact in facts:
        if output_keys and _fact_keys(fact) & output_keys:
            if not allowed_sources or _fact_has_allowed_source(fact, allowed_sources):
                return True
        if not output_keys and allowed_sources and _fact_has_allowed_source(fact, allowed_sources):
            return True
    return False


def _contract_output_keys(output) -> set[str]:
    keys = {
        str(getattr(output, "id", "") or "").strip().lower(),
        str(getattr(output, "output_type", "") or "").strip().lower(),
    }
    keys.update(str(item).strip().lower() for item in getattr(output, "measures", []) or [])
    keys.update(str(item).strip().lower() for item in getattr(output, "dimensions", []) or [])
    return {key for key in keys if key}


def _fact_keys(fact) -> set[str]:
    keys = {
        str(getattr(fact, "fact_id", "") or "").strip().lower(),
        str(getattr(fact, "fact_key", "") or "").strip().lower(),
        str(getattr(fact, "name", "") or "").strip().lower(),
        str(getattr(fact, "fact_type", "") or "").strip().lower(),
        str(getattr(fact, "subject", "") or "").strip().lower(),
        str(getattr(fact, "unit", "") or "").strip().lower(),
    }
    dimensions = getattr(fact, "dimensions", None)
    if isinstance(dimensions, dict):
        for key, value in dimensions.items():
            keys.add(str(key).strip().lower())
            if isinstance(value, (str, int, float, bool)):
                keys.add(str(value).strip().lower())
    trace = getattr(fact, "calculation_trace", None)
    if isinstance(trace, dict):
        for key in ("value_key", "metric_key", "operator", "operation", "measure", "field"):
            value = trace.get(key)
            if isinstance(value, (str, int, float, bool)):
                keys.add(str(value).strip().lower())
    keys.update(str(item or "").strip().lower() for item in getattr(fact, "derived_from", []) or [])
    return {key for key in keys if key}


def _contract_allowed_source_types(output) -> set[str]:
    evidence_kind = str(getattr(output, "evidence_kind", "") or "").strip().lower()
    if not evidence_kind:
        return set()
    if evidence_kind in {"query", "database", "database_evidence", "sql", "raw"}:
        return {"query", "database", "database_evidence", "sql_query"}
    if evidence_kind in {"analysis", "derived", "statistical", "computed", "calculated"}:
        return {"analysis", "code_interpreter"}
    if evidence_kind == "forecast":
        return {"forecast"}
    if evidence_kind == "anomaly":
        return {"anomaly"}
    return {evidence_kind}


def _fact_has_allowed_source(fact, allowed_sources: set[str]) -> bool:
    method = str(getattr(fact, "method", "") or "").strip().lower()
    if method in allowed_sources:
        return True
    for ref in getattr(fact, "evidence_refs", []) or []:
        if str(getattr(ref, "source_type", "") or "").strip().lower() in allowed_sources:
            return True
    return False


def _contract_output_requires_code_analysis(output) -> bool:
    evidence_kind = str(output.evidence_kind or "").strip().lower()
    if evidence_kind in {"derived", "analysis", "statistical", "computed", "calculated"}:
        return True

    searchable = " ".join(
        [
            str(output.id or ""),
            str(output.description or ""),
            str(output.output_type or ""),
            " ".join(str(item) for item in output.measures),
            " ".join(str(item) for item in output.dimensions),
            str(output.success_criteria or ""),
        ]
    ).lower()
    analysis_terms = {
        "analysis",
        "analyze",
        "computed",
        "calculated",
        "derived",
        "change",
        "pct",
        "percent",
        "percentage",
        "ratio",
        "rate",
        "delta",
        "growth",
        "return",
        "trend",
        "volatility",
        "drawdown",
        "correlation",
        "normalize",
        "window",
        "rolling",
        "median",
        "quantile",
        "std",
        "variance",
        "涨跌",
        "变化",
        "变动",
        "涨幅",
        "跌幅",
        "百分比",
        "占比",
        "比例",
        "收益",
        "回报",
        "趋势",
        "波动",
        "回撤",
        "相关",
        "标准差",
        "方差",
        "中位数",
        "分位",
    }
    return any(term in searchable for term in analysis_terms)


def _missing_contract_outputs(request_state: RequestStateModel, gap: dict | None) -> list[str]:
    contract = request_state.task_contract
    if contract is None:
        return []
    required = [
        output
        for output in contract.required_outputs
        if output.required and not _contract_output_is_terminal_answer(output)
    ]
    if not required:
        return []
    state_missing = [
        output.id
        for output in required
        if not _contract_output_is_covered_by_state(request_state, output)
    ]
    if not state_missing:
        return []
    if gap is None:
        return state_missing
    if gap.get("can_answer") is not True:
        gap_missing = set(_gap_blocking_items(gap))
        if gap_missing:
            uncovered_gap_missing = [
                output.id
                for output in required
                if _contract_output_matches_gap_item(output, gap_missing)
                and not _contract_output_is_covered_by_state(request_state, output)
            ]
            return uncovered_gap_missing or state_missing
        return state_missing
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
        if _contract_output_is_covered_by_state(request_state, output):
            continue
        if keys & missing:
            missing_required.append(output.id)
            continue
        if not keys & covered:
            missing_required.append(output.id)
    return missing_required


def _contract_output_is_covered_by_state(request_state: RequestStateModel, output) -> bool:
    if _contract_output_is_covered_by_verified_fact(request_state, output):
        return True
    evidence_kind = str(getattr(output, "evidence_kind", "") or "").strip().lower()
    if _contract_output_is_terminal_answer(output):
        return request_state.final_answer_draft is not None
    if not evidence_kind:
        inferred = _contract_output_inferred_capabilities(output)
        if "forecast" in inferred and _latest_forecast_is_usable(request_state):
            return True
        if "anomaly" in inferred and request_state.latest_anomaly is not None:
            return True
        if "analysis" in inferred and request_state.latest_analysis_id is not None:
            return True
        if "query" in inferred and request_state.latest_database_evidence is not None and not _latest_database_evidence_is_empty(request_state):
            return True
        if not inferred:
            return bool(_all_answer_refs(request_state))
    if evidence_kind in {"query", "database", "database_evidence", "sql", "raw"}:
        return request_state.latest_database_evidence is not None and not _latest_database_evidence_is_empty(request_state)
    if evidence_kind in {"analysis", "derived", "statistical", "computed", "calculated"}:
        return request_state.latest_analysis_id is not None
    if evidence_kind == "forecast":
        return _latest_forecast_is_usable(request_state)
    if evidence_kind == "anomaly":
        return request_state.latest_anomaly is not None
    return False


def _contract_output_inferred_capabilities(output) -> set[str]:
    text = " ".join(
        str(value or "")
        for value in (
            getattr(output, "id", None),
            getattr(output, "description", None),
            getattr(output, "output_type", None),
            " ".join(str(item) for item in getattr(output, "measures", []) or []),
            " ".join(str(item) for item in getattr(output, "dimensions", []) or []),
            getattr(output, "success_criteria", None),
        )
    ).strip().lower()
    if not text:
        return set()
    inferred: set[str] = set()
    capability_terms = {
        "forecast": ("forecast", "prediction", "predict", "未来", "预测"),
        "anomaly": ("anomaly", "outlier", "spike", "异常", "离群"),
        "analysis": ("analysis", "metric", "statistics", "statistical", "计算", "指标", "统计", "分析"),
        "query": ("query", "evidence", "data", "rows", "points", "查询", "数据"),
    }
    for capability, terms in capability_terms.items():
        if any(term in text for term in terms):
            inferred.add(capability)
    return inferred


def _contract_output_matches_gap_item(output, gap_items: set[str]) -> bool:
    normalized_gap = {
        str(item or "").strip().lower()
        for item in gap_items
        if str(item or "").strip()
    }
    keys = {
        str(getattr(output, "id", "") or "").strip().lower(),
        str(getattr(output, "description", "") or "").strip().lower(),
        str(getattr(output, "evidence_kind", "") or "").strip().lower(),
        str(getattr(output, "output_type", "") or "").strip().lower(),
    }
    keys.update(str(item or "").strip().lower() for item in getattr(output, "measures", []) or [])
    keys.update(str(item or "").strip().lower() for item in getattr(output, "dimensions", []) or [])
    keys = {key for key in keys if key}
    return bool(keys & normalized_gap)


def _contract_output_is_terminal_answer(output) -> bool:
    evidence_kind = str(getattr(output, "evidence_kind", "") or "").strip().lower()
    output_type = str(getattr(output, "output_type", "") or "").strip().lower()
    return evidence_kind in {"conclusion", "answer", "final_answer"} or output_type in {
        "conclusion",
        "answer",
        "final_answer",
    }


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
        request_state=request_state,
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


def _gap_missing_is_covered_by_artifacts(request_state: RequestStateModel, missing: list[str]) -> bool:
    if not missing:
        return False
    missing_keys = {
        str(item).strip().lower()
        for item in missing
        if str(item).strip()
    }
    if not missing_keys:
        return False
    contract = request_state.task_contract
    outputs = getattr(contract, "required_outputs", []) if contract is not None else []
    matched_outputs = [
        output
        for output in outputs or []
        if _contract_output_matches_gap_item(output, missing_keys)
    ]
    if matched_outputs:
        return all(_contract_output_is_covered_by_state(request_state, output) for output in matched_outputs)
    structured_coverage = {
        "forecast": _latest_forecast_is_usable(request_state),
        "anomaly": request_state.latest_anomaly is not None,
        "analysis": request_state.latest_analysis_id is not None or _data_facts_cover_required_analysis(request_state),
        "database_evidence": request_state.latest_database_evidence is not None and not _latest_database_evidence_is_empty(request_state),
        "query": request_state.latest_database_evidence is not None and not _latest_database_evidence_is_empty(request_state),
    }
    known_missing_keys = set(structured_coverage)
    return all(key in known_missing_keys and structured_coverage[key] for key in missing_keys)


def _hard_gate_previous_assessment(
    *,
    request_state: RequestStateModel,
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
    refs = assessment.evidence_refs or evidence_refs_for_payload(payload)
    current_task_type = str(current.get("task_type") or "").strip().lower()
    if current_task_type in {"generic", "plan"}:
        current_task_type = "" if current_task_type == "generic" else "query"

    if tool_name in {"todowrite"}:
        return CompletionEvaluation(False, "Todo planning observations do not complete business todo steps.")
    if tool_name == "terminate" and current_task_type != "answer":
        return CompletionEvaluation(False, "Terminal observations cannot complete non-answer todo steps.")
    if current_task_type == "answer" and tool_name != "terminate":
        if _action_is_prerequisite_repair(tool_name, current_task_type, previous_observation):
            return CompletionEvaluation(
                False,
                f"Previous observation '{tool_name}' updated prerequisite evidence while final answer todo is active.",
                evidence_refs=refs,
                next_action_hint="Assemble the final answer after all evidence quality checks pass.",
            )
        return CompletionEvaluation(False, "Only a terminal final-answer observation can complete an answer todo.")
    if current_task_type and not _action_matches_task_type(tool_name, current_task_type):
        if _action_is_prerequisite_repair(tool_name, current_task_type, previous_observation):
            return CompletionEvaluation(
                False,
                f"Previous observation '{tool_name}' provided prerequisite evidence for active todo '{current_task_type}'.",
                evidence_refs=refs,
                next_action_hint=_hint_for_task_type(current_task_type),
            )
        return CompletionEvaluation(
            False,
            f"Previous observation '{tool_name}' does not match active todo type '{current_task_type}'.",
            next_action_hint=_hint_for_task_type(current_task_type),
        )

    if tool_name == "sql_query":
        if _query_requires_followup(payload):
            return CompletionEvaluation(
                False,
                "SQL observation is incomplete for the requested evidence contract.",
                missing_evidence=_query_runtime_missing(payload) or ["complete_query_result"],
                evidence_refs=refs,
                next_action_hint=_query_next_action_hint(payload) or "Issue a focused sql_query for the missing fields.",
            )
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
        if tool_name == "forecast" and not _forecast_payload_is_usable(payload):
            return CompletionEvaluation(
                False,
                "Forecast observation did not provide completed forecast points.",
                missing_evidence=["forecast"],
                evidence_refs=refs,
                next_action_hint="Run forecast to completion so the result status is succeeded and forecast_points is non-empty.",
            )
        if tool_name == "code_interpreter":
            analysis_quality_gaps = _analysis_quality_gaps(request_state)
            if analysis_quality_gaps:
                return CompletionEvaluation(
                    False,
                    "Derived analysis is inconsistent with detected anomalies on the same evidence.",
                    missing_evidence=analysis_quality_gaps,
                    evidence_refs=refs,
                    next_action_hint=(
                        "Rerun code_interpreter on the affected evidence and include transparent outlier treatment "
                        "details, or explicitly compute metrics on the anomaly-adjusted series."
                    ),
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
        analysis_quality_gaps = _analysis_quality_gaps(request_state)
        if analysis_quality_gaps:
            return CompletionEvaluation(
                False,
                "Final answer cannot rely on analysis that conflicts with detected anomalies.",
                missing_evidence=analysis_quality_gaps,
                evidence_refs=refs,
                next_action_hint=(
                    "Rerun code_interpreter with transparent outlier treatment before assembling the final answer."
                ),
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
    analysis_quality_gaps = _analysis_quality_gaps(request_state)
    if analysis_quality_gaps:
        evaluation = CompletionEvaluation(
            False,
            "Global progress assessment cannot complete todos while analysis conflicts with detected anomalies.",
            missing_evidence=analysis_quality_gaps,
            evidence_refs=assessment.evidence_refs or _all_answer_refs(request_state),
            next_action_hint=(
                "Rerun code_interpreter on the affected evidence and include transparent outlier treatment details."
            ),
        )
        request_state.completion_state["latest_step"] = _assessment_state(
            evaluation,
            tool_name=previous_observation.tool_name if previous_observation else None,
            todo_index=current_index,
            todo=current,
            accepted=False,
        )
        return evaluation

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

    missing_todo_outputs = []
    todo_refs: dict[int, str] = {}
    for index, todo in enumerate(request_state.todo_list):
        if todo.get("status") == "completed":
            continue
        task_type = str(todo.get("task_type") or "").strip().lower()
        if task_type == "answer":
            continue
        todo_ref = _artifact_ref_for_todo(request_state, task_type)
        if todo_ref is None:
            missing_todo_outputs.append(str(todo.get("content") or task_type or index + 1))
        else:
            todo_refs[index] = todo_ref
    if missing_todo_outputs:
        evaluation = CompletionEvaluation(
            False,
            "Global progress assessment cannot complete todos without matching tool artifacts.",
            missing_evidence=missing_todo_outputs,
            evidence_refs=refs,
            next_action_hint="Run the tool that matches the active todo before terminating.",
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
        completed_todo["result_ref"] = todo_refs.get(index) or completed_todo.get("result_ref")
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


def _artifact_ref_for_todo(request_state: RequestStateModel, task_type: str) -> str | None:
    if task_type == "query":
        evidence = request_state.latest_database_evidence
        return f"evidence:{evidence.evidence_id}" if evidence is not None else None
    if task_type == "code_interpreter":
        analysis_id = request_state.latest_analysis_id
        if analysis_id and _analysis_quality_gaps(request_state, analysis_ids={analysis_id}):
            return None
        return f"analysis:{analysis_id}" if analysis_id else None
    if task_type == "forecast":
        forecast = request_state.latest_forecast
        if forecast is not None and _latest_forecast_is_usable(request_state):
            return f"forecast:{forecast.forecast_id}"
        return None
    if task_type == "anomaly":
        anomaly = request_state.latest_anomaly
        return f"anomaly:{anomaly.anomaly_id}" if anomaly is not None else None
    if task_type == "rag" and request_state.latest_rag:
        return "rag:latest"
    if task_type == "skill" and request_state.latest_skill:
        return f"skill:{request_state.latest_skill.get('skill_name', 'latest')}"
    if not task_type:
        refs = _all_answer_refs(request_state)
        return refs[0] if refs else None
    return None


def _invalid_evidence_refs(request_state: RequestStateModel, refs: list[str]) -> list[str]:
    if not refs:
        return ["evidence_refs"]
    valid_refs = set(_all_answer_refs(request_state))
    if request_state.latest_database_evidence is not None:
        valid_refs.add(request_state.latest_database_evidence.evidence_id)
    valid_refs.update(request_state.analysis_artifacts.keys())
    valid_refs.update(f"analysis:{analysis_id}" for analysis_id in request_state.analysis_artifacts)
    if _latest_forecast_is_usable(request_state):
        valid_refs.add(request_state.latest_forecast.forecast_id)
    if request_state.latest_anomaly is not None:
        valid_refs.add(request_state.latest_anomaly.anomaly_id)
    valid_refs.update(f"fact:{fact.fact_id}" for fact in _verified_data_facts(request_state))
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
    return default_capability_registry().action_matches_task_type(tool_name, task_type)


def _action_is_prerequisite_repair(
    tool_name: str,
    task_type: str,
    previous_observation: ToolObservation | None,
) -> bool:
    if previous_observation is None or not previous_observation.success:
        return False
    return default_capability_registry().action_is_prerequisite(tool_name, task_type)


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


def _query_requires_followup(payload: dict) -> bool:
    coverage = _task_coverage(payload)
    return bool(coverage.get("runtime_requires_followup"))


def _query_runtime_missing(payload: dict) -> list[str]:
    coverage = _task_coverage(payload)
    return _string_items(coverage.get("runtime_missing")) or _string_items(coverage.get("missing"))


def _query_next_action_hint(payload: dict) -> str | None:
    coverage = _task_coverage(payload)
    hint = coverage.get("next_action_hint")
    return str(hint).strip() if str(hint or "").strip() else None


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
    return default_capability_registry().hint_for_task_type(task_type)


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
    refs.extend(f"fact:{fact.fact_id}" for fact in _verified_data_facts(request_state))
    if _latest_forecast_is_usable(request_state):
        refs.append(f"forecast:{request_state.latest_forecast.forecast_id}")
    if request_state.latest_anomaly is not None:
        refs.append(f"anomaly:{request_state.latest_anomaly.anomaly_id}")
    if request_state.latest_rag:
        refs.append("rag:latest")
    if request_state.latest_skill:
        refs.append(f"skill:{request_state.latest_skill.get('skill_name', 'latest')}")
    return refs


def _latest_forecast_is_usable(request_state: RequestStateModel) -> bool:
    forecast = request_state.latest_forecast
    if forecast is None:
        return False
    full_forecast = request_state.forecast_artifacts.get(forecast.forecast_id)
    return _forecast_payload_is_usable(
        (full_forecast or forecast).model_dump(mode="json")
    )


def _analysis_quality_gaps(request_state: RequestStateModel, analysis_ids: set[str] | None = None) -> list[str]:
    if not _contract_requires_outlier_treatment(request_state):
        return []
    gaps: list[str] = []
    for analysis_id, analysis in request_state.analysis_artifacts.items():
        if analysis_ids is not None and analysis_id not in analysis_ids:
            continue
        if _analysis_conflicts_with_detected_anomalies(request_state, analysis):
            gaps.append(f"analysis:{analysis_id}:requires_outlier_transparency")
    return gaps


def _contract_requires_outlier_treatment(request_state: RequestStateModel) -> bool:
    contract = request_state.task_contract
    outputs = getattr(contract, "required_outputs", []) if contract is not None else []
    for output in outputs or []:
        text = " ".join(
            str(value or "")
            for value in (
                getattr(output, "id", None),
                getattr(output, "description", None),
                getattr(output, "success_criteria", None),
                getattr(output, "evidence_kind", None),
                getattr(output, "output_type", None),
            )
        ).lower()
        if any(
            token in text
            for token in (
                "outlier_treatment",
                "adjusted_metrics",
                "excluded_rows",
                "异常值处理",
                "异常处理",
                "剔除",
            )
        ):
            return True
    return False


def _analysis_conflicts_with_detected_anomalies(request_state: RequestStateModel, analysis) -> bool:
    input_evidence_id = str(getattr(analysis, "input_evidence_id", "") or "").strip()
    if not input_evidence_id:
        return False
    if _analysis_has_transparent_outlier_treatment(analysis):
        return False
    for anomaly in request_state.anomaly_artifacts.values():
        if not _anomaly_has_points(anomaly):
            continue
        if _anomaly_matches_evidence(anomaly, input_evidence_id):
            return True
    return False


def _analysis_has_transparent_outlier_treatment(analysis) -> bool:
    result = getattr(analysis, "result", None)
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return False
    required = {
        "outlier_rule",
        "threshold_or_formula",
        "rationale",
        "excluded_rows",
        "raw_metrics",
        "adjusted_metrics",
    }
    return required <= set(details)


def _anomaly_has_points(anomaly) -> bool:
    points = getattr(anomaly, "anomaly_points", None)
    return isinstance(points, list) and len(points) > 0


def _anomaly_matches_evidence(anomaly, evidence_id: str) -> bool:
    diagnostics = getattr(anomaly, "diagnostics", None)
    if isinstance(diagnostics, dict):
        for key in ("resolved_evidence_id", "selected_evidence_id", "input_evidence_id"):
            if str(diagnostics.get(key) or "").strip() == evidence_id:
                return True
    anomaly_id = str(getattr(anomaly, "anomaly_id", "") or "")
    return anomaly_id == f"anomaly_{evidence_id}" or anomaly_id.endswith(evidence_id)


def _forecast_payload_is_usable(payload: dict) -> bool:
    if str(payload.get("status") or "").strip().lower() != "succeeded":
        return False
    forecast_points = payload.get("forecast_points")
    if not isinstance(forecast_points, list) or not forecast_points:
        return False
    horizon = payload.get("horizon")
    if isinstance(horizon, int) and horizon > 0:
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        full_count = diagnostics.get("forecast_point_count")
        if isinstance(full_count, int) and full_count > 0:
            return full_count >= horizon
        return len(forecast_points) >= horizon
    return True
