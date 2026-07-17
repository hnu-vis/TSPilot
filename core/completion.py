"""Runtime completion checks for ReAct task and goal progress."""
from __future__ import annotations

from dataclasses import dataclass, field
from schemas.state import RequestStateModel

KNOWN_EVIDENCE_NEEDS = {
    "database_evidence",
    "schema",
    "sample_rows",
    "count",
    "aggregate",
    "filtered_table",
    "time_series",
    "analysis_result",
    "anomaly_result",
    "forecast_result",
    "rag_result",
    "skill_result",
    "final_answer",
}

EVIDENCE_NEED_ALIASES = {
    "timeseries": "time_series",
    "time-series": "time_series",
    "raw_rows": "sample_rows",
    "rows": "sample_rows",
    "samples": "sample_rows",
    "statistics": "aggregate",
    "stats": "aggregate",
    "analysis": "analysis_result",
    "insight": "analysis_result",
    "anomaly": "anomaly_result",
    "forecast": "forecast_result",
    "answer": "final_answer",
}


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
    """Attach evidence needs derived from the todo content and criteria."""
    normalized = dict(todo)
    existing = normalized.get("evidence_needed")
    if isinstance(existing, str):
        needs = _normalize_evidence_needs([existing])
    elif isinstance(existing, list):
        needs = _normalize_evidence_needs([str(item) for item in existing if item not in (None, "")])
    else:
        needs = []
    inferred = infer_evidence_needs(normalized)
    normalized["evidence_needed"] = _dedupe([*needs, *inferred])
    normalized.setdefault("result_ref", None)
    normalized.setdefault("completion_reason", None)
    return normalized


def infer_evidence_needs(todo: dict) -> list[str]:
    text = " ".join(
        str(todo.get(key) or "")
        for key in ("content", "acceptance_criteria", "notes", "task_type")
    ).lower()
    task_type = str(todo.get("task_type") or "").lower()
    if task_type == "answer":
        return ["final_answer"]
    if task_type == "insight":
        return ["analysis_result"]
    if task_type == "anomaly":
        return ["anomaly_result"]
    if task_type == "forecast":
        return ["forecast_result"]
    if task_type in {"rag", "skill"}:
        return [f"{task_type}_result"]
    if any(token in text for token in ["schema", "字段", "表结构", "measurement", "metric list", "指标列表"]):
        return ["schema"]
    if any(token in text for token in ["sample", "样例", "示例", "预览"]):
        return ["sample_rows"]
    if any(token in text for token in ["总数", "总共", "多少条", "行数", "count", "row count"]):
        return ["count"]
    if any(token in text for token in ["最大", "最小", "平均", "均值", "sum", "avg", "average", "max", "min", "aggregate"]):
        return ["aggregate"]
    if any(token in text for token in ["时序", "趋势", "周期", "时间序列", "timeseries", "trend", "seasonality"]):
        return ["time_series"]
    if task_type == "query":
        return ["database_evidence"]
    return []


def _normalize_evidence_needs(values: list[str]) -> list[str]:
    normalized = []
    for value in values:
        key = value.strip().lower().replace(" ", "_")
        key = EVIDENCE_NEED_ALIASES.get(key, key)
        if key in KNOWN_EVIDENCE_NEEDS:
            normalized.append(key)
    return _dedupe(normalized)


def evaluate_step_completion(
    request_state: RequestStateModel,
    *,
    tool_name: str,
    full_payload: dict,
) -> CompletionEvaluation:
    current = current_todo(request_state)
    if current is None:
        return CompletionEvaluation(False, "No active todo step.", next_action_hint=None)

    task_type = str(current.get("task_type") or "").strip().lower()
    expected_tool = _task_type_for_tool(tool_name)
    if task_type and task_type != "generic" and expected_tool and task_type != expected_tool:
        return CompletionEvaluation(
            completed=False,
            reason=f"Tool '{tool_name}' does not satisfy current task type '{task_type}'.",
            missing_evidence=current.get("evidence_needed") or [],
            next_action_hint=_hint_for_needs(current.get("evidence_needed") or []),
        )

    needs = list(current.get("evidence_needed") or [])
    if not needs:
        needs = infer_evidence_needs(current)
    if not needs:
        needs = [_need_for_tool(tool_name)]

    satisfied = satisfied_needs(request_state, full_payload=full_payload)
    missing = [need for need in needs if not _need_satisfied(need, satisfied)]
    refs = evidence_refs_for_payload(full_payload)
    if missing:
        return CompletionEvaluation(
            completed=False,
            reason="The action succeeded, but the active todo still lacks required evidence.",
            missing_evidence=missing,
            evidence_refs=refs,
            next_action_hint=_hint_for_needs(missing),
        )
    return CompletionEvaluation(
        completed=True,
        reason="Active todo acceptance criteria are satisfied by the latest artifact.",
        evidence_refs=refs,
    )


def evaluate_goal_completion(request_state: RequestStateModel) -> GoalCompletionEvaluation:
    if request_state.database_context is None:
        return GoalCompletionEvaluation(True, "No database context is required for this response.")

    incomplete = [
        todo
        for todo in request_state.todo_list
        if todo.get("status") != "completed"
        and str(todo.get("task_type") or "").lower() != "answer"
    ]
    if incomplete:
        current = next((todo for todo in incomplete if todo.get("status") == "in_progress"), incomplete[0])
        missing = list(current.get("evidence_needed") or infer_evidence_needs(current) or ["database_evidence"])
        return GoalCompletionEvaluation(
            can_answer=False,
            reason="The active plan still has unfinished evidence or analysis steps.",
            missing_evidence=missing,
            answerable_from=_all_answer_refs(request_state),
            next_action_hint=_hint_for_needs(missing),
        )

    refs = _all_answer_refs(request_state)
    if refs:
        return GoalCompletionEvaluation(True, "Available artifacts are sufficient to assemble the final answer.", answerable_from=refs)

    return GoalCompletionEvaluation(
        can_answer=False,
        reason="No database-backed evidence or derived analysis is available yet.",
        missing_evidence=["database_evidence"],
        next_action_hint="Call sql_query to obtain grounded database evidence first.",
    )


def current_todo(request_state: RequestStateModel) -> dict | None:
    return next((todo for todo in request_state.todo_list if todo.get("status") == "in_progress"), None)


def satisfied_needs(request_state: RequestStateModel, *, full_payload: dict | None = None) -> set[str]:
    needs: set[str] = set()
    evidence = request_state.latest_database_evidence
    if evidence is not None:
        needs.add("database_evidence")
        result_type = evidence.result_type
        data = evidence.data if isinstance(evidence.data, dict) else {}
        if result_type in {"schema", "metric_list"}:
            needs.add("schema")
        if result_type == "statistics":
            needs.update({"count", "aggregate"})
        if result_type == "table":
            needs.update({"filtered_table", "aggregate"})
            if _table_payload_has_count({"columns": evidence.columns, "data": data}):
                needs.add("count")
            if _table_payload_has_timeseries({"columns": evidence.columns, "data": data}):
                needs.add("time_series")
        if result_type == "timeseries":
            needs.add("time_series")
        if isinstance(data.get("rows"), list) and data.get("rows"):
            needs.add("sample_rows")
        if isinstance(data.get("points"), list) and data.get("points"):
            needs.add("time_series")
    if request_state.latest_analysis_id or request_state.latest_insight:
        needs.add("analysis_result")
    if request_state.latest_anomaly is not None:
        needs.add("anomaly_result")
    if request_state.latest_forecast is not None:
        needs.add("forecast_result")
    if request_state.latest_rag:
        needs.add("rag_result")
    if request_state.latest_skill:
        needs.add("skill_result")
    if request_state.final_answer_draft is not None:
        needs.add("final_answer")

    payload_needs = _needs_from_payload(full_payload or {})
    needs.update(payload_needs)
    return needs


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


def _needs_from_payload(payload: dict) -> set[str]:
    needs: set[str] = set()
    result_type = payload.get("result_type")
    if result_type:
        needs.add("database_evidence")
    if result_type in {"schema", "metric_list"}:
        needs.add("schema")
    if result_type == "statistics":
        needs.update({"count", "aggregate"})
    if result_type == "table":
        needs.update({"filtered_table", "aggregate"})
        if _table_payload_has_count(payload):
            needs.add("count")
        if _table_payload_has_timeseries(payload):
            needs.add("time_series")
    if result_type == "timeseries":
        needs.add("time_series")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    if isinstance(data.get("rows"), list) and data.get("rows"):
        needs.add("sample_rows")
    if isinstance(data.get("points"), list) and data.get("points"):
        needs.add("time_series")
    if payload.get("analysis_id") or payload.get("insight_id"):
        needs.add("analysis_result")
    if payload.get("forecast_id"):
        needs.add("forecast_result")
    if payload.get("anomaly_id"):
        needs.add("anomaly_result")
    if payload.get("results") is not None:
        needs.add("rag_result")
    if payload.get("skill_name"):
        needs.add("skill_result")
    if payload.get("sections") is not None or payload.get("summary") and payload.get("references") is not None:
        needs.add("final_answer")
    return needs


def _need_satisfied(need: str, satisfied: set[str]) -> bool:
    if need in satisfied:
        return True
    if need == "aggregate" and {"count", "filtered_table"} & satisfied:
        return True
    if need == "sample_rows" and {"filtered_table", "time_series"} & satisfied:
        return True
    if need == "database_evidence" and {"schema", "count", "aggregate", "filtered_table", "time_series"} & satisfied:
        return True
    return False


def _table_payload_has_count(payload: dict) -> bool:
    columns = payload.get("columns") if isinstance(payload.get("columns"), list) else []
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    count_names = {"count", "count(*)", "row_count", "rows_count", "total", "total_count"}
    if any(str(column).strip().lower() in count_names for column in columns):
        return True
    if rows:
        first = rows[0]
        if isinstance(first, dict) and any(str(key).strip().lower() in count_names for key in first):
            return True
    return False


def _table_payload_has_timeseries(payload: dict) -> bool:
    columns = [str(column).strip().lower() for column in payload.get("columns", []) if column is not None]
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    if not rows:
        return False
    time_names = {"time", "timestamp", "_time", "datetime", "date"}
    value_names = {"value", "_value", "metric_value"}
    has_time = any(column in time_names or column.endswith("_time") or column.endswith("_timestamp") for column in columns)
    has_value = any(column in value_names for column in columns)
    if not has_value:
        first = rows[0]
        if isinstance(first, dict):
            has_value = any(
                isinstance(value, (int, float))
                for key, value in first.items()
                if str(key).strip().lower() not in time_names
            )
    return has_time and has_value


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


def _task_type_for_tool(tool_name: str) -> str | None:
    return {
        "todowrite": "plan",
        "sql_query": "query",
        "insight": "insight",
        "anomaly": "anomaly",
        "forecast": "forecast",
        "format_answer": "answer",
        "rag": "rag",
        "skill": "skill",
    }.get(tool_name)


def _need_for_tool(tool_name: str) -> str:
    return {
        "sql_query": "database_evidence",
        "insight": "analysis_result",
        "anomaly": "anomaly_result",
        "forecast": "forecast_result",
        "format_answer": "final_answer",
        "rag": "rag_result",
        "skill": "skill_result",
    }.get(tool_name, "database_evidence")


def _hint_for_needs(needs: list[str]) -> str:
    if any(need in needs for need in ["schema", "sample_rows", "count", "aggregate", "filtered_table", "time_series", "database_evidence"]):
        return "Call sql_query with a focused message or read-only query that fills the missing evidence."
    if "analysis_result" in needs:
        return "Call insight over the relevant database evidence."
    if "anomaly_result" in needs:
        return "Call anomaly over time-series evidence."
    if "forecast_result" in needs:
        return "Call forecast over time-series evidence."
    return "Choose the next action that directly fills the missing evidence."


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
