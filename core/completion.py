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
) -> CompletionEvaluation:
    """Advance visible todo progress after a successful tool observation."""
    current = current_todo(request_state)
    if current is None:
        return CompletionEvaluation(False, "No active todo step.", next_action_hint=None)

    refs = evidence_refs_for_payload(full_payload)
    return CompletionEvaluation(
        completed=True,
        reason="Active todo advanced after a successful tool observation.",
        evidence_refs=refs,
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

