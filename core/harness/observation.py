"""Observation frame construction for the ReAct harness."""
from __future__ import annotations

from dataclasses import dataclass, field

from schemas.state import RequestStateModel


@dataclass(frozen=True)
class ArtifactInventory:
    has_database_evidence: bool = False
    has_analysis: bool = False
    has_forecast: bool = False
    has_anomaly: bool = False
    has_rag: bool = False
    has_skill: bool = False
    verified_insight_count: int = 0
    visualization_count: int = 0
    analysis_count: int = 0

    def model_view(self) -> dict:
        return {
            "has_database_evidence": self.has_database_evidence,
            "has_analysis": self.has_analysis,
            "has_forecast": self.has_forecast,
            "has_anomaly": self.has_anomaly,
            "has_rag": self.has_rag,
            "has_skill": self.has_skill,
            "verified_insight_count": self.verified_insight_count,
            "visualization_count": self.visualization_count,
            "analysis_count": self.analysis_count,
        }


@dataclass(frozen=True)
class ObservationFrame:
    message: str
    requested_capabilities: tuple[str, ...]
    has_database_context: bool
    has_todo_plan: bool
    artifacts: ArtifactInventory
    active_todo_task_type: str | None = None
    latest_failed_tool: str | None = None
    latest_failure_payload: dict = field(default_factory=dict)
    latest_database_evidence_empty: bool = False
    downstream_analysis_request: dict | None = None
    shape_recovery_request: dict | None = None
    completion_missing_outputs: tuple[str, ...] = ()
    completion_reason: str | None = None
    requires_initial_todo_plan: bool = False

    def model_view(self) -> dict:
        return {
            "message": self.message,
            "requested_capabilities": list(self.requested_capabilities),
            "has_database_context": self.has_database_context,
            "has_todo_plan": self.has_todo_plan,
            "active_todo_task_type": self.active_todo_task_type,
            "artifacts": self.artifacts.model_view(),
            "latest_failed_tool": self.latest_failed_tool,
            "latest_failure_payload": self.latest_failure_payload,
            "latest_database_evidence_empty": self.latest_database_evidence_empty,
            "downstream_analysis_request": self.downstream_analysis_request,
            "shape_recovery_request": self.shape_recovery_request,
            "completion_missing_outputs": list(self.completion_missing_outputs),
            "completion_reason": self.completion_reason,
            "requires_initial_todo_plan": self.requires_initial_todo_plan,
        }


def build_observation_frame(
    request_state: RequestStateModel,
    *,
    requires_initial_todo_plan: bool = False,
    latest_database_evidence_empty: bool = False,
    downstream_analysis_request: dict | None = None,
    shape_recovery_request: dict | None = None,
    completion_missing_outputs: list[str] | None = None,
    completion_reason: str | None = None,
) -> ObservationFrame:
    latest_failure = _active_repair_failure(request_state)
    requested_capabilities = state_capabilities(request_state)
    return ObservationFrame(
        message=request_state.message,
        requested_capabilities=tuple(requested_capabilities),
        has_database_context=request_state.database_context is not None,
        has_todo_plan=bool(request_state.todo_list),
        active_todo_task_type=_active_todo_task_type(request_state),
        artifacts=ArtifactInventory(
            has_database_evidence=request_state.latest_database_evidence is not None,
            has_analysis=bool(request_state.analysis_artifacts),
            has_forecast=request_state.latest_forecast is not None,
            has_anomaly=request_state.latest_anomaly is not None,
            has_rag=request_state.latest_rag is not None,
            has_skill=request_state.latest_skill is not None,
            verified_insight_count=sum(
                1 for insight in request_state.insight_set.insights if insight.status == "verified"
            ),
            visualization_count=len(request_state.visualizations),
            analysis_count=len(request_state.analysis_artifacts),
        ),
        latest_failed_tool=latest_failure.tool_name if latest_failure else None,
        latest_failure_payload=latest_failure.payload if latest_failure and isinstance(latest_failure.payload, dict) else {},
        latest_database_evidence_empty=latest_database_evidence_empty,
        downstream_analysis_request=downstream_analysis_request,
        shape_recovery_request=shape_recovery_request,
        completion_missing_outputs=tuple(completion_missing_outputs or []),
        completion_reason=completion_reason,
        requires_initial_todo_plan=requires_initial_todo_plan,
    )


def _active_todo_task_type(request_state: RequestStateModel) -> str | None:
    current = next((todo for todo in request_state.todo_list if todo.get("status") == "in_progress"), None)
    if not isinstance(current, dict):
        return None
    task_type = str(current.get("task_type") or "").strip().lower()
    return task_type or None


def _active_repair_failure(request_state: RequestStateModel):
    """Return the failure that should drive the current repair gate.

    Repair is a state for the next transition after a failed observation, not a
    permanent property of the whole request history. Once any later observation
    succeeds, the repair action has either completed or produced new state for
    the normal completion gap logic to consume.
    """

    latest = request_state.observations[-1] if request_state.observations else None
    if latest is None or latest.success:
        return None
    return latest


def state_capabilities(request_state: RequestStateModel) -> list[str]:
    """Resolve runtime capabilities from LLM-authored intent and task contract."""
    values: list[str] = [
        str(item).strip().lower()
        for item in (request_state.requested_capabilities or [])
        if str(item).strip()
    ]
    contract = request_state.task_contract
    for output in getattr(contract, "required_outputs", []) if contract is not None else []:
        if not getattr(output, "required", False):
            continue
        evidence_kind = str(getattr(output, "evidence_kind", "") or "").strip().lower()
        if any(token in evidence_kind for token in ("anomaly", "outlier", "spike")):
            values.append("anomaly")
        elif any(token in evidence_kind for token in ("forecast", "predict")):
            values.append("forecast")
        elif any(token in evidence_kind for token in ("analysis", "derived", "statistical", "computed", "calculated", "custom")):
            values.append("analysis")
        elif any(token in evidence_kind for token in ("query", "database", "sql", "raw", "time_series", "timeseries")):
            values.append("query")
        elif evidence_kind in {"rag", "skill"}:
            values.append(evidence_kind)
        contract_text = " ".join(
            str(value or "")
            for value in (
                getattr(output, "id", None),
                getattr(output, "description", None),
                getattr(output, "success_criteria", None),
                getattr(output, "output_type", None),
            )
        ).lower()
        if any(token in contract_text for token in ("outlier", "anomaly", "spike", "excluded_rows", "异常", "离群", "剔除")):
            values.append("anomaly")
        if any(token in contract_text for token in ("visualization", "visual", "chart", "plot", "graph", "可视化", "图表", "曲线")):
            values.append("visualization")
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
