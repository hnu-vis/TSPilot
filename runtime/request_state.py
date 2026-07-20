"""Request-state helpers."""
from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from app.settings import Settings
from core.completion import (
    evaluate_goal_completion,
    evaluate_step_completion,
    normalize_todo_for_completion,
)
from core.time_range import normalize_time_range
from runtime.artifacts import persist_json_artifact
from runtime.trace import TraceEventModel
from schemas.api import ChatRequest, ChatResponse
from schemas.database_context import DatabaseContext
from schemas.output import FinalAnswer
from schemas.state import ConversationStateModel, RequestStateModel
from schemas.tool import ToolObservation

if TYPE_CHECKING:
    from tools.registry import ToolSpec


def normalize_chat_request(request: ChatRequest) -> ChatRequest:
    """Normalize legacy database aliases into database_context."""
    if request.database_context is None and request.selected_database:
        request.database_context = DatabaseContext(
            database_id=request.selected_database,
            database_type=request.selected_database_type or "unknown",
            display_name=request.selected_database,
        )
    return request


def build_request_state(request: ChatRequest, settings: Settings) -> RequestStateModel:
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    requested_fact_types: list[str] = []
    answer_requirements = ["conclusion"]
    focus = request.message
    database_context = _with_schema_hint(request.database_context, settings)
    return RequestStateModel(
        request_id=request_id,
        conversation_id=conversation_id,
        message=request.message,
        database_context=database_context,
        selected_database=request.selected_database,
        selected_database_type=request.selected_database_type,
        time_range=normalize_time_range(request.time_range),
        constraints=request.constraints,
        history=request.history,
        status="running",
        current_intent="chat_analysis",
        intent_profile={},
        requested_fact_types=requested_fact_types,
        answer_requirements=answer_requirements,
        answer_coverage={requirement: False for requirement in answer_requirements},
        focus=focus,
        todo_list=[],
        plan_current_step=0,
        planning_complete=False,
        iteration=0,
        max_iterations=settings.max_iterations,
        context_budget={
            "max_prompt_tokens": settings.max_prompt_tokens,
            "max_history_messages": settings.max_history_messages,
            "max_tool_history_items": settings.max_tool_history_items,
            "max_observation_chars": settings.max_observation_chars,
            "max_visible_rows": settings.max_visible_rows,
            "max_visible_points": settings.max_visible_points,
            "overflow_policy": "fail_closed",
        },
        context_status="ok",
        context_overflow_reason=None,
        completion_state={},
        latest_database_evidence=None,
        database_evidence_artifacts={},
        latest_insight=None,
        insight_artifacts={},
        latest_analysis_id=None,
        analysis_artifacts={},
        latest_forecast=None,
        forecast_artifacts={},
        latest_anomaly=None,
        anomaly_artifacts={},
        latest_rag=None,
        latest_skill=None,
        verified_facts=[],
        rejected_facts=[],
        final_answer_draft=None,
        visualizations=[],
        tool_history=[],
        observations=[],
        errors=[],
        prompt_context_summary=None,
    )


def build_conversation_state(request: ChatRequest, conversation_id: str) -> ConversationStateModel:
    return ConversationStateModel(
        conversation_id=conversation_id,
        database_context=request.database_context,
        recent_messages=request.history,
        session_summary=None,
        intent_profile={},
        todo_list=[],
        plan_current_step=0,
        planning_complete=False,
        recent_todo_summary=None,
        latest_database_evidence=None,
        database_evidence_artifacts={},
        latest_insight=None,
        insight_artifacts={},
        latest_analysis_id=None,
        analysis_artifacts={},
        latest_forecast=None,
        forecast_artifacts={},
        latest_anomaly=None,
        anomaly_artifacts={},
        latest_rag=None,
        latest_skill=None,
        recent_visualizations=[],
        updated_at=None,
        context_budget=None,
    )


def _with_schema_hint(database_context: DatabaseContext | None, settings: Settings) -> DatabaseContext | None:
    """Attach bounded datasource structure known from local config before the first tool call."""
    if database_context is None or database_context.schema_hint:
        return database_context
    config = _load_cached_database_config(database_context.database_id, settings)
    if not config:
        return database_context
    schema_hint = _build_schema_hint(config, settings)
    if not schema_hint:
        return database_context
    return database_context.model_copy(update={"schema_hint": schema_hint})


def _load_cached_database_config(database_id: str, settings: Settings) -> dict | None:
    cache_path = Path(settings.tspilot_root) / "cache_data" / "database" / "databases.json"
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    databases = payload.get("databases") if isinstance(payload, dict) else None
    config = databases.get(database_id) if isinstance(databases, dict) else None
    return config if isinstance(config, dict) else None


def _build_schema_hint(config: dict, settings: Settings) -> dict:
    database_type = str(config.get("type") or config.get("db_type") or "unknown")
    hint = {
        "source": "local_database_config",
        "database_id": config.get("id") or config.get("name"),
        "database_type": database_type,
        "query_language": _query_language_for_database_type(database_type),
        "tables_or_measurements": [],
    }
    reference_dataset = config.get("reference_dataset")
    if isinstance(reference_dataset, dict):
        table_name = (
            reference_dataset.get("measurement")
            or reference_dataset.get("metric_name")
            or reference_dataset.get("table")
            or reference_dataset.get("series_name")
        )
        dataset_path = _resolve_dataset_path(reference_dataset.get("dataset_path"), settings)
        field_columns = reference_dataset.get("field_columns")
        if not isinstance(field_columns, list):
            value_column = reference_dataset.get("value_column")
            field_columns = [value_column] if value_column else []
        time_column = reference_dataset.get("timestamp_column")
        sample_columns = [
            str(column)
            for column in [time_column, *field_columns[:8]]
            if column not in (None, "")
        ]
        sample_rows = _project_rows(_read_sample_rows(dataset_path, limit=3), sample_columns)
        table_hint = {
            "name": table_name,
            "row_count": _count_csv_rows(dataset_path),
            "time_column": time_column,
            "field_columns": [str(column) for column in field_columns if column not in (None, "")][:60],
            "sample_rows": sample_rows,
        }
        hint["tables_or_measurements"].append({k: v for k, v in table_hint.items() if v not in (None, [], "")})
        return hint

    configured_names = config.get("schema_measurement_names") or config.get("schema_metric_names")
    if isinstance(configured_names, str):
        configured_names = [configured_names]
    if isinstance(configured_names, list):
        hint["tables_or_measurements"] = [
            {"name": str(name)}
            for name in configured_names[:20]
            if name not in (None, "")
        ]
    return hint if hint["tables_or_measurements"] else {}


def _query_language_for_database_type(database_type: str) -> str:
    normalized = database_type.lower()
    if normalized == "influxdb":
        return "flux"
    if normalized == "prometheus":
        return "promql"
    return "sql"


def _resolve_dataset_path(raw_path: object, settings: Settings) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (Path(settings.tspilot_root) / path).resolve()
    return path


def _count_csv_rows(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except Exception:
        return None


def _read_sample_rows(path: Path | None, *, limit: int) -> list[dict]:
    if path is None or not path.exists() or limit <= 0:
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(dict(row))
                if len(rows) >= limit:
                    break
    except Exception:
        return []
    return rows


def _project_rows(rows: list[dict], columns: list[str]) -> list[dict]:
    if not columns:
        return rows
    return [
        {column: row[column] for column in columns if column in row}
        for row in rows
    ]


def append_trace(request_state: RequestStateModel, event_type: str, payload: dict) -> TraceEventModel:
    event = TraceEventModel(event_type=event_type, payload=payload)
    return event


def build_final_response(request_state: RequestStateModel, trace_events: list[TraceEventModel]) -> ChatResponse:
    if request_state.final_answer_draft is not None:
        return ChatResponse(
            conversation_id=request_state.conversation_id or "",
            request_id=request_state.request_id,
            status="completed",
            response_kind="final_answer",
            used_tools=[call.tool_name for call in request_state.tool_history],
            answer=request_state.final_answer_draft,
            trace=trace_events,
            error=None,
        )

    return ChatResponse(
        conversation_id=request_state.conversation_id or "",
        request_id=request_state.request_id,
        status="failed",
        response_kind="error",
        used_tools=[call.tool_name for call in request_state.tool_history],
        answer=None,
        trace=trace_events,
        error="Request did not reach a final answer.",
    )


def apply_observation(
    request_state: RequestStateModel,
    observation: ToolObservation,
    full_payload: dict,
    tool_spec: "ToolSpec",
) -> None:
    request_state.observations.append(observation)
    if not observation.success:
        return

    if tool_spec.result_target == "todo":
        _apply_todo_payload(request_state, full_payload)
    elif tool_spec.result_target == "evidence":
        _apply_evidence_payload(request_state, full_payload)
        _advance_plan_after_success(request_state, observation.tool_name, full_payload)
    elif tool_spec.result_target == "analysis":
        _apply_analysis_payload(request_state, full_payload)
        _advance_plan_after_success(request_state, observation.tool_name, full_payload)


async def apply_observation_async(
    request_state: RequestStateModel,
    observation: ToolObservation,
    full_payload: dict,
    tool_spec: "ToolSpec",
    completion_evaluator=None,
) -> None:
    request_state.observations.append(observation)
    if not observation.success:
        return

    if tool_spec.result_target == "todo":
        _apply_todo_payload(request_state, full_payload)
    elif tool_spec.result_target == "evidence":
        _apply_evidence_payload(request_state, full_payload)
        await _advance_plan_after_success_async(request_state, observation.tool_name, full_payload, completion_evaluator)
    elif tool_spec.result_target == "analysis":
        _apply_analysis_payload(request_state, full_payload)
        await _advance_plan_after_success_async(request_state, observation.tool_name, full_payload, completion_evaluator)
    elif tool_spec.result_target == "presentation":
        _apply_presentation_payload(request_state, full_payload)
        await _advance_plan_after_success_async(request_state, observation.tool_name, full_payload, completion_evaluator)
    elif tool_spec.result_target == "presentation":
        _apply_presentation_payload(request_state, full_payload)
        _advance_plan_after_success(request_state, observation.tool_name, full_payload)


def enrich_observation_payload(
    request_state: RequestStateModel,
    observation: ToolObservation,
    full_payload: dict,
    tool_spec: "ToolSpec",
) -> ToolObservation:
    """Replace observation payload with the latest prompt-safe state view."""

    payload = observation.payload
    if tool_spec.result_target == "evidence" and request_state.latest_database_evidence is not None:
        if request_state.latest_database_evidence.evidence_id == full_payload.get("evidence_id"):
            payload = request_state.latest_database_evidence.model_dump(mode="json")
    elif tool_spec.result_target == "analysis":
        if "analysis_id" in full_payload:
            analysis_id = str(full_payload.get("analysis_id"))
            analysis = request_state.analysis_artifacts.get(analysis_id)
            if analysis is not None:
                payload = _build_prompt_safe_analysis(analysis)
        elif "insight_id" in full_payload and request_state.latest_insight is not None:
            payload = request_state.latest_insight.model_dump(mode="json")
        elif "forecast_id" in full_payload and request_state.latest_forecast is not None:
            payload = request_state.latest_forecast.model_dump(mode="json")
        elif "anomaly_id" in full_payload and request_state.latest_anomaly is not None:
            payload = request_state.latest_anomaly.model_dump(mode="json")
    elif tool_spec.result_target == "presentation" and request_state.final_answer_draft is not None:
        payload = request_state.final_answer_draft.model_dump(mode="json")

    return observation.model_copy(
        update={
            "payload": payload,
            "payload_truncated": observation.payload_truncated or payload != full_payload,
        }
    )


def _apply_todo_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    request_state.todo_list = [
        normalize_todo_for_completion(todo)
        for todo in list(full_payload.get("todos", []))
        if isinstance(todo, dict)
    ]
    request_state.plan_current_step = int(full_payload.get("current_step") or 0)
    request_state.planning_complete = bool(full_payload.get("planning_complete", False))
    if request_state.todo_list:
        request_state.answer_coverage["plan"] = True
        request_state.max_iterations = max(
            request_state.max_iterations,
            min(20, len(request_state.todo_list) * 3 + 2),
        )
    request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()


def _advance_plan_after_success(request_state: RequestStateModel, tool_name: str, full_payload: dict) -> None:
    if not request_state.todo_list:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    task_type = _task_type_for_tool(tool_name)
    if task_type is None:
        return
    current_index = next((index for index, todo in enumerate(request_state.todo_list) if todo.get("status") == "in_progress"), None)
    if current_index is None:
        return
    current_todo = dict(request_state.todo_list[current_index])
    current_task_type = str(current_todo.get("task_type") or "").strip().lower()
    if current_task_type and current_task_type != "generic" and current_task_type != task_type:
        return
    evaluation = evaluate_step_completion(request_state, tool_name=tool_name, full_payload=full_payload)
    request_state.completion_state["latest_step"] = {
        **evaluation.model_dump(),
        "tool_name": tool_name,
        "todo_index": current_index,
        "todo": current_todo,
    }
    if not evaluation.completed:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current_todo["status"] = "completed"
    current_todo["result_ref"] = evaluation.evidence_refs[0] if evaluation.evidence_refs else current_todo.get("result_ref")
    current_todo["completion_reason"] = evaluation.reason
    request_state.todo_list[current_index] = current_todo
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


async def _advance_plan_after_success_async(
    request_state: RequestStateModel,
    tool_name: str,
    full_payload: dict,
    completion_evaluator=None,
) -> None:
    if not request_state.todo_list:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    task_type = _task_type_for_tool(tool_name)
    if task_type is None:
        return
    current_index = next((index for index, todo in enumerate(request_state.todo_list) if todo.get("status") == "in_progress"), None)
    if current_index is None:
        return
    current_todo = dict(request_state.todo_list[current_index])
    current_task_type = str(current_todo.get("task_type") or "").strip().lower()
    if current_task_type and current_task_type != "generic" and current_task_type != task_type:
        return

    if completion_evaluator is not None:
        evaluation = await completion_evaluator.evaluate_step_completion(
            request_state=request_state,
            tool_name=tool_name,
            full_payload=full_payload,
        )
    else:
        evaluation = evaluate_step_completion(request_state, tool_name=tool_name, full_payload=full_payload)

    verdict = getattr(completion_evaluator, "last_step_verdict", None) if completion_evaluator is not None else None
    request_state.completion_state["latest_step"] = {
        **evaluation.model_dump(),
        "tool_name": tool_name,
        "todo_index": current_index,
        "todo": current_todo,
        "completion_verdict": verdict,
    }
    if not evaluation.completed:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current_todo["status"] = "completed"
    current_todo["result_ref"] = evaluation.evidence_refs[0] if evaluation.evidence_refs else current_todo.get("result_ref")
    current_todo["completion_reason"] = evaluation.reason
    current_todo["completion_verdict"] = verdict
    request_state.todo_list[current_index] = current_todo
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


def _task_type_for_tool(tool_name: str) -> str | None:
    mapping = {
        "todowrite": "plan",
        "sql_query": "query",
        "insight": "insight",
        "anomaly": "anomaly",
        "forecast": "forecast",
        "format_answer": "answer",
        "rag": "rag",
        "skill": "skill",
    }
    return mapping.get(tool_name)


def _apply_evidence_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    from schemas.database import DatabaseEvidence

    full_evidence = DatabaseEvidence.model_validate(full_payload)
    request_state.database_evidence_artifacts[full_evidence.evidence_id] = full_evidence
    request_state.latest_database_evidence = _build_prompt_safe_evidence(full_evidence)


def _build_prompt_safe_evidence(evidence):
    from schemas.database import DatabaseEvidence

    data = dict(evidence.data)
    summary_stats = {
        "points_count": len(data.get("points", [])) if isinstance(data.get("points"), list) else None,
        "rows_count": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
        "series_count": len(data.get("series", [])) if isinstance(data.get("series"), list) else None,
    }
    if isinstance(data.get("points"), list):
        data["points"] = _sample_edges(data["points"], limit=24)
    if isinstance(data.get("rows"), list):
        data["rows"] = _sample_edges(data["rows"], limit=12)
    if isinstance(data.get("series"), list):
        summarized_series = []
        for series in data["series"][:6]:
            item = dict(series)
            if isinstance(item.get("points"), list):
                item["points"] = _sample_edges(item["points"], limit=12)
                item["points_count"] = len(series.get("points", []))
            summarized_series.append(item)
        data["series"] = summarized_series
    diagnostics = dict(evidence.diagnostics)
    diagnostics["artifact_kind"] = "database_evidence"
    diagnostics["artifact_ref"] = f"evidence:{evidence.evidence_id}"
    diagnostics["summary_stats"] = {key: value for key, value in summary_stats.items() if value is not None}
    if "query_trace" in diagnostics and isinstance(diagnostics["query_trace"], dict):
        query_trace = dict(diagnostics["query_trace"])
        raw_result_summary = dict(query_trace.get("raw_result_summary") or {})
        raw_result_summary.pop("columns", None)
        query_trace["raw_result_summary"] = raw_result_summary
        diagnostics["query_trace"] = query_trace
    return DatabaseEvidence(
        evidence_id=evidence.evidence_id,
        result_type=evidence.result_type,
        database=evidence.database,
        query_language=evidence.query_language,
        query=evidence.query,
        summary=evidence.summary,
        data=data,
        columns=evidence.columns,
        metadata=dict(evidence.metadata),
        diagnostics=diagnostics,
    )


def _sample_edges(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return items
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return [*items[:head], *items[-tail:]]


def _summarize_visualization_dict(payload: dict) -> dict:
    item = dict(payload)
    chart = item.get("chart")
    if isinstance(chart, dict):
        x_axis_data = list(chart.get("x_axis_data") or [])
        series_data = list(chart.get("series_data") or [])
        item["chart"] = {
            "x_axis_count": len(x_axis_data),
            "x_axis_preview": [*x_axis_data[:3], *x_axis_data[-3:]] if len(x_axis_data) > 6 else x_axis_data,
            "series_data": [
                {
                    "name": series.get("name"),
                    "points_count": len(series.get("data") or []),
                }
                for series in series_data[:4]
                if isinstance(series, dict)
            ],
        }
    if isinstance(item.get("annotations"), list):
        item["annotations"] = item["annotations"][:12]
    if isinstance(item.get("rows"), list):
        item["rows"] = _sample_edges(item["rows"], limit=12)
    if isinstance(item.get("display_rows"), list):
        item["display_rows"] = _sample_edges(item["display_rows"], limit=12)
    return item


def _build_prompt_safe_insight(insight):
    from schemas.insight import InsightResult

    payload = insight.model_dump(mode="json")
    payload["fact_candidates"] = payload.get("fact_candidates", [])[:8]
    payload["completed_facts"] = payload.get("completed_facts", [])[:6]
    payload["verified_facts"] = payload.get("verified_facts", [])[:6]
    payload["rejected_facts"] = payload.get("rejected_facts", [])[:6]
    payload["summary_blocks"] = payload.get("summary_blocks", [])[:6]
    payload["visualizations"] = [
        _summarize_visualization_dict(item)
        for item in payload.get("visualizations", [])[:4]
    ]
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics["artifact_kind"] = "insight_result"
    diagnostics["artifact_ref"] = f"insight:{insight.insight_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=insight.insight_id,
        artifact_kind="insight_result",
        payload=insight.model_dump(mode="json"),
        subdir="analysis_snapshots",
    )
    payload["diagnostics"] = diagnostics
    return InsightResult.model_validate(payload)


def _build_prompt_safe_analysis(analysis):
    from schemas.analysis import AnalysisResult

    payload = analysis.model_dump(mode="json")
    result = dict(payload.get("result") or {})
    if isinstance(result.get("details"), list):
        result["details"] = _sample_edges([item for item in result["details"] if isinstance(item, dict)], limit=12)
    if isinstance(result.get("rows"), list):
        result["rows"] = _sample_edges([item for item in result["rows"] if isinstance(item, dict)], limit=12)
    payload["result"] = result
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics["artifact_kind"] = "analysis_result"
    diagnostics["artifact_ref"] = f"analysis:{analysis.analysis_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=analysis.analysis_id,
        artifact_kind="analysis_result",
        payload=analysis.model_dump(mode="json"),
        subdir="analysis_snapshots",
    )
    payload["diagnostics"] = diagnostics
    return AnalysisResult.model_validate(payload).model_dump(mode="json")


def _build_prompt_safe_forecast(forecast):
    from schemas.timeseries import ForecastResult

    payload = forecast.model_dump(mode="json")
    payload["forecast_points"] = payload.get("forecast_points", [])[:12]
    payload["confidence_interval"] = payload.get("confidence_interval", [])[:12]
    payload["visualizations"] = [
        _summarize_visualization_dict(item)
        for item in payload.get("visualizations", [])[:3]
    ]
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics["artifact_kind"] = "forecast_result"
    diagnostics["artifact_ref"] = f"forecast:{forecast.forecast_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=forecast.forecast_id,
        artifact_kind="forecast_result",
        payload=forecast.model_dump(mode="json"),
        subdir="analysis_snapshots",
    )
    payload["diagnostics"] = diagnostics
    return ForecastResult.model_validate(payload)


def _build_prompt_safe_anomaly(anomaly):
    from schemas.timeseries import AnomalyResult

    payload = anomaly.model_dump(mode="json")
    payload["anomaly_points"] = payload.get("anomaly_points", [])[:12]
    payload["anomaly_spans"] = payload.get("anomaly_spans", [])[:12]
    payload["scores"] = payload.get("scores", [])[:12]
    payload["visualizations"] = [
        _summarize_visualization_dict(item)
        for item in payload.get("visualizations", [])[:3]
    ]
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics["artifact_kind"] = "anomaly_result"
    diagnostics["artifact_ref"] = f"anomaly:{anomaly.anomaly_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=anomaly.anomaly_id,
        artifact_kind="anomaly_result",
        payload=anomaly.model_dump(mode="json"),
        subdir="analysis_snapshots",
    )
    payload["diagnostics"] = diagnostics
    return AnomalyResult.model_validate(payload)


def _apply_analysis_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    if "analysis_id" in full_payload:
        from schemas.analysis import AnalysisResult

        analysis = AnalysisResult.model_validate(full_payload)
        request_state.analysis_artifacts[analysis.analysis_id] = analysis
        request_state.latest_analysis_id = analysis.analysis_id
        request_state.answer_coverage["analysis"] = True
        for requirement in list(request_state.answer_coverage):
            if requirement not in {"plan", "forecast", "anomaly"}:
                request_state.answer_coverage[requirement] = True
        return

    if "insight_id" in full_payload:
        from schemas.insight import InsightResult

        insight = InsightResult.model_validate(full_payload)
        request_state.insight_artifacts[insight.insight_id] = insight
        request_state.latest_insight = _build_prompt_safe_insight(insight)
        request_state.verified_facts = insight.verified_facts
        request_state.rejected_facts = insight.rejected_facts
        request_state.visualizations = list(request_state.latest_insight.visualizations)
        for fact in insight.verified_facts:
            fact_type = str(fact.fact_type)
            request_state.answer_coverage[fact_type] = True
            if fact_type == "outlier":
                request_state.answer_coverage["anomaly"] = True
        return

    if "forecast_id" in full_payload:
        from schemas.timeseries import ForecastResult

        forecast = ForecastResult.model_validate(full_payload)
        request_state.forecast_artifacts[forecast.forecast_id] = forecast
        request_state.latest_forecast = _build_prompt_safe_forecast(forecast)
        request_state.visualizations.extend(request_state.latest_forecast.visualizations)
        request_state.answer_coverage["forecast"] = True
        return

    if "anomaly_id" in full_payload:
        from schemas.timeseries import AnomalyResult

        anomaly = AnomalyResult.model_validate(full_payload)
        request_state.anomaly_artifacts[anomaly.anomaly_id] = anomaly
        request_state.latest_anomaly = _build_prompt_safe_anomaly(anomaly)
        request_state.visualizations.extend(request_state.latest_anomaly.visualizations)
        request_state.answer_coverage["anomaly"] = True
        return

    if "skill_name" in full_payload:
        request_state.latest_skill = full_payload
        return

    if "results" in full_payload:
        request_state.latest_rag = full_payload


def _apply_presentation_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    request_state.final_answer_draft = FinalAnswer.model_validate(full_payload)
    request_state.answer_coverage["conclusion"] = True
