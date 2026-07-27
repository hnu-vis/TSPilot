"""Request-state helpers."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from app.settings import Settings
from core.completion import (
    evaluate_goal_completion,
    normalize_todo_for_completion,
)
from core.intent import build_intent_profile_fallback
from core.time_range import normalize_time_range
from runtime.artifacts import persist_json_artifact
from runtime.language import detect_response_language
from runtime.token_usage import token_usage_summary
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
    conversation_run_dir = None
    request_log_dir = None
    if settings.conversation_log_enabled:
        conversation_run_dir = _resolve_conversation_run_dir(settings, conversation_id)
        request_log_dir = conversation_run_dir / "requests" / request_id
        request_log_dir.mkdir(parents=True, exist_ok=True)
        _write_conversation_meta(conversation_run_dir, conversation_id)
    intent_profile = build_intent_profile_fallback(request.message)
    requested_capabilities = list(intent_profile.get("requested_capabilities") or [])
    focus = request.message
    database_context = _with_schema_hint(request.database_context, settings)
    return RequestStateModel(
        request_id=request_id,
        conversation_id=conversation_id,
        conversation_run_dir=str(conversation_run_dir) if conversation_run_dir else None,
        request_log_dir=str(request_log_dir) if request_log_dir else None,
        message=request.message,
        response_language=detect_response_language(request.message),
        database_context=database_context,
        selected_database=request.selected_database,
        selected_database_type=request.selected_database_type,
        time_range=normalize_time_range(request.time_range),
        constraints=request.constraints,
        history=request.history,
        status="running",
        current_intent="chat_analysis",
        intent_profile=intent_profile,
        requested_capabilities=requested_capabilities,
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


def _resolve_conversation_run_dir(settings: Settings, conversation_id: str) -> Path:
    root = settings.resolved_conversation_log_dir
    root.mkdir(parents=True, exist_ok=True)
    safe_conversation_id = _safe_path_name(conversation_id)
    existing = sorted(root.glob(f"*_{safe_conversation_id}"))
    for path in reversed(existing):
        if path.is_dir():
            return path.resolve()

    local_now = datetime.now().astimezone()
    timestamp = local_now.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / f"{timestamp}_{safe_conversation_id}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{timestamp}_{safe_conversation_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def _write_conversation_meta(conversation_run_dir: Path, conversation_id: str) -> None:
    meta_path = conversation_run_dir / "conversation.json"
    if meta_path.exists():
        return
    local_now = datetime.now().astimezone()
    payload = {
        "conversation_id": conversation_id,
        "created_at_local": local_now.isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": local_now.tzname(),
        "run_dir": str(conversation_run_dir),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_path_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return normalized or "unknown"


def build_conversation_state(request: ChatRequest, conversation_id: str) -> ConversationStateModel:
    return ConversationStateModel(
        conversation_id=conversation_id,
        database_context=request.database_context,
        recent_messages=request.history,
        session_summary=None,
        intent_profile={},
        requested_capabilities=[],
        todo_list=[],
        plan_current_step=0,
        planning_complete=False,
        recent_todo_summary=None,
        latest_database_evidence=None,
        database_evidence_artifacts={},
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
            used_tools=_visible_used_tools(request_state),
            answer=request_state.final_answer_draft,
            trace=trace_events,
            token_usage=token_usage_summary(request_state),
            error=None,
        )

    return ChatResponse(
        conversation_id=request_state.conversation_id or "",
        request_id=request_state.request_id,
        status="failed",
        response_kind="error",
        used_tools=_visible_used_tools(request_state),
        answer=None,
        trace=trace_events,
        token_usage=token_usage_summary(request_state),
        error="Request did not reach a final answer.",
    )


def _visible_used_tools(request_state: RequestStateModel) -> list[str]:
    return [
        call.tool_name
        for call in request_state.tool_history
        if call.tool_name != "terminate"
    ]


def apply_observation(
    request_state: RequestStateModel,
    observation: ToolObservation,
    full_payload: dict,
    tool_spec: "ToolSpec",
    *,
    thought: str | None = None,
    action_reason: str | None = None,
) -> ToolObservation:
    if not observation.success:
        safe_observation = _build_prompt_safe_failure_observation(observation)
        request_state.observations.append(safe_observation)
        return safe_observation

    if tool_spec.result_target == "todo":
        _apply_todo_payload(request_state, full_payload)
    elif tool_spec.result_target == "evidence":
        _apply_evidence_payload(request_state, full_payload)
    elif tool_spec.result_target == "analysis":
        _apply_analysis_payload(request_state, full_payload)
    elif tool_spec.result_target == "presentation":
        _apply_presentation_payload(request_state, full_payload)
        _complete_answer_todo_after_terminal(request_state, observation.tool_name, full_payload)

    safe_observation = enrich_observation_payload(request_state, observation, full_payload, tool_spec)
    request_state.observations.append(safe_observation)
    return safe_observation


async def apply_observation_async(
    request_state: RequestStateModel,
    observation: ToolObservation,
    full_payload: dict,
    tool_spec: "ToolSpec",
    *,
    thought: str | None = None,
    action_reason: str | None = None,
) -> ToolObservation:
    if not observation.success:
        safe_observation = _build_prompt_safe_failure_observation(observation)
        request_state.observations.append(safe_observation)
        return safe_observation

    if tool_spec.result_target == "todo":
        _apply_todo_payload(request_state, full_payload)
    elif tool_spec.result_target == "evidence":
        _apply_evidence_payload(request_state, full_payload)
    elif tool_spec.result_target == "analysis":
        _apply_analysis_payload(request_state, full_payload)
    elif tool_spec.result_target == "presentation":
        _apply_presentation_payload(request_state, full_payload)
        _complete_answer_todo_after_terminal(request_state, observation.tool_name, full_payload)

    safe_observation = enrich_observation_payload(request_state, observation, full_payload, tool_spec)
    request_state.observations.append(safe_observation)
    return safe_observation


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
                payload = _build_prompt_safe_analysis(analysis, request_state)
        elif "forecast_id" in full_payload and request_state.latest_forecast is not None:
            payload = request_state.latest_forecast.model_dump(mode="json")
        elif "anomaly_id" in full_payload and request_state.latest_anomaly is not None:
            payload = request_state.latest_anomaly.model_dump(mode="json")
    elif tool_spec.result_target == "presentation" and request_state.final_answer_draft is not None:
        payload = request_state.final_answer_draft.model_dump(mode="json")

    payload = _deduplicate_observation_payload(observation, payload)

    return observation.model_copy(
        update={
            "payload": payload,
            "payload_truncated": observation.payload_truncated or payload != full_payload,
        }
    )


def _build_prompt_safe_failure_observation(observation: ToolObservation) -> ToolObservation:
    payload = _bounded_value(observation.payload, max_string_chars=1600, max_list_items=8, max_dict_items=12)
    payload = _deduplicate_observation_payload(observation, payload)
    return observation.model_copy(
        update={
            "summary": _truncate_text(observation.summary, 1600),
            "error": _truncate_text(observation.error, 1600),
            "payload": payload,
            "payload_truncated": observation.payload_truncated or _is_large_value(observation.payload),
        }
    )


def _deduplicate_observation_payload(observation: ToolObservation, payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    cleaned = dict(payload)
    envelope_values = {
        "summary": observation.summary,
        "error": observation.error,
        "success": observation.success,
        "tool_name": observation.tool_name,
    }
    for key, envelope_value in envelope_values.items():
        if key in cleaned and cleaned.get(key) == envelope_value:
            cleaned.pop(key, None)
    return cleaned


def _truncate_text(value: str | None, max_chars: int) -> str | None:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _bounded_value(
    value,
    *,
    max_string_chars: int = 800,
    max_list_items: int = 8,
    max_dict_items: int = 12,
):
    if isinstance(value, str):
        return _truncate_text(value, max_string_chars)
    if isinstance(value, list):
        items = [
            _bounded_value(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            items.append({"truncated_items": len(value) - max_list_items})
        return items
    if isinstance(value, dict):
        bounded = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_dict_items:
                bounded["truncated_keys"] = len(value) - max_dict_items
                break
            bounded[key] = _bounded_value(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
        return bounded
    return value


def _is_large_value(value) -> bool:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str)) > 2000
    except Exception:
        return len(str(value)) > 2000


def _apply_todo_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    request_state.todo_list = [
        normalize_todo_for_completion(todo)
        for todo in list(full_payload.get("todos", []))
        if isinstance(todo, dict)
    ]
    request_state.plan_current_step = int(full_payload.get("current_step") or 0)
    request_state.planning_complete = bool(full_payload.get("planning_complete", False))
    if request_state.todo_list:
        request_state.max_iterations = max(
            request_state.max_iterations,
            min(20, len(request_state.todo_list) * 3 + 2),
        )
    request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()


def _complete_answer_todo_after_terminal(
    request_state: RequestStateModel,
    tool_name: str,
    full_payload: dict,
) -> None:
    if not request_state.todo_list:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current_index = next((index for index, todo in enumerate(request_state.todo_list) if todo.get("status") == "in_progress"), None)
    if current_index is None:
        return
    current_todo = dict(request_state.todo_list[current_index])
    if str(current_todo.get("task_type") or "").strip().lower() != "answer":
        request_state.completion_state["latest_step"] = {
            "completed": False,
            "reason": "Terminal payload cannot complete a non-answer todo.",
            "missing_evidence": ["answer_todo"],
            "evidence_refs": [],
            "next_action_hint": "Complete the active evidence or analysis todo before terminating.",
            "tool_name": tool_name,
            "todo_index": current_index,
            "todo": current_todo,
        }
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current_todo = normalize_todo_for_completion(current_todo)
    if not _has_final_answer_payload(full_payload):
        request_state.completion_state["latest_step"] = {
            "completed": False,
            "reason": "Terminal payload did not contain a final answer.",
            "missing_evidence": ["final_answer"],
            "evidence_refs": [],
            "next_action_hint": "Assemble the final answer again after evidence is ready.",
            "tool_name": tool_name,
            "todo_index": current_index,
            "todo": current_todo,
        }
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current_todo["status"] = "completed"
    current_todo["result_ref"] = "final_answer:latest"
    current_todo["completion_reason"] = f"Tool '{tool_name}' produced the final answer."
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
    request_state.completion_state["latest_step"] = {
        "completed": True,
        "reason": current_todo["completion_reason"],
        "missing_evidence": [],
        "evidence_refs": ["final_answer:latest"],
        "next_action_hint": None,
        "tool_name": tool_name,
        "todo_index": current_index,
        "todo": current_todo,
    }
    request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()


def _apply_evidence_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    from schemas.database import DatabaseEvidence

    full_evidence = DatabaseEvidence.model_validate(full_payload)
    diagnostics = dict(full_evidence.diagnostics)
    diagnostics["artifact_kind"] = "database_evidence"
    diagnostics["artifact_ref"] = f"evidence:{full_evidence.evidence_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=full_evidence.evidence_id,
        artifact_kind="database_evidence",
        payload=full_evidence.model_dump(mode="json"),
        directory=_artifact_directory(request_state, "evidence"),
    )
    full_evidence = full_evidence.model_copy(update={"diagnostics": diagnostics})
    request_state.database_evidence_artifacts[full_evidence.evidence_id] = full_evidence
    request_state.latest_database_evidence = _build_prompt_safe_evidence(full_evidence)


def _artifact_directory(request_state: RequestStateModel, artifact_group: str) -> str | Path:
    if request_state.request_log_dir:
        return Path(request_state.request_log_dir) / "artifacts" / artifact_group
    fallback = "analysis_snapshots" if artifact_group == "analysis" else f"{artifact_group}_artifacts"
    return Path(__file__).resolve().parents[1] / "cache_data" / fallback


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
    visible_counts = {
        "points_count": len(data.get("points", [])) if isinstance(data.get("points"), list) else None,
        "rows_count": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
        "series_count": len(data.get("series", [])) if isinstance(data.get("series"), list) else None,
    }
    full_counts = {key: value for key, value in summary_stats.items() if value is not None}
    diagnostics["prompt_sampling"] = {
        "policy": "head_tail_edges",
        "sampled_for_prompt": any(
            isinstance(full_counts.get(key), int)
            and isinstance(visible_counts.get(key), int)
            and visible_counts[key] < full_counts[key]
            for key in ("points_count", "rows_count", "series_count")
        ),
        "full_counts": full_counts,
        "visible_counts": {key: value for key, value in visible_counts.items() if value is not None},
        "full_artifact_ref": f"evidence:{evidence.evidence_id}",
    }
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


def _build_prompt_safe_analysis(analysis, request_state: RequestStateModel):
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
        directory=_artifact_directory(request_state, "analysis"),
    )
    payload["diagnostics"] = diagnostics
    return AnalysisResult.model_validate(payload).model_dump(mode="json")


def _build_prompt_safe_forecast(forecast, request_state: RequestStateModel):
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
        directory=_artifact_directory(request_state, "analysis"),
    )
    payload["diagnostics"] = diagnostics
    return ForecastResult.model_validate(payload)


def _build_prompt_safe_anomaly(anomaly, request_state: RequestStateModel):
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
        directory=_artifact_directory(request_state, "analysis"),
    )
    payload["diagnostics"] = diagnostics
    return AnomalyResult.model_validate(payload)


def _apply_analysis_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    if "analysis_id" in full_payload:
        from schemas.analysis import AnalysisResult

        analysis = AnalysisResult.model_validate(full_payload)
        request_state.analysis_artifacts[analysis.analysis_id] = analysis
        request_state.latest_analysis_id = analysis.analysis_id
        return

    if "forecast_id" in full_payload:
        from schemas.timeseries import ForecastResult

        forecast = ForecastResult.model_validate(full_payload)
        request_state.forecast_artifacts[forecast.forecast_id] = forecast
        request_state.latest_forecast = _build_prompt_safe_forecast(forecast, request_state)
        request_state.visualizations.extend(request_state.latest_forecast.visualizations)
        return

    if "anomaly_id" in full_payload:
        from schemas.timeseries import AnomalyResult

        anomaly = AnomalyResult.model_validate(full_payload)
        request_state.anomaly_artifacts[anomaly.anomaly_id] = anomaly
        request_state.latest_anomaly = _build_prompt_safe_anomaly(anomaly, request_state)
        request_state.visualizations.extend(request_state.latest_anomaly.visualizations)
        return

    if "skill_name" in full_payload:
        request_state.latest_skill = full_payload
        return

    if "results" in full_payload:
        request_state.latest_rag = full_payload


def _apply_presentation_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    request_state.final_answer_draft = FinalAnswer.model_validate(full_payload)


def _has_final_answer_payload(payload: dict) -> bool:
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return True
    sections = payload.get("sections")
    return isinstance(sections, list) and bool(sections)
