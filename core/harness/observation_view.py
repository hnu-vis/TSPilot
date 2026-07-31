"""Bounded observation views for model and public stream consumers."""
from __future__ import annotations

import re
from typing import Any

from schemas.tool import ToolObservation


QUERY_CODE_PATTERNS = (
    re.compile(r"from\s*\([^)]*\)(?:\s*\|>.*?)(?=(?:['`\"。；;]|\n\n|$))", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bselect\b\s+.*?\bfrom\b\s+.*?(?=(?:['`\"。；;]|\n\n|$))", re.IGNORECASE | re.DOTALL),
)

INTERNAL_KEYS = {
    "query",
    "query_language",
    "query_trace",
    "schema_linking",
    "schema_linking_generation",
    "llm_query_generation",
    "query_generation",
    "query_task_contract",
    "executed_query",
    "generated_query",
    "repaired_from_query",
    "previous_error",
    "repair_contract",
    "raw_rule_diagnostics",
    "runtime_ms",
    "sandbox",
    "sql_query",
}


def model_observation_view(observation: ToolObservation | dict | None) -> dict | None:
    """Return the model-visible observation view.

    This is intentionally not the raw observation. It keeps enough result and
    recovery information for the outer ReAct loop while hiding database query
    code, dialect internals, schema-linking traces, and bulky payloads.
    """

    return _observation_view(observation, consumer="model")


def public_observation_view(observation: ToolObservation | dict | None) -> dict | None:
    """Return the SSE/frontend-visible observation view."""

    return _observation_view(observation, consumer="public")


def _observation_view(observation: ToolObservation | dict | None, *, consumer: str) -> dict | None:
    payload = _observation_payload(observation)
    if payload is None:
        return None
    tool_name = str(payload.get("tool_name") or "")
    visible_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    view = {
        "tool_name": tool_name,
        "success": bool(payload.get("success", False)),
        "summary": _strip_query_code(payload.get("summary")),
        "payload": _payload_view(tool_name, visible_payload, consumer=consumer),
    }
    if view["success"] is False:
        failure = _generic_payload_view(visible_payload, consumer=consumer)
        if failure:
            view["payload"] = _drop_empty({**view["payload"], **failure})
    if payload.get("error"):
        view["error"] = _strip_query_code(payload.get("error"))
    if payload.get("payload_truncated"):
        view["payload_truncated"] = True
    if payload.get("payload_ref"):
        view["payload_ref"] = payload.get("payload_ref")
    artifact_ref = _artifact_ref(visible_payload)
    if artifact_ref:
        view["artifact_ref"] = artifact_ref
    return view


def _observation_payload(observation: ToolObservation | dict | None) -> dict | None:
    if observation is None:
        return None
    if isinstance(observation, ToolObservation):
        return observation.model_dump(mode="json")
    if isinstance(observation, dict):
        return observation
    return None


def _payload_view(tool_name: str, payload: dict, *, consumer: str) -> dict:
    if not isinstance(payload, dict):
        return {}
    if tool_name in {"sql_query", "query_database"}:
        return _database_payload_view(payload, consumer=consumer)
    if tool_name == "code_interpreter":
        return _analysis_payload_view(payload)
    if tool_name == "forecast":
        return _forecast_payload_view(payload)
    if tool_name == "anomaly":
        return _anomaly_payload_view(payload)
    if tool_name == "todowrite":
        return _todo_payload_view(payload)
    if tool_name == "terminate":
        return _terminate_payload_view(payload)
    return _generic_payload_view(payload, consumer=consumer)


def _database_payload_view(payload: dict, *, consumer: str) -> dict:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    points = data.get("points") if isinstance(data.get("points"), list) else []
    series = data.get("series") if isinstance(data.get("series"), list) else []
    view = {
        "evidence_id": payload.get("evidence_id"),
        "result_type": payload.get("result_type"),
        "database": payload.get("database"),
        "summary": _strip_query_code(payload.get("summary")),
        "columns": [_public_column_name(column) for column in (payload.get("columns") or [])[:20]]
        if isinstance(payload.get("columns"), list)
        else [],
        "row_count": summary_stats.get("rows_count", len(rows)),
        "point_count": summary_stats.get("points_count", len(points)),
        "series_count": summary_stats.get("series_count", len(series)),
        "data_preview": {},
        "diagnostics": _diagnostics_view(diagnostics),
    }
    if consumer == "public":
        view["query_language"] = payload.get("query_language")
        view["query"] = _truncate_text(str(payload.get("query") or ""), 5000) if payload.get("query") else None
    preview = view["data_preview"]
    if rows:
        preview["rows"] = [_public_row(row) for row in rows[:5]]
    if points:
        preview["points"] = [_public_row(point) for point in points[:6]]
    if series:
        preview["series"] = [_series_view(item) for item in series[:2] if isinstance(item, dict)]
    artifact_ref = _artifact_ref(payload)
    if artifact_ref:
        view["artifact_ref"] = artifact_ref
        view["raw_available_in_artifact"] = True
    produced = payload.get("produced_facts")
    if consumer == "public" and isinstance(produced, list):
        view["produced_fact_count"] = len(produced)
        view["produced_facts_preview"] = [_fact_view(item) for item in produced[:6] if isinstance(item, dict)]
    return _drop_empty(view)


def _analysis_payload_view(payload: dict) -> dict:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    view = {
        "analysis_id": payload.get("analysis_id"),
        "status": payload.get("status"),
        "analysis_goal": payload.get("analysis_goal"),
        "summary": _strip_query_code(payload.get("summary")),
        "input_evidence_id": payload.get("input_evidence_id"),
        "input_row_count": payload.get("input_row_count"),
        "code_type": payload.get("code_type"),
        "code_hash": payload.get("code_hash"),
        "metrics_preview": _bounded_value(result.get("metrics") if isinstance(result.get("metrics"), dict) else {}, max_dict_items=16),
        "details_preview": _bounded_value(result.get("details") if isinstance(result.get("details"), dict) else {}, max_dict_items=12),
        "diagnostics": _diagnostics_view(diagnostics),
    }
    artifact_ref = _artifact_ref(payload)
    if artifact_ref:
        view["artifact_ref"] = artifact_ref
    return _drop_empty(view)


def _forecast_payload_view(payload: dict) -> dict:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    points = payload.get("forecast_points") if isinstance(payload.get("forecast_points"), list) else []
    view = {
        "forecast_id": payload.get("forecast_id"),
        "status": payload.get("status"),
        "model_name": payload.get("model_name"),
        "horizon": payload.get("horizon"),
        "summary": _strip_query_code(payload.get("summary")),
        "forecast_point_count": len(points),
        "forecast_points_preview": [_public_row(point) for point in _sample_edges(points, limit=6)],
        "diagnostics": _diagnostics_view(diagnostics),
    }
    artifact_ref = _artifact_ref(payload)
    if artifact_ref:
        view["artifact_ref"] = artifact_ref
    return _drop_empty(view)


def _anomaly_payload_view(payload: dict) -> dict:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    points = payload.get("anomaly_points") if isinstance(payload.get("anomaly_points"), list) else []
    scores = payload.get("scores") if isinstance(payload.get("scores"), list) else []
    view = {
        "anomaly_id": payload.get("anomaly_id"),
        "detector_name": payload.get("detector_name"),
        "summary": _strip_query_code(payload.get("summary")),
        "anomaly_point_count": len(points),
        "anomaly_points_preview": [_public_row(point) for point in _sample_edges(points, limit=6)],
        "scores_preview": _sample_edges(scores, limit=6),
        "diagnostics": _diagnostics_view(diagnostics),
    }
    artifact_ref = _artifact_ref(payload)
    if artifact_ref:
        view["artifact_ref"] = artifact_ref
    return _drop_empty(view)


def _todo_payload_view(payload: dict) -> dict:
    todos = payload.get("todos") if isinstance(payload.get("todos"), list) else []
    return _drop_empty(
        {
            "current_step": payload.get("current_step"),
            "planning_complete": payload.get("planning_complete"),
            "todo_total": len(todos),
            "completed_count": payload.get("completed_count"),
            "pending_count": payload.get("pending_count"),
            "todos_preview": [
                {
                    key: _strip_query_code(todo.get(key))
                    for key in ("content", "task_type", "status", "priority", "acceptance_criteria")
                    if isinstance(todo, dict) and todo.get(key) not in (None, "", [], {})
                }
                for todo in todos[:8]
                if isinstance(todo, dict)
            ],
        }
    )


def _terminate_payload_view(payload: dict) -> dict:
    answer = payload.get("answer") if isinstance(payload.get("answer"), dict) else {}
    return _drop_empty(
        {
            "title": answer.get("title") if answer else payload.get("title"),
            "summary": _strip_query_code(answer.get("summary") if answer else payload.get("summary")),
            "section_count": len(answer.get("sections", [])) if isinstance(answer.get("sections"), list) else None,
        }
    )


def _generic_payload_view(payload: dict, *, consumer: str) -> dict:
    keys = (
        "error",
        "recovery_hint",
        "error_type",
        "retryable",
        "rejected_action",
        "recommended_next_action",
        "recommended_strategy",
        "blocked_strategy",
        "failure_signature",
        "repeated_failure_count",
        "next_action_constraints",
        "evidence_id",
        "analysis_id",
        "forecast_id",
        "anomaly_id",
        "current_step",
        "planning_complete",
        "allowed_next_actions",
    )
    view = {}
    for key in keys:
        if key in payload:
            view[key] = _sanitize_value(payload.get(key), max_string_chars=900)
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    if diagnostics:
        view["diagnostics"] = _diagnostics_view(diagnostics)
    validation_failure = payload.get("validation_failure") if isinstance(payload.get("validation_failure"), dict) else None
    if validation_failure:
        view["validation_failure"] = _validation_failure_view(validation_failure, include_retry=(consumer == "model"))
    return _drop_empty(view)


def _diagnostics_view(diagnostics: dict) -> dict:
    view = {}
    for key in ("artifact_ref", "summary_stats", "prompt_sampling", "recommended_downstream_action", "next_action_hint", "coverage", "strategy_hint"):
        if key in diagnostics:
            view[key] = _sanitize_value(diagnostics.get(key), max_string_chars=700)
    return _drop_empty(view)


def _validation_failure_view(validation_failure: dict, *, include_retry: bool) -> dict:
    view = {}
    for key in ("tool_name", "error_type", "message", "missing_required_filters", "missing_requirements"):
        if key in validation_failure:
            view[key] = _sanitize_value(validation_failure.get(key), max_string_chars=700)
    if include_retry and isinstance(validation_failure.get("retry_policy"), dict):
        view["retry_policy"] = _sanitize_value(validation_failure.get("retry_policy"), max_string_chars=500)
    return _drop_empty(view)


def _series_view(series: dict) -> dict:
    view = {}
    for key, value in series.items():
        if key in {"points", "rows"}:
            continue
        if key in {"value_field", "time_field"}:
            continue
        view[_public_column_name(key)] = _sanitize_value(value, max_string_chars=300)
    points = series.get("points") if isinstance(series.get("points"), list) else []
    rows = series.get("rows") if isinstance(series.get("rows"), list) else []
    if points:
        view["points_count"] = series.get("points_count") or len(points)
        view["points_preview"] = [_public_row(point) for point in _sample_edges(points, limit=4)]
    if rows:
        view["rows_count"] = series.get("rows_count") or len(rows)
        view["rows_preview"] = [_public_row(row) for row in _sample_edges(rows, limit=4)]
    return _drop_empty(view)


def _fact_view(fact: dict) -> dict:
    return _drop_empty(
        {
            "fact_id": fact.get("fact_id"),
            "name": fact.get("name"),
            "fact_type": fact.get("fact_type"),
            "statement": _strip_query_code(fact.get("statement")),
            "value": _sanitize_value(fact.get("value"), max_string_chars=300),
            "status": fact.get("status"),
        }
    )


def _artifact_ref(payload: dict) -> str | None:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    ref = diagnostics.get("artifact_ref") or payload.get("artifact_ref")
    if isinstance(ref, str) and ref.strip():
        return ref.strip()
    snapshot = diagnostics.get("snapshot_ref") if isinstance(diagnostics.get("snapshot_ref"), dict) else {}
    kind = snapshot.get("artifact_kind")
    artifact_id = snapshot.get("artifact_id")
    if kind and artifact_id:
        return f"{kind}:{artifact_id}"
    return None


def _public_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return _sanitize_value(row, max_string_chars=300)
    normalized = {}
    for key, value in row.items():
        name = _public_column_name(key)
        if not name:
            continue
        if name in normalized:
            suffix = 2
            candidate = f"{name}_{suffix}"
            while candidate in normalized:
                suffix += 1
                candidate = f"{name}_{suffix}"
            name = candidate
        normalized[name] = _sanitize_value(value, max_string_chars=300)
    return normalized


def _public_column_name(column: Any) -> str:
    text = str(column or "").strip()
    if text.startswith("_"):
        text = text.lstrip("_")
    return text


def _sanitize_value(value: Any, *, max_string_chars: int = 700) -> Any:
    if isinstance(value, str):
        return _truncate_text(_strip_query_code(value), max_string_chars)
    if isinstance(value, list):
        return [_sanitize_value(item, max_string_chars=max_string_chars) for item in value[:8]]
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 12:
                result["truncated_keys"] = len(value) - 12
                break
            if str(key) in INTERNAL_KEYS:
                continue
            result[_public_column_name(key)] = _sanitize_value(item, max_string_chars=max_string_chars)
        return _drop_empty(result)
    return value


def _bounded_value(value: Any, *, max_dict_items: int = 12) -> Any:
    value = _sanitize_value(value)
    if isinstance(value, dict) and len(value) > max_dict_items:
        items = list(value.items())[:max_dict_items]
        return {**dict(items), "truncated_keys": len(value) - max_dict_items}
    return value


def _strip_query_code(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value
    for marker in (" for query '", " for query `", " Query statement:", "\nQuery statement:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    for pattern in QUERY_CODE_PATTERNS:
        text = pattern.sub("[query omitted]", text)
    return _truncate_text(text, 1200)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _sample_edges(items: list, *, limit: int) -> list:
    if len(items) <= limit:
        return items
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return [*items[:head], *items[-tail:]]


def _drop_empty(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }
