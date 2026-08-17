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
    rendered_payload = _payload_view(tool_name, visible_payload, consumer=consumer)
    if isinstance(visible_payload.get("coverage_delta"), dict):
        rendered_payload["coverage_delta"] = _sanitize_value(
            visible_payload["coverage_delta"],
            max_string_chars=700,
        )
    view = {
        "tool_name": tool_name,
        "success": bool(payload.get("success", False)),
        "summary": _strip_query_code(payload.get("summary")),
        "payload": rendered_payload,
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
    if tool_name == "sql_query":
        return _database_payload_view(payload, consumer=consumer)
    if tool_name == "code_interpreter":
        return _analysis_payload_view(payload, consumer=consumer)
    if tool_name == "forecast":
        return _forecast_payload_view(payload)
    if tool_name == "anomaly":
        return _anomaly_payload_view(payload)
    if tool_name == "visualization":
        return _visualization_payload_view(payload)
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
    if consumer == "model":
        # ReAct needs the shape and integrity of the produced artifact, not
        # three overlapping representations of its rows. Exact values and
        # full data remain available through evidence/insight state.
        view.pop("data_preview", None)
        view["full_fidelity"] = _full_fidelity(diagnostics)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        if isinstance(metadata.get("time_range"), dict):
            view["time_range"] = metadata["time_range"]
        if isinstance(payload.get("insight_coverage"), dict):
            view["insight_coverage"] = _insight_coverage_receipt(payload["insight_coverage"])
        view["diagnostics"] = _react_diagnostics_view(diagnostics)
    if consumer == "public":
        view["query_language"] = payload.get("query_language")
        view["query"] = str(payload.get("query") or "") if payload.get("query") else None
    if consumer == "public":
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
        if consumer == "public":
            view["raw_available_in_artifact"] = True
    produced = payload.get("produced_insights")
    if consumer == "public" and isinstance(produced, list):
        view["produced_insight_count"] = len(produced)
        view["produced_insights_preview"] = [_insight_view(item) for item in produced[:6] if isinstance(item, dict)]
    if consumer == "public" and isinstance(payload.get("insight_coverage"), dict):
        view["insight_coverage"] = payload["insight_coverage"]
    return _drop_empty(view)


def _analysis_payload_view(payload: dict, *, consumer: str) -> dict:
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
        "data_views": _bounded_value(
            payload.get("data_views") if isinstance(payload.get("data_views"), list) else [],
            max_dict_items=12,
        ),
        "diagnostics": _diagnostics_view(diagnostics),
    }
    executed_code = diagnostics.get("executed_code_preview") if isinstance(diagnostics.get("executed_code_preview"), dict) else {}
    generated_code = diagnostics.get("generated_code_preview") if isinstance(diagnostics.get("generated_code_preview"), dict) else {}
    code_view = executed_code or generated_code
    full_code = diagnostics.get("executed_code")
    if consumer == "public" and isinstance(full_code, str) and full_code.strip():
        view["code_preview"] = full_code
        view["analysis_code_chars"] = len(full_code)
    elif consumer == "public" and isinstance(code_view.get("preview"), str):
        view["code_preview"] = code_view["preview"]
        view["analysis_code_chars"] = code_view.get("char_count") or len(code_view["preview"])
    if consumer == "public" and diagnostics.get("runtime_ms") is not None:
        view["runtime_ms"] = diagnostics.get("runtime_ms")
    artifact_ref = _artifact_ref(payload)
    if artifact_ref:
        view["artifact_ref"] = artifact_ref
    produced = payload.get("produced_insights")
    if consumer == "public" and isinstance(produced, list):
        view["produced_insight_count"] = len(produced)
        view["produced_insights_preview"] = [_insight_view(item) for item in produced[:6] if isinstance(item, dict)]
    if consumer == "public" and isinstance(payload.get("insight_coverage"), dict):
        view["insight_coverage"] = payload["insight_coverage"]
    return _drop_empty(view)


def _forecast_payload_view(payload: dict) -> dict:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    points = payload.get("forecast_points") if isinstance(payload.get("forecast_points"), list) else []
    point_count = diagnostics.get("forecast_point_count")
    if not isinstance(point_count, int) or point_count < len(points):
        point_count = len(points)
    view = {
        "forecast_id": payload.get("forecast_id"),
        "status": payload.get("status"),
        "model_name": payload.get("model_name"),
        "horizon": payload.get("horizon"),
        "summary": _strip_query_code(payload.get("summary")),
        "forecast_point_count": point_count,
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
    public_todos = [
        {
            key: _strip_query_code(todo.get(key))
            for key in ("content", "task_type", "status", "priority", "acceptance_criteria", "result_ref", "completion_reason")
            if isinstance(todo, dict) and todo.get(key) not in (None, "", [], {})
        }
        for todo in todos[:8]
        if isinstance(todo, dict)
    ]
    return _drop_empty(
        {
            "current_step": payload.get("current_step"),
            "planning_complete": payload.get("planning_complete"),
            "todo_total": len(todos),
            "completed_count": payload.get("completed_count"),
            "pending_count": payload.get("pending_count"),
            "todos": public_todos,
            "todos_preview": public_todos,
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


def _visualization_payload_view(payload: dict) -> dict:
    visualizations = payload.get("visualizations") if isinstance(payload.get("visualizations"), list) else []
    return _drop_empty({
        "visualization_ids": payload.get("visualization_ids") or [
            item.get("visualization_id") for item in visualizations if isinstance(item, dict)
        ],
        "grounded_by": payload.get("grounded_by", []),
        "verification": payload.get("verification", []),
        "coverage_delta": payload.get("coverage_delta"),
    })


def _full_fidelity(diagnostics: dict) -> bool | None:
    value = diagnostics.get("is_full_fidelity")
    if isinstance(value, bool):
        return value
    sampling = diagnostics.get("prompt_sampling")
    if isinstance(sampling, dict):
        value = sampling.get("is_full_fidelity")
        if isinstance(value, bool):
            return value
        if sampling.get("full_artifact_ref"):
            return True
    return None


def _insight_coverage_receipt(coverage: dict) -> dict:
    return _drop_empty({
        key: coverage.get(key)
        for key in ("requested", "verified", "missing", "unavailable", "rejected", "partial")
    })


def _react_diagnostics_view(diagnostics: dict) -> dict:
    """Keep only diagnostics that can change the next ReAct action."""
    return _drop_empty({
        "coverage": diagnostics.get("coverage"),
        "recommended_downstream_action": diagnostics.get("recommended_downstream_action"),
        "next_action_hint": diagnostics.get("next_action_hint"),
    })


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
        view["validation_failure"] = _validation_failure_view(validation_failure, include_retry=True)
    return _drop_empty(view)


def _diagnostics_view(diagnostics: dict) -> dict:
    view = {}
    for key in ("artifact_ref", "summary_stats", "prompt_sampling", "recommended_downstream_action", "next_action_hint", "coverage", "strategy_hint"):
        if key in diagnostics:
            view[key] = _sanitize_value(diagnostics.get(key), max_string_chars=700)
    return _drop_empty(view)


def _validation_failure_view(validation_failure: dict, *, include_retry: bool) -> dict:
    view = {}
    for key in (
        "tool",
        "tool_name",
        "scope",
        "capability",
        "error_code",
        "error_type",
        "message",
        "missing_required_filters",
        "missing_requirements",
    ):
        if key in validation_failure:
            view[key] = _sanitize_value(validation_failure.get(key), max_string_chars=700)
    repair_contract = validation_failure.get("repair_contract")
    if include_retry and isinstance(repair_contract, dict):
        view["repair_contract"] = _repair_contract_view(repair_contract)
    if include_retry and isinstance(validation_failure.get("retry_policy"), dict):
        view["retry_policy"] = _sanitize_value(validation_failure.get("retry_policy"), max_string_chars=500)
    return _drop_empty(view)


def _repair_contract_view(repair_contract: dict) -> dict:
    view = {}
    for key in (
        "mode",
        "input_evidence",
        "analysis_goal",
        "required_metrics",
        "missing_metrics",
        "required_details_fields",
        "available_inputs",
        "canonical_inputs",
        "error_classification",
        "failed_code_summary",
        "instruction",
        "expected_result_shape",
    ):
        if key in repair_contract:
            view[key] = _sanitize_value(repair_contract.get(key), max_string_chars=700)
    if "failed_code" in repair_contract:
        view["failed_code"] = str(repair_contract.get("failed_code") or "")
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


def _insight_view(insight: dict) -> dict:
    evidence_refs = insight.get("evidence_refs") if isinstance(insight.get("evidence_refs"), list) else []
    return _drop_empty(
        {
            "insight_id": insight.get("insight_id"),
            "insight_key": insight.get("insight_key"),
            "name": insight.get("name"),
            "insight_type": insight.get("insight_type"),
            "statement": _strip_query_code(insight.get("statement")),
            "value": _sanitize_value(insight.get("value"), max_string_chars=300),
            "unit": insight.get("unit"),
            "method": insight.get("method"),
            "status": insight.get("status"),
            "derived_from": _sanitize_value(insight.get("derived_from"), max_string_chars=160),
            "evidence_refs": [
                _drop_empty(
                    {
                        "source_type": item.get("source_type"),
                        "source_id": item.get("source_id"),
                        "locator": _sanitize_value(item.get("locator"), max_string_chars=160),
                    }
                )
                for item in evidence_refs[:6]
                if isinstance(item, dict)
            ],
            "calculation_trace": _sanitize_value(insight.get("calculation_trace"), max_string_chars=300),
            "unavailable_reason": _sanitize_value(insight.get("unavailable_reason"), max_string_chars=300),
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
