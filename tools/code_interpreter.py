"""Subprocess code interpreter tool for grounded data analysis."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from core.analysis.python_runner import AnalysisCodeError
from sandbox import execute_python_sandbox_v1
from schemas.analysis import AnalysisResult
from schemas.database import DatabaseEvidence
from schemas.data_fact import DataFactRequest
from tools.base import BaseTool, StructuredToolError


class CodeInterpreterInput(BaseModel):
    mode: str | None = None
    repair_contract: dict | None = None
    database_evidence: DatabaseEvidence | dict | str | None = None
    analysis_goal: str | None = None
    code: str | None = None
    analysis_request: dict | None = None
    required_outputs: list[str] = Field(default_factory=list)
    expected_result_schema: dict | None = None
    constraints: dict | None = Field(default_factory=dict)
    fact_requests: list[DataFactRequest] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_code_aliases(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("code") and data.get("analysis_code"):
                data["code"] = data["analysis_code"]
            data["analysis_request"] = _normalize_analysis_request(data.get("analysis_request"))
            data["required_outputs"] = _normalize_required_outputs(data.get("required_outputs"))
            if not isinstance(data.get("constraints"), dict):
                data["constraints"] = (
                    {"value": data.get("constraints")}
                    if data.get("constraints") not in (None, "", [], {})
                    else {}
                )
        return data


class CodeInterpreterTool(BaseTool):
    async def execute(self, validated_input: CodeInterpreterInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = _resolve_database_evidence(database_evidence, request_state)
        if database_evidence is None:
            raise ValueError("code_interpreter requires database_evidence or a latest_database_evidence in request state.")
        rows, points, columns = _analysis_inputs(database_evidence)
        goal = validated_input.analysis_goal or "Code interpreter analysis"
        constraints = validated_input.constraints or {}
        if not validated_input.code or not validated_input.code.strip():
            output = _execute_analysis_request(
                rows=rows,
                points=points,
                columns=columns,
                goal=goal,
                required_outputs=validated_input.required_outputs,
                analysis_request=validated_input.analysis_request,
                constraints=constraints,
            )
            code_type = "analysis_request_v1"
            code_hash = _code_hash(json.dumps(output.get("diagnostics", {}), ensure_ascii=False, sort_keys=True, default=str))
            runtime_ms = 0
        else:
            code_hash = _code_hash(validated_input.code)
            requires_numeric_series = _analysis_requires_numeric_series(goal, validated_input.expected_result_schema or {})
            if requires_numeric_series and rows and not points:
                raise AnalysisCodeError(
                    "code_interpreter analysis requires numeric time-series values, but the selected evidence "
                    "does not expose a usable timestamp/value pair. Query evidence with a numeric value column first."
                )
            sandbox_output = execute_python_sandbox_v1(
                code=validated_input.code,
                rows=rows,
                points=points,
                columns=columns,
                metadata=database_evidence.metadata,
                diagnostics=database_evidence.diagnostics,
                timeout_seconds=int(constraints.get("timeout_seconds", 5)),
                work_dir=_code_interpreter_work_dir(request_state, code_hash),
            )
            output = {"result": sandbox_output.result, "diagnostics": {"runtime_ms": sandbox_output.runtime_ms}}
            code_type = "code_interpreter_v1"
            runtime_ms = sandbox_output.runtime_ms
        requires_numeric_series = _analysis_requires_numeric_series(goal, validated_input.expected_result_schema or {})
        result_payload = output["result"]
        try:
            _validate_expected_result_schema(result_payload, validated_input.expected_result_schema or {})
            _validate_outlier_treatment_transparency(result_payload)
            _validate_result_has_numeric_analysis(result_payload, requires_numeric_series=requires_numeric_series, input_rows=len(rows))
        except AnalysisCodeError as exc:
            raise _structured_analysis_validation_error(
                exc,
                goal=goal,
                database_evidence=database_evidence,
                result_payload=result_payload,
            ) from exc
        result = AnalysisResult(
            analysis_id=_analysis_id(database_evidence.evidence_id, goal, code_hash),
            analysis_goal=goal,
            code_type=code_type,
            code_hash=code_hash,
            input_evidence_id=database_evidence.evidence_id,
            input_row_count=len(rows),
            status="succeeded",
            summary=str(result_payload["summary"]),
            result=result_payload,
            diagnostics={
                "runtime_ms": runtime_ms,
                "expected_result_schema": validated_input.expected_result_schema or {},
                "input_columns": columns,
                "input_points_count": len(points),
                "sandbox": "subprocess_code_interpreter_v1" if code_type == "code_interpreter_v1" else "analysis_request_template_v1",
                **output.get("diagnostics", {}),
            },
        )
        return result.model_dump(mode="json")


def _resolve_database_evidence(database_evidence, request_state):
    if database_evidence is None:
        latest = request_state.latest_database_evidence
        if latest is None:
            return None
        return request_state.database_evidence_artifacts.get(latest.evidence_id, latest)
    if isinstance(database_evidence, str):
        evidence_ref = database_evidence.strip()
        if evidence_ref in {"latest", "latest_database_evidence", "current"}:
            return _resolve_database_evidence(None, request_state)
        if evidence_ref.startswith("evidence:"):
            evidence_ref = evidence_ref.split(":", 1)[1]
        resolved = request_state.database_evidence_artifacts.get(evidence_ref)
        if resolved is None:
            raise ValueError(f"code_interpreter could not resolve database_evidence reference: {database_evidence}")
        return resolved
    if isinstance(database_evidence, dict):
        evidence_id = database_evidence.get("evidence_id")
        if evidence_id:
            return request_state.database_evidence_artifacts.get(evidence_id) or DatabaseEvidence.model_validate(database_evidence)
        latest = getattr(request_state, "latest_database_evidence", None)
        if latest is not None:
            return request_state.database_evidence_artifacts.get(latest.evidence_id, latest)
        return DatabaseEvidence.model_validate(database_evidence)
    return request_state.database_evidence_artifacts.get(database_evidence.evidence_id, database_evidence)


def _normalize_analysis_request(value) -> dict | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, dict):
        normalized = dict(value)
        if "required_outputs" in normalized:
            normalized["required_outputs"] = _normalize_required_outputs(normalized.get("required_outputs"))
        return normalized
    if isinstance(value, list):
        return {"required_outputs": _normalize_required_outputs(value)}
    return {"goal": str(value)}


def _normalize_required_outputs(value) -> list[str]:
    if value in (None, "", False):
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                label = item.get("id") or item.get("description") or item.get("output_type") or item.get("evidence_kind")
            else:
                label = item
            if str(label or "").strip():
                result.append(str(label).strip())
        return result
    if isinstance(value, dict):
        label = value.get("id") or value.get("description") or value.get("output_type") or value.get("evidence_kind")
        return [str(label).strip()] if str(label or "").strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _analysis_inputs(evidence: DatabaseEvidence) -> tuple[list[dict], list[dict], list[str]]:
    data = evidence.data or {}
    time_field = str(data.get("time_field") or "timestamp")
    value_field = str(data.get("value_field") or "value")
    raw_rows = data.get("rows")
    if isinstance(raw_rows, list) and raw_rows:
        rows = [dict(row) for row in raw_rows if isinstance(row, dict)]
    else:
        rows = []
        for point in data.get("points", []) or []:
            if not isinstance(point, dict):
                continue
            row = {
                time_field: point.get("timestamp"),
                value_field: point.get("value"),
            }
            for key, value in point.items():
                if key not in {"timestamp", "value"}:
                    row[key] = value
            rows.append(row)
    points = [dict(point) for point in data.get("points", []) or [] if isinstance(point, dict)]
    if not points and rows:
        inferred_time_field = _first_present_key(rows, [time_field, "timestamp", "_time", "time"])
        inferred_value_field = _first_present_key(rows, [value_field, "value", "price", "_value"])
        if inferred_time_field and inferred_value_field:
            points = []
            for row in rows:
                point = {
                    "timestamp": row.get(inferred_time_field),
                    "value": row.get(inferred_value_field),
                }
                for key, value in row.items():
                    if key not in {inferred_time_field, inferred_value_field}:
                        point[key] = value
                points.append(point)
    columns = list(evidence.columns or [])
    if not columns and rows:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    return rows, points, columns


def _execute_analysis_request(
    *,
    rows: list[dict],
    points: list[dict],
    columns: list[str],
    goal: str,
    required_outputs: list[str],
    analysis_request: dict | None,
    constraints: dict,
) -> dict:
    canonical_points = _canonical_numeric_points(rows=rows, points=points)
    if not canonical_points:
        raise AnalysisCodeError("analysis_request requires row or point evidence with numeric values.")
    sorted_points = sorted(canonical_points, key=lambda item: str(item.get("timestamp") or ""))
    values = [float(item["value"]) for item in sorted_points]
    first = sorted_points[0]
    last = sorted_points[-1]
    max_point = max(sorted_points, key=lambda item: float(item["value"]))
    min_point = min(sorted_points, key=lambda item: float(item["value"]))
    metrics = {
        "record_count": len(sorted_points),
        "start_value": first["value"],
        "end_value": last["value"],
        "highest_value": max_point["value"],
        "lowest_value": min_point["value"],
        "max_value": max_point["value"],
        "min_value": min_point["value"],
        "max_min_difference": float(max_point["value"]) - float(min_point["value"]),
        "difference": float(max_point["value"]) - float(min_point["value"]),
        "start_end_change": float(last["value"]) - float(first["value"]),
        "change": float(last["value"]) - float(first["value"]),
    }
    details = {
        "start_time": first.get("timestamp"),
        "end_time": last.get("timestamp"),
        "max_time": max_point.get("timestamp"),
        "min_time": min_point.get("timestamp"),
        "first_point": first,
        "last_point": last,
        "max_point": max_point,
        "min_point": min_point,
        "columns": columns,
        "requested_outputs": _string_list(required_outputs or (analysis_request or {}).get("required_outputs")),
    }
    if _analysis_context_indicates_outlier_treatment(goal, analysis_request, constraints):
        details.update(
            {
                "outlier_rule": "canonical_transparency_template_no_exclusion",
                "threshold_or_formula": "No exclusion threshold was applied by the canonical metrics template.",
                "rationale": (
                    "The analysis goal requested anomaly/outlier handling. "
                    "This template reports transparent raw and adjusted metrics; no rows are excluded unless a detector explicitly supplies excluded_rows."
                ),
                "excluded_rows": [],
                "raw_metrics": dict(metrics),
                "adjusted_metrics": dict(metrics),
            }
        )
    summary = _analysis_request_summary(goal, metrics, details)
    return {
        "result": {
            "summary": summary,
            "metrics": metrics,
            "details": details,
        },
        "diagnostics": {
            "analysis_request": analysis_request or {},
            "template": "canonical_timeseries_metrics_v1",
            "constraints": constraints,
            "input_point_count": len(sorted_points),
        },
    }


def _analysis_context_indicates_outlier_treatment(goal: str, analysis_request: dict | None, constraints: dict | None) -> bool:
    repair_contract = None
    if isinstance(constraints, dict):
        repair_contract = constraints.get("_repair_contract")
    text = " ".join(
        [
            goal or "",
            _flatten_text(analysis_request or {}),
            _flatten_text(repair_contract or {}),
        ]
    ).lower()
    return any(
        marker in text
        for marker in (
            "outlier",
            "anomaly",
            "anomalous",
            "excluded",
            "异常",
            "离群",
            "剔除",
            "排除",
        )
    )


def _structured_analysis_validation_error(
    exc: AnalysisCodeError,
    *,
    goal: str,
    database_evidence: DatabaseEvidence,
    result_payload: dict,
) -> StructuredToolError:
    message = str(exc)
    required_fields = [
        "outlier_rule",
        "threshold_or_formula",
        "rationale",
        "excluded_rows",
        "raw_metrics",
        "adjusted_metrics",
    ]
    if "transparent details fields" in message or "outlier treatment" in message:
        error_code = "analysis_transparency_missing"
        repair_contract = {
            "mode": "analysis_artifact_repair",
            "input_evidence": database_evidence.evidence_id,
            "analysis_goal": goal,
            "required_details_fields": required_fields,
            "previous_result_summary": result_payload.get("summary") if isinstance(result_payload, dict) else None,
            "expected_result_shape": {"summary": "string", "metrics": "object", "details": "object"},
        }
        validation_failure = {
            "scope": "artifact_output",
            "capability": "analysis",
            "tool": "code_interpreter",
            "error_code": error_code,
            "message": message,
            "failed_artifact": {
                "input_evidence_id": database_evidence.evidence_id,
                "result_summary": result_payload.get("summary") if isinstance(result_payload, dict) else None,
            },
            "required_contract": {"required_details_fields": required_fields},
            "repair_contract": repair_contract,
            "retry_policy": {
                "required_action": "code_interpreter",
                "max_equivalent_retries": 2,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        }
        return StructuredToolError(
            message,
            error_type=error_code,
            retryable=True,
            recommended_next_action="code_interpreter",
            diagnostics={"repair_contract": repair_contract, "required_details_fields": required_fields},
            validation_failure=validation_failure,
        )
    return StructuredToolError(
        message,
        error_type="analysis_validation_failed",
        retryable=True,
        recommended_next_action="code_interpreter",
        diagnostics={},
        validation_failure={
            "scope": "artifact_output",
            "capability": "analysis",
            "tool": "code_interpreter",
            "error_code": "analysis_validation_failed",
            "message": message,
            "failed_artifact": {"input_evidence_id": database_evidence.evidence_id},
            "required_contract": {},
            "repair_contract": {
                "mode": "analysis_artifact_repair",
                "input_evidence": database_evidence.evidence_id,
                "analysis_goal": goal,
            },
            "retry_policy": {
                "required_action": "code_interpreter",
                "max_equivalent_retries": 2,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        },
    )


def _canonical_numeric_points(*, rows: list[dict], points: list[dict]) -> list[dict]:
    candidates = points if points else rows
    time_key = _first_present_key(candidates, ["timestamp", "_time", "time", "date"])
    value_key = _first_present_key(candidates, ["value", "_value", "price", "close", "amount"])
    if not value_key:
        value_key = _first_numeric_key(candidates)
    normalized = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        value = _number(item.get(value_key)) if value_key else None
        if value is None:
            continue
        normalized.append(
            {
                "timestamp": item.get(time_key) if time_key else None,
                "value": value,
                "row": item,
            }
        )
    return normalized


def _first_numeric_key(rows: list[dict]) -> str | None:
    keys = []
    for row in rows[:20]:
        for key in row:
            if key not in keys:
                keys.append(key)
    for key in keys:
        if any(_number(row.get(key)) is not None for row in rows[:20]):
            return key
    return None


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _analysis_request_summary(goal: str, metrics: dict, details: dict) -> str:
    return (
        f"{goal}: record_count={metrics['record_count']}, "
        f"start_value={metrics['start_value']} at {details.get('start_time')}, "
        f"end_value={metrics['end_value']} at {details.get('end_time')}, "
        f"max_value={metrics['max_value']} at {details.get('max_time')}, "
        f"min_value={metrics['min_value']} at {details.get('min_time')}, "
        f"difference={metrics['difference']}."
    )


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_present_key(rows: list[dict], candidates: list[str]) -> str | None:
    for key in candidates:
        if any(key in row for row in rows):
            return key
    return None


def _validate_expected_result_schema(result: dict, schema: dict) -> None:
    if not schema:
        return
    normalized_schema = _normalize_expected_result_schema(result, schema)
    if normalized_schema is not schema:
        schema = normalized_schema
    _validate_schema_node(result, schema, path="result")


def _normalize_expected_result_schema(result: dict, schema: dict) -> dict:
    if not isinstance(schema, dict) or not isinstance(result, dict):
        return schema
    structural_keys = {"summary", "metrics", "details"}
    if set(schema) & structural_keys:
        return schema
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    schema_keys = {str(key) for key in schema}
    if schema_keys and schema_keys <= set(metrics):
        return {"metrics": schema}
    if schema_keys and schema_keys <= set(details):
        return {"details": schema}
    return schema


def _validate_outlier_treatment_transparency(result: dict) -> None:
    if not _result_indicates_outlier_treatment(result):
        return
    details = result.get("details")
    if not isinstance(details, dict):
        raise AnalysisCodeError("analysis result using outlier treatment must include result.details.")
    required = {
        "outlier_rule",
        "threshold_or_formula",
        "rationale",
        "excluded_rows",
        "raw_metrics",
        "adjusted_metrics",
    }
    missing = sorted(key for key in required if key not in details)
    if missing:
        raise AnalysisCodeError(
            "analysis result using outlier treatment must include transparent details fields: "
            + ", ".join(missing)
        )
    if not str(details.get("outlier_rule") or "").strip():
        raise AnalysisCodeError("analysis result using outlier treatment must include non-empty details.outlier_rule.")
    if not str(details.get("rationale") or "").strip():
        raise AnalysisCodeError("analysis result using outlier treatment must include non-empty details.rationale.")
    if not isinstance(details.get("excluded_rows"), list):
        raise AnalysisCodeError("analysis result using outlier treatment must include details.excluded_rows as a list.")
    duplicate_excluded_rows = _duplicate_json_rows(details["excluded_rows"])
    if duplicate_excluded_rows:
        raise AnalysisCodeError("analysis result using outlier treatment must not include duplicate details.excluded_rows.")
    if not isinstance(details.get("raw_metrics"), dict):
        raise AnalysisCodeError("analysis result using outlier treatment must include details.raw_metrics as an object.")
    if not isinstance(details.get("adjusted_metrics"), dict):
        raise AnalysisCodeError("analysis result using outlier treatment must include details.adjusted_metrics as an object.")


def _analysis_requires_numeric_series(goal: str | None, expected_schema: dict) -> bool:
    text = " ".join([goal or "", _flatten_text(expected_schema)]).lower()
    markers = {
        "return",
        "returns",
        "volatility",
        "drawdown",
        "yield",
        "pct_change",
        "percentage_change",
        "price change",
        "trend",
        "收益",
        "收益率",
        "波动",
        "回撤",
        "涨跌",
        "价格变化",
        "趋势",
    }
    return any(marker in text for marker in markers)


def _validate_result_has_numeric_analysis(result: dict, *, requires_numeric_series: bool, input_rows: int) -> None:
    if not requires_numeric_series or input_rows == 0:
        return
    metrics = result.get("metrics") if isinstance(result, dict) else None
    if not isinstance(metrics, dict) or not metrics:
        raise AnalysisCodeError("code_interpreter numeric analysis must return non-empty result.metrics.")

    meaningful_values = [
        value for key, value in metrics.items()
        if str(key).lower() not in {"row_count", "count", "n", "series_length", "return_count"}
    ]
    if meaningful_values and any(value is not None for value in meaningful_values):
        return

    details = result.get("details") if isinstance(result, dict) else None
    if isinstance(details, dict):
        for key in ("raw_metrics", "adjusted_metrics"):
            nested_metrics = details.get(key)
            if isinstance(nested_metrics, dict) and any(value is not None for value in nested_metrics.values()):
                return

    raise AnalysisCodeError(
        "code_interpreter numeric analysis produced no non-empty computed metric values; "
        "use evidence with numeric values and compute the requested metrics."
    )


def _result_indicates_outlier_treatment(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    textual_markers = {
        "outlier",
        "anomaly",
        "anomalous",
        "excluded",
        "exclude",
        "filtered",
        "filter",
        "winsor",
        "adjusted",
        "有效",
        "剔除",
        "过滤",
        "排除",
        "异常",
        "离群",
        "调整",
    }
    text = _flatten_outlier_relevant_text(result).lower()
    if any(marker in text for marker in textual_markers):
        return True
    metrics = result.get("metrics")
    details = result.get("details")
    if isinstance(metrics, dict):
        if metrics.get("has_outlier_like_values") is True or metrics.get("has_outliers") is True:
            return True
        for key in metrics:
            key_text = str(key).lower()
            if key_text in {"used_point_count", "filtered_count", "excluded_count"}:
                return True
    if isinstance(details, dict):
        for key in details:
            key_text = str(key).lower()
            if key_text in {"excluded_rows", "outlier_like_values", "outliers", "adjusted_metrics"}:
                return True
    return False


def _duplicate_json_rows(rows: list) -> bool:
    seen: set[str] = set()
    for row in rows:
        key = json_dumps_stable(row)
        if key in seen:
            return True
        seen.add(key)
    return False


def json_dumps_stable(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return ""


def _flatten_outlier_relevant_text(value: Any, *, key: str | None = None) -> str:
    if key is not None and key.strip().lower() in {"assumption", "assumptions", "input_assumption", "input_assumptions"}:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(
            _flatten_outlier_relevant_text(item, key=str(child_key))
            for child_key, item in value.items()
        )
    if isinstance(value, list):
        return " ".join(_flatten_outlier_relevant_text(item, key=key) for item in value)
    return ""


def _validate_schema_node(value: Any, schema: Any, *, path: str) -> None:
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise AnalysisCodeError(f"analysis result field '{path}' must be an object.")
        for key, child_schema in schema.items():
            if key not in value:
                raise AnalysisCodeError(f"analysis result is missing required field '{path}.{key}'.")
            _validate_schema_node(value[key], child_schema, path=f"{path}.{key}")
        return

    if isinstance(schema, list):
        if not isinstance(value, list):
            raise AnalysisCodeError(f"analysis result field '{path}' must be an array.")
        if schema:
            item_schema = schema[0]
            for index, item in enumerate(value):
                _validate_schema_node(item, item_schema, path=f"{path}[{index}]")
        return

    if isinstance(schema, str):
        _validate_type_name(value, schema, path=path)


def _validate_type_name(value: Any, type_name: str, *, path: str) -> None:
    normalized = type_name.strip().lower()
    if not normalized or normalized == "any":
        return
    if "|" in normalized:
        allowed = [item.strip() for item in normalized.split("|") if item.strip()]
        if any(_matches_type(value, item) for item in allowed):
            return
        raise AnalysisCodeError(f"analysis result field '{path}' must match one of: {', '.join(allowed)}.")
    if not _matches_type(value, normalized):
        raise AnalysisCodeError(f"analysis result field '{path}' must be {normalized}.")


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name in {"str", "string"}:
        return isinstance(value, str)
    if type_name in {"int", "integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name in {"float", "number", "numeric"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name in {"bool", "boolean"}:
        return isinstance(value, bool)
    if type_name in {"dict", "object"}:
        return isinstance(value, dict)
    if type_name in {"list", "array"}:
        return isinstance(value, list)
    if type_name in {"none", "null"}:
        return value is None
    return True


def _code_hash(code: str) -> str:
    return "sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _code_interpreter_work_dir(request_state, code_hash: str):
    if request_state is None or not getattr(request_state, "request_log_dir", None):
        return None
    safe_hash = code_hash.replace(":", "_")
    return Path(request_state.request_log_dir) / "artifacts" / "code_interpreter" / safe_hash


def _analysis_id(evidence_id: str, goal: str, code_hash: str) -> str:
    raw = f"{evidence_id}:{goal}:{code_hash}:code_interpreter"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", goal.strip().lower())[:32].strip("_")
    return f"ana_{slug or 'code_interpreter'}_{digest}"
