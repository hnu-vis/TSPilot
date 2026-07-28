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
from tools.base import BaseTool


class CodeInterpreterInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    analysis_goal: str | None = None
    code: str
    expected_result_schema: dict | None = None
    constraints: dict | None = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_code_aliases(cls, data):
        if isinstance(data, dict) and not data.get("code") and data.get("analysis_code"):
            data = dict(data)
            data["code"] = data["analysis_code"]
        return data


class CodeInterpreterTool(BaseTool):
    async def execute(self, validated_input: CodeInterpreterInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = _resolve_database_evidence(database_evidence, request_state)
        if database_evidence is None:
            raise ValueError("code_interpreter requires database_evidence or a latest_database_evidence in request state.")
        if not validated_input.code or not validated_input.code.strip():
            raise ValueError("code_interpreter requires non-empty code.")

        rows, points, columns = _analysis_inputs(database_evidence)
        goal = validated_input.analysis_goal or "Code interpreter analysis"
        code_hash = _code_hash(validated_input.code)
        constraints = validated_input.constraints or {}
        output = execute_python_sandbox_v1(
            code=validated_input.code,
            rows=rows,
            points=points,
            columns=columns,
            metadata=database_evidence.metadata,
            diagnostics=database_evidence.diagnostics,
            timeout_seconds=int(constraints.get("timeout_seconds", 5)),
            work_dir=_code_interpreter_work_dir(request_state, code_hash),
        )
        _validate_expected_result_schema(output.result, validated_input.expected_result_schema or {})
        _validate_outlier_treatment_transparency(output.result)
        result = AnalysisResult(
            analysis_id=_analysis_id(database_evidence.evidence_id, goal, code_hash),
            analysis_goal=goal,
            code_type="code_interpreter_v1",
            code_hash=code_hash,
            input_evidence_id=database_evidence.evidence_id,
            input_row_count=len(rows),
            status="succeeded",
            summary=str(output.result["summary"]),
            result=output.result,
            diagnostics={
                "runtime_ms": output.runtime_ms,
                "expected_result_schema": validated_input.expected_result_schema or {},
                "input_columns": columns,
                "input_points_count": len(points),
                "sandbox": "subprocess_code_interpreter_v1",
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
        return DatabaseEvidence.model_validate(database_evidence)
    return request_state.database_evidence_artifacts.get(database_evidence.evidence_id, database_evidence)


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


def _first_present_key(rows: list[dict], candidates: list[str]) -> str | None:
    for key in candidates:
        if any(key in row for row in rows):
            return key
    return None


def _validate_expected_result_schema(result: dict, schema: dict) -> None:
    if not schema:
        return
    _validate_schema_node(result, schema, path="result")


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
    text = _flatten_text(result).lower()
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
