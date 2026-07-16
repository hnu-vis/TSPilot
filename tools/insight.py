"""Generated-code insight tool."""
from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel, Field

from core.analysis import execute_python_rows_v1
from schemas.analysis import AnalysisResult
from schemas.database import DatabaseEvidence
from tools.base import BaseTool


class InsightInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    analysis_goal: str | None = None
    code_type: str = "python_rows_v1"
    analysis_code: str | None = None
    expected_result_schema: dict | None = None
    requested_fact_types: list[str] = Field(default_factory=list)
    focus: str | None = None
    constraints: dict | None = Field(default_factory=dict)


class InsightTool(BaseTool):
    async def execute(self, validated_input: InsightInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        if request_state is not None:
            database_evidence = _resolve_database_evidence(database_evidence, request_state)
        if database_evidence is None:
            raise ValueError("Insight requires database_evidence or a latest_database_evidence in request state.")
        if validated_input.code_type != "python_rows_v1":
            raise ValueError("Insight only supports code_type='python_rows_v1'.")
        if not validated_input.analysis_code or not validated_input.analysis_code.strip():
            raise ValueError("Insight requires analysis_code for generated-code analysis.")

        rows, points, columns = _analysis_inputs(database_evidence)
        goal = validated_input.analysis_goal or validated_input.focus or "Generated evidence analysis"
        code_hash = _code_hash(validated_input.analysis_code)
        output = execute_python_rows_v1(
            code=validated_input.analysis_code,
            rows=rows,
            points=points,
            columns=columns,
            metadata=database_evidence.metadata,
            diagnostics=database_evidence.diagnostics,
            timeout_seconds=int((validated_input.constraints or {}).get("timeout_seconds", 2)),
        )
        result = AnalysisResult(
            analysis_id=_analysis_id(database_evidence.evidence_id, goal, code_hash),
            analysis_goal=goal,
            code_type="python_rows_v1",
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
                "sandbox": "restricted_python_rows_v1",
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
        return request_state.database_evidence_artifacts.get(database_evidence)
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
    columns = list(evidence.columns or [])
    if not columns and rows:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    return rows, points, columns


def _code_hash(code: str) -> str:
    return "sha256:" + hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _analysis_id(evidence_id: str, goal: str, code_hash: str) -> str:
    raw = f"{evidence_id}:{goal}:{code_hash}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", goal.strip().lower())[:32].strip("_")
    return f"ana_{slug or 'analysis'}_{digest}"

