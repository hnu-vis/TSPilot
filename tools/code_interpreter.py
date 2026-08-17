"""Computation-only Code Interpreter over grounded evidence."""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.analysis.python_runner import AnalysisCodeError
from core.key_insight.binder import LLMInsightBinder
from core.key_insight.contracts import insight_request_contract_error
from runtime.prompt_locale import prompt_locale_instruction
from sandbox import execute_python_sandbox_v1
from sandbox.analysis_context import build_canonical_analysis_context
from schemas.analysis import AnalysisResult, ComputedInsight, DerivedEvidence
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsightRequest
from tools.base import BaseTool


class CodeInterpreterInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    analysis_goal: str
    code: str | None = None
    constraints: dict = Field(default_factory=dict)
    insight_requests: list[KeyInsightRequest] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            if not data.get("code") and data.get("analysis_code"):
                data["code"] = data["analysis_code"]
            if not isinstance(data.get("constraints"), dict):
                data["constraints"] = {}
        return data

    @model_validator(mode="after")
    def validate_insight_contracts(self):
        errors = [
            error
            for request in self.insight_requests
            if (error := insight_request_contract_error(request, "code_interpreter"))
        ]
        if errors:
            raise ValueError(" ".join(errors))
        keys = [request.insight_key for request in self.insight_requests]
        if len(keys) != len(set(keys)):
            raise ValueError("code_interpreter insight_requests must have unique insight_key values")
        return self


class _GeneratedCode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)


class CodeInterpreterTool(BaseTool):
    """Calculate requested Insight values; semantic binding is a separate LLM stage."""

    def __init__(self, llm=None, binder: LLMInsightBinder | None = None):
        self._llm = llm
        self._binder = binder or LLMInsightBinder(llm)

    async def execute(self, validated_input: CodeInterpreterInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        evidence = _resolve_database_evidence(validated_input.database_evidence, request_state)
        if evidence is None:
            raise ValueError("code_interpreter requires grounded database_evidence")
        rows, points, columns = _analysis_inputs(evidence)
        input_insights = _input_insights(request_state, validated_input.insight_requests)
        context = build_canonical_analysis_context(
            rows=rows,
            points=points,
            columns=columns,
            metadata=evidence.metadata,
            diagnostics=evidence.diagnostics,
        )
        context["input_insights"] = input_insights
        anomaly_context = _authoritative_anomaly_context(request_state, evidence.evidence_id)
        if anomaly_context:
            context["anomaly_context"] = anomaly_context

        response_language = getattr(request_state, "response_language", "en")
        code = str(validated_input.code or "").strip()
        if not code:
            code = await self._generate_code(
                goal=validated_input.analysis_goal,
                requests=validated_input.insight_requests,
                context=context,
                response_language=response_language,
            )
        preflight_error = _preflight_analysis_code(code)
        if preflight_error:
            raise AnalysisCodeError(preflight_error)

        code_hash = _code_hash(code)
        sandbox_output = execute_python_sandbox_v1(
            code=code,
            rows=rows,
            points=points,
            columns=columns,
            metadata=evidence.metadata,
            diagnostics=evidence.diagnostics,
            input_insights=input_insights,
            analysis_context=context,
            timeout_seconds=int(validated_input.constraints.get("timeout_seconds", 5)),
            work_dir=_work_dir(request_state, code_hash),
        )
        analysis_id = _analysis_id(evidence.evidence_id, validated_input.analysis_goal, code_hash)
        derived_evidence, derived_name_map = _materialize_derived_evidence(
            sandbox_output.result.get("derived_evidence", []),
            analysis_id=analysis_id,
            input_evidence_id=evidence.evidence_id,
            input_insights=input_insights,
        )
        computed = _validate_computed_insights(
            sandbox_output.result.get("computed_insights", []),
            requests=validated_input.insight_requests,
            derived_name_map=derived_name_map,
        )
        produced = await self._binder.bind(
            requests=validated_input.insight_requests,
            computed=computed,
            analysis_id=analysis_id,
            analysis_goal=validated_input.analysis_goal,
            input_evidence_id=evidence.evidence_id,
            response_language=response_language,
        )
        result = AnalysisResult(
            analysis_id=analysis_id,
            analysis_goal=validated_input.analysis_goal,
            code_hash=code_hash,
            input_evidence_id=evidence.evidence_id,
            input_row_count=len(rows),
            status="succeeded",
            summary=f"Computed {len(computed)} requested insight value(s).",
            computed_insights=computed,
            derived_evidence=derived_evidence,
            produced_insights=produced,
            diagnostics={
                "runtime_ms": sandbox_output.runtime_ms,
                "sandbox": "subprocess_code_interpreter_v2",
                "executed_code": code,
                "input_columns": columns,
                "canonical_inputs": context.get("schema"),
                "binder": "llm_insight_binder_v1",
            },
        )
        return result.model_dump(mode="json")

    async def _generate_code(
        self,
        *,
        goal: str,
        requests: list[KeyInsightRequest],
        context: dict,
        response_language: str,
    ) -> str:
        if self._llm is None:
            raise AnalysisCodeError("code_interpreter requires code or an LLM code generator")
        system = prompt_locale_instruction(response_language) + (
            "You generate Python for a computation-only Code Interpreter. Return exactly one JSON object with key code. "
            "The sandbox provides df, time, value, time_col, value_col, series, analysis_context, rows, points, columns, "
            "metadata, diagnostics, input_insights, insight_by_key, pd, np, math, and statistics. Do not import or access files, "
            "network, processes, environment variables, or clocks. Assign one dict to result with exactly two keys: "
            "computed_insights and derived_evidence. computed_insights must contain exactly one object per requested insight_key. "
            "Each object contains insight_key, value or items, a non-empty calculation_trace describing formula/inputs as concise text or JSON, and optional "
            "unavailable_reason only when the grounded inputs make the requested calculation impossible. Never invent a placeholder value. "
            "derived_evidence_names. Do not produce statements, names, semantic classes, display roles, Key Insight objects, Data Views, "
            "charts, summaries, repair policies, or final-answer prose. derived_evidence is normally empty; include a named artifact only "
            "when a complete calculated table or series is needed to verify or reuse an Insight. Each artifact contains name, shape, rows "
            "or scalar, and transform_summary. Use exact authoritative anomaly_context points when present; do not run another detector. "
            "When anomaly_context is used, include its exact source_ref in every affected calculation_trace. "
            "Convert numpy/pandas values and timestamps to JSON-native scalars and ISO strings. Never return NaN or Infinity."
        )
        payload = {
            "goal": goal,
            "insight_requests": [request.model_dump(mode="json", exclude_none=True) for request in requests],
            "canonical_inputs": context.get("schema", {}),
            "input_insights": context.get("input_insights", []),
            "anomaly_context": context.get("anomaly_context"),
        }
        messages = [("system", system), ("human", json.dumps(payload, ensure_ascii=False, default=str))]
        try:
            if hasattr(self._llm, "with_structured_output"):
                runnable = self._llm.with_structured_output(_GeneratedCode, method="json_schema")
                response = await asyncio.wait_for(runnable.ainvoke(messages), timeout=30)
                generated = response if isinstance(response, _GeneratedCode) else _GeneratedCode.model_validate(response)
            else:
                response = await asyncio.wait_for(self._llm.ainvoke(messages), timeout=30)
                generated = _GeneratedCode.model_validate_json(_llm_content(response))
        except Exception as exc:
            raise AnalysisCodeError(f"code generation LLM violated its output contract: {exc}") from exc
        return generated.code.strip()


def _validate_computed_insights(raw: Any, *, requests: list[KeyInsightRequest], derived_name_map: dict[str, str]) -> list[ComputedInsight]:
    if not isinstance(raw, list):
        raise AnalysisCodeError("computed_insights must be a list")
    requested_keys = [request.insight_key for request in requests]
    result: list[ComputedInsight] = []
    for item in raw:
        if not isinstance(item, dict):
            raise AnalysisCodeError("each computed insight must be an object")
        payload = dict(item)
        names = payload.pop("derived_evidence_names", [])
        if not isinstance(names, list):
            raise AnalysisCodeError("derived_evidence_names must be a list")
        unknown = [name for name in names if str(name) not in derived_name_map]
        if unknown:
            raise AnalysisCodeError(f"computed insight references unknown derived evidence names: {unknown}")
        payload["derived_evidence_ids"] = [derived_name_map[str(name)] for name in names]
        try:
            result.append(ComputedInsight.model_validate(payload))
        except Exception as exc:
            raise AnalysisCodeError(f"invalid computed insight: {exc}") from exc
    actual_keys = [item.insight_key for item in result]
    if actual_keys != requested_keys:
        raise AnalysisCodeError(
            "computed_insights must preserve request order and exactly match requested keys; "
            f"requested={requested_keys}, computed={actual_keys}"
        )
    return result


def _materialize_derived_evidence(
    raw: Any,
    *,
    analysis_id: str,
    input_evidence_id: str,
    input_insights: list[dict],
) -> tuple[list[DerivedEvidence], dict[str, str]]:
    if not isinstance(raw, list):
        raise AnalysisCodeError("derived_evidence must be a list")
    lineage = [
        f"evidence:{input_evidence_id}",
        *[
            f"insight:{item['insight_id']}"
            for item in input_insights
            if isinstance(item, dict) and item.get("insight_id")
        ],
    ]
    result: list[DerivedEvidence] = []
    name_map: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise AnalysisCodeError("each derived evidence artifact must be an object")
        name = str(item.get("name") or "").strip()
        if not name or name in name_map:
            raise AnalysisCodeError("derived evidence names must be non-empty and unique")
        evidence_id = f"dev_{analysis_id}_{_slug(name)}"
        try:
            artifact = DerivedEvidence.model_validate({**item, "evidence_id": evidence_id, "lineage": lineage})
        except Exception as exc:
            raise AnalysisCodeError(f"invalid derived evidence '{name}': {exc}") from exc
        result.append(artifact)
        name_map[name] = evidence_id
    return result, name_map


def _resolve_database_evidence(value, request_state) -> DatabaseEvidence | None:
    if isinstance(value, DatabaseEvidence):
        if request_state is None:
            return value
        return request_state.database_evidence_artifacts.get(value.evidence_id, value)
    if isinstance(value, dict):
        return DatabaseEvidence.model_validate(value)
    if request_state is None:
        return None
    if value in (None, "", "latest", "latest_database_evidence", "current"):
        return request_state.latest_database_evidence
    ref = str(value).removeprefix("evidence:")
    return request_state.database_evidence_artifacts.get(ref)


def _analysis_inputs(evidence: DatabaseEvidence) -> tuple[list[dict], list[dict], list[str]]:
    data = evidence.data or {}
    rows = [dict(item) for item in data.get("rows", []) if isinstance(item, dict)]
    points = [dict(item) for item in data.get("points", []) if isinstance(item, dict)]
    if not rows:
        rows = [dict(item) for item in points]
    columns = list(evidence.columns or [])
    if not columns and rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    return rows, points, columns


def _input_insights(request_state, requests: list[KeyInsightRequest]) -> list[dict]:
    if request_state is None:
        return []
    needed = {key for request in requests for key in request.derived_from}
    return [
        insight.model_dump(mode="json", exclude_none=True)
        for insight in request_state.insight_set.insights
        if insight.status == "verified" and insight.insight_key in needed
    ]


def _authoritative_anomaly_context(request_state, evidence_id: str) -> dict | None:
    if request_state is None:
        return None
    for anomaly in reversed(list(request_state.anomaly_artifacts.values())):
        diagnostics = anomaly.diagnostics if isinstance(anomaly.diagnostics, dict) else {}
        resolved = str(diagnostics.get("resolved_evidence_id") or diagnostics.get("input_evidence_id") or "").removeprefix("evidence:")
        if resolved == evidence_id:
            return {
                "source_ref": f"anomaly:{anomaly.anomaly_id}",
                "anomaly_points": [dict(item) for item in anomaly.anomaly_points if isinstance(item, dict)],
            }
    return None


def _preflight_analysis_code(code: str | None) -> str | None:
    text = str(code or "")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return f"analysis code is invalid Python: {exc}"
    if not any(isinstance(node, (ast.Assign, ast.AnnAssign)) and _assigns_result(node) for node in ast.walk(tree)):
        return "analysis code must assign a dict to result"
    blocked_nodes = (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Global, ast.Nonlocal)
    blocked_calls = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
    for node in ast.walk(tree):
        if isinstance(node, blocked_nodes):
            return f"analysis code contains blocked syntax: {type(node).__name__}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
            return f"analysis code calls blocked function: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "analysis code cannot access dunder attributes"
    return None


def _assigns_result(node: ast.Assign | ast.AnnAssign) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return any(isinstance(target, ast.Name) and target.id == "result" for target in targets)


def _llm_content(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def _code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _analysis_id(evidence_id: str, goal: str, code_hash: str) -> str:
    digest = hashlib.sha1(f"{evidence_id}:{goal}:{code_hash}".encode()).hexdigest()[:16]
    return f"ana_{digest}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")[:48] or "derived"


def _work_dir(request_state, code_hash: str) -> Path | None:
    log_dir = getattr(request_state, "request_log_dir", None)
    if not log_dir:
        return None
    return Path(log_dir) / "artifacts" / "code_interpreter" / code_hash[:16]
