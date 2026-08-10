"""Subprocess code interpreter tool for grounded data analysis."""
from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from core.analysis.python_runner import AnalysisCodeError
from core.data_fact.contracts import fact_request_contract_error
from sandbox import execute_python_sandbox_v1
from sandbox.analysis_context import build_canonical_analysis_context
from schemas.analysis import AnalysisResult
from schemas.database import DatabaseEvidence
from schemas.data_fact import DataFactRequest, normalize_fact_key
from tools.base import BaseTool, StructuredToolError


TEMPLATE_SUPPORTED_METRICS = {
    "record_count",
    "start_value",
    "end_value",
    "highest_value",
    "lowest_value",
    "max_value",
    "min_value",
    "max_min_difference",
    "difference",
    "start_end_change",
    "change",
}


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

    @model_validator(mode="after")
    def validate_fact_contracts(self):
        errors = [
            error
            for request in self.fact_requests
            if (error := fact_request_contract_error(request, "code_interpreter"))
        ]
        if errors:
            raise ValueError(" ".join(errors))
        return self


class CodeInterpreterTool(BaseTool):
    def __init__(self, llm=None):
        self._llm = llm

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
        analysis_request = validated_input.analysis_request or {}
        repair_contract = validated_input.repair_contract
        if repair_contract is None and isinstance(analysis_request.get("repair_contract"), dict):
            repair_contract = analysis_request.get("repair_contract")
        if repair_contract is None and isinstance(constraints.get("_repair_contract"), dict):
            repair_contract = constraints.get("_repair_contract")
        code_text = str(validated_input.code or "").strip()
        generated_code_preview = None
        executed_code_preview = None
        canonical_context = build_canonical_analysis_context(
            rows=rows,
            points=points,
            columns=columns,
            metadata=database_evidence.metadata,
            diagnostics=database_evidence.diagnostics,
        )
        input_facts = _input_facts_for_requests(request_state, validated_input.fact_requests)
        canonical_context["fact_contracts"] = [
            request.model_dump(mode="json", exclude_none=True)
            for request in validated_input.fact_requests
        ]
        canonical_context["input_facts"] = input_facts
        if not code_text:
            missing_metrics = _missing_template_metrics(
                goal=goal,
                required_outputs=validated_input.required_outputs,
                analysis_request=validated_input.analysis_request,
            )
            if missing_metrics:
                if self._llm is not None:
                    try:
                        code_text = await self._generate_analysis_code(
                            goal=goal,
                            required_outputs=validated_input.required_outputs,
                            required_metrics=_requested_metric_labels(
                                goal=goal,
                                required_outputs=validated_input.required_outputs,
                                analysis_request=validated_input.analysis_request,
                            ),
                            canonical_context=canonical_context,
                            fact_requests=validated_input.fact_requests,
                            analysis_request=analysis_request,
                            expected_result_schema=validated_input.expected_result_schema or {},
                            repair_contract=repair_contract,
                        )
                        generated_code_preview = _failed_code_summary(code_text)
                    except (AnalysisCodeError, asyncio.TimeoutError):
                        raise _structured_code_required_error(
                            goal=goal,
                            database_evidence=database_evidence,
                            required_metrics=_requested_metric_labels(
                                goal=goal,
                                required_outputs=validated_input.required_outputs,
                                analysis_request=validated_input.analysis_request,
                            ),
                            missing_metrics=missing_metrics,
                        )
                else:
                    raise _structured_code_required_error(
                        goal=goal,
                        database_evidence=database_evidence,
                        required_metrics=_requested_metric_labels(
                            goal=goal,
                            required_outputs=validated_input.required_outputs,
                            analysis_request=validated_input.analysis_request,
                        ),
                        missing_metrics=missing_metrics,
                    )
            if not code_text:
                output = _execute_analysis_request(
                    rows=rows,
                    points=points,
                    columns=columns,
                    goal=goal,
                    required_outputs=validated_input.required_outputs,
                    analysis_request=validated_input.analysis_request,
                    constraints=constraints,
                    fact_requests=validated_input.fact_requests,
                    input_facts=input_facts,
                )
                code_type = "analysis_request_v1"
                code_hash = _code_hash(json.dumps(output.get("diagnostics", {}), ensure_ascii=False, sort_keys=True, default=str))
                runtime_ms = 0
        if code_text:
            code_hash = _code_hash(code_text)
            requires_numeric_series = _analysis_requires_numeric_series(goal, validated_input.expected_result_schema or {})
            if requires_numeric_series and rows and not points:
                raise AnalysisCodeError(
                    "code_interpreter analysis requires numeric time-series values, but the selected evidence "
                    "does not expose a usable timestamp/value pair. Query evidence with a numeric value column first."
                )
            sandbox_output, final_code_text = await self._execute_code_once(
                code=code_text,
                goal=goal,
                database_evidence=database_evidence,
                rows=rows,
                points=points,
                columns=columns,
                canonical_context=canonical_context,
                constraints=constraints,
                request_state=request_state,
                required_outputs=validated_input.required_outputs,
                analysis_request=validated_input.analysis_request,
                expected_result_schema=validated_input.expected_result_schema or {},
                input_facts=input_facts,
            )
            code_hash = _code_hash(final_code_text)
            executed_code_preview = _failed_code_summary(final_code_text)
            generated_code_preview = _failed_code_summary(final_code_text) if generated_code_preview else None
            output = {"result": sandbox_output.result, "diagnostics": {"runtime_ms": sandbox_output.runtime_ms}}
            code_type = "code_interpreter_v1"
            runtime_ms = sandbox_output.runtime_ms
        elif not validated_input.code and missing_metrics:
            # Kept for static analyzers; all branches above either set code_text,
            # execute the template path, or raise a structured code-required error.
            raise _structured_code_required_error(
                goal=goal,
                database_evidence=database_evidence,
                required_metrics=_requested_metric_labels(
                    goal=goal,
                    required_outputs=validated_input.required_outputs,
                    analysis_request=validated_input.analysis_request,
                ),
                missing_metrics=missing_metrics,
            )
        else:
            # Template path has already populated output/code_type/runtime_ms.
            pass
        requires_numeric_series = _analysis_requires_numeric_series(goal, validated_input.expected_result_schema or {})
        result_payload = output["result"]
        try:
            _validate_expected_result_schema(result_payload, validated_input.expected_result_schema or {})
            _validate_outlier_treatment_transparency(result_payload)
            _validate_result_has_numeric_analysis(result_payload, requires_numeric_series=requires_numeric_series, input_rows=len(rows))
            fact_binding = _validate_fact_output_contract(
                result_payload,
                validated_input.fact_requests,
                input_row_count=len(rows),
                input_facts=input_facts,
            )
        except AnalysisCodeError as exc:
            raise _structured_analysis_validation_error(
                exc,
                goal=goal,
                database_evidence=database_evidence,
                result_payload=result_payload,
                failed_code=final_code_text if code_text else None,
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
                "executed_code": final_code_text if code_text else None,
                "generated_code_preview": generated_code_preview,
                "executed_code_preview": executed_code_preview,
                "execution_attempts": 1 if code_text else 0,
                "fact_binding": fact_binding,
                "canonical_inputs": canonical_context.get("schema"),
                **output.get("diagnostics", {}),
            },
        )
        return result.model_dump(mode="json")

    async def _execute_code_once(
        self,
        *,
        code: str,
        goal: str,
        database_evidence: DatabaseEvidence,
        rows: list[dict],
        points: list[dict],
        columns: list[str],
        canonical_context: dict,
        constraints: dict,
        request_state,
        required_outputs: list[str],
        analysis_request: dict | None,
        expected_result_schema: dict,
        input_facts: list[dict],
    ):
        preflight_error = _preflight_analysis_code(code, canonical_context)
        if preflight_error is None:
            try:
                sandbox_output = execute_python_sandbox_v1(
                    code=code,
                    rows=rows,
                    points=points,
                    columns=columns,
                    metadata=database_evidence.metadata,
                    diagnostics=database_evidence.diagnostics,
                    input_facts=input_facts,
                    timeout_seconds=int(constraints.get("timeout_seconds", 5)),
                    work_dir=_code_interpreter_work_dir(request_state, _code_hash(code)),
                )
                return sandbox_output, code
            except AnalysisCodeError as exc:
                execution_error = exc
        else:
            execution_error = AnalysisCodeError(preflight_error)
        raise _structured_code_execution_error(
            execution_error,
            goal=goal,
            database_evidence=database_evidence,
            columns=columns,
            required_outputs=required_outputs,
            analysis_request=analysis_request,
            expected_result_schema=expected_result_schema,
            failed_code=code,
            canonical_context=canonical_context,
        )

    async def _generate_analysis_code(
        self,
        *,
        goal: str,
        required_outputs: list[str],
        required_metrics: list[str],
        canonical_context: dict,
        fact_requests: list[DataFactRequest],
        analysis_request: dict,
        expected_result_schema: dict,
        repair_contract: dict | None,
    ) -> str:
        return await self._invoke_code_llm(
            {
                "mode": "repair" if repair_contract else "generate",
                "goal": goal,
                "required_outputs": required_outputs,
                "required_metrics": required_metrics,
                "analysis_request": analysis_request,
                "expected_result_schema": expected_result_schema,
                "repair_contract": repair_contract,
                "canonical_inputs": canonical_context.get("schema", {}),
                "fact_contracts": [request.model_dump(mode="json", exclude_none=True) for request in fact_requests],
                "input_facts": canonical_context.get("input_facts", []),
            }
        )

    async def _invoke_code_llm(self, payload: dict) -> str:
        if self._llm is None:
            raise AnalysisCodeError("code_interpreter has no LLM available for code generation.")
        system = (
            "You generate Python code for TSPilot code_interpreter. Return exactly one JSON object "
            "with key code and no markdown. The sandbox already provides df, time, value, time_col, "
            "value_col, series, analysis_context, data, rows, points, columns, metadata, diagnostics, input_facts, fact_by_key, "
            "math, statistics, pd, and np. Prefer value/time/df/series. For multi-series input, inspect "
            "analysis_context['schema']['dimension_cols'] and group by an available dimension; never assume a field, metric, or series column name. "
            "Do not use pd.np or ellipsis placeholders. Do not invent field names. The code must assign result as one dict containing the stable contract fields "
            "summary, metrics, details, and facts. summary must be a non-empty string. metrics must be a dict/object and may be empty "
            "when the requested answer is primarily table/list/detail output. details must be a dict/object. Prefer details "
            "for row records, top-k tables, time/value pairs, intermediate arrays, and calculation traces. All metric/detail "
            "values must be JSON-serializable plain Python scalars, lists, or dicts. facts must be a list with one object per "
            "satisfied fact_contract, preserving its fact_key, name, fact_type, and derived_from. Facts calculated directly from database rows may have empty derived_from; "
            "derived_from is reserved for verified parent Facts. Each fact must include value, "
            "statement, and calculation_trace. Use input_facts or fact_by_key for dependencies; never invent a dependency value. "
            "Treat analysis_request, expected_result_schema, and repair_contract as authoritative output requirements. When a "
            "repair_contract lists required_details_fields, result.details must contain every listed field with the requested types. "
            "When the goal or analysis_request performs outlier/anomaly treatment, result.details must include outlier_rule, "
            "threshold_or_formula, rationale, excluded_rows as a row list, raw_metrics, and adjusted_metrics."
        )
        messages = [
            ("system", system),
            ("human", json.dumps(payload, ensure_ascii=False, default=str)),
        ]
        response = await asyncio.wait_for(self._llm.ainvoke(messages), timeout=30)
        content = _llm_content(response)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AnalysisCodeError(f"code generation LLM returned invalid JSON: {exc}") from exc
        code = parsed.get("code") if isinstance(parsed, dict) else None
        if not isinstance(code, str) or not code.strip():
            raise AnalysisCodeError("code generation LLM returned empty code.")
        return code.strip()


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


def _input_facts_for_requests(request_state, requests: list[DataFactRequest]) -> list[dict]:
    if request_state is None:
        return []
    required_keys = {dependency for request in requests for dependency in request.derived_from}
    if not required_keys:
        return []
    result: list[dict] = []
    for fact in request_state.fact_set.facts:
        if fact.status != "verified" or fact.fact_key not in required_keys:
            continue
        result.append(
            fact.model_dump(
                mode="json",
                include={
                    "fact_id",
                    "fact_key",
                    "name",
                    "fact_type",
                    "value",
                    "unit",
                    "subject",
                    "dimensions",
                    "time_range",
                    "derived_from",
                    "calculation_trace",
                },
            )
        )
    return result


def _validate_fact_output_contract(
    result: dict,
    requests: list[DataFactRequest],
    *,
    input_row_count: int = 0,
    input_facts: list[dict] | None = None,
) -> dict:
    """Bind candidate facts without converting coverage gaps into code failures."""

    diagnostics = {"bound": [], "missing": [], "rejected": []}
    if not requests:
        return diagnostics
    facts = result.get("facts")
    if facts is None:
        facts = []
    if not isinstance(facts, list):
        diagnostics["rejected"].append({"reason": "result.facts must be a list"})
        facts = []
    requests_by_alias: dict[str, DataFactRequest] = {}
    for request in requests:
        requests_by_alias[request.fact_key] = request
        requests_by_alias[normalize_fact_key(request.name)] = request
    verified_input_keys = {
        normalize_fact_key(fact.get("fact_key") or fact.get("fact_id") or fact.get("name") or "")
        for fact in (input_facts or [])
        if isinstance(fact, dict) and fact.get("status", "verified") == "verified"
    }
    candidate_keys = {
        request.fact_key
        for raw in facts
        if isinstance(raw, dict)
        and (request := requests_by_alias.get(normalize_fact_key(raw.get("fact_key") or raw.get("name") or "")))
    }
    bound_facts: list[dict] = []
    seen: set[str] = set()
    for raw in facts:
        if not isinstance(raw, dict):
            diagnostics["rejected"].append({"reason": "candidate fact must be an object"})
            continue
        candidate = dict(raw)
        candidate_key = normalize_fact_key(candidate.get("fact_key") or candidate.get("name") or "")
        request = requests_by_alias.get(candidate_key)
        if request is None:
            diagnostics["rejected"].append({"fact_key": candidate_key, "reason": "unrequested candidate fact"})
            continue
        if request.fact_key in seen:
            diagnostics["rejected"].append({"fact_key": request.fact_key, "reason": "duplicate candidate fact"})
            continue
        if candidate.get("value") is None or not str(candidate.get("statement") or "").strip():
            diagnostics["rejected"].append(
                {"fact_key": request.fact_key, "reason": "candidate fact requires value and statement"}
            )
            continue
        dependencies = [
            normalize_fact_key(item)
            for item in (candidate.get("derived_from") if "derived_from" in candidate else request.derived_from) or []
            if item
        ]
        quality_flags = list(candidate.get("quality_flags") or [])
        missing_parents = [
            key
            for key in dependencies
            if key not in verified_input_keys and key not in candidate_keys
        ]
        if request.fact_key in dependencies:
            missing_parents.append(request.fact_key)
        if missing_parents and "unverified_dependencies" not in quality_flags:
            quality_flags.append("unverified_dependencies")
        if input_row_count <= 0 and not dependencies and "ungrounded_candidate" not in quality_flags:
            quality_flags.append("ungrounded_candidate")
        if (
            not isinstance(candidate.get("calculation_trace"), dict)
            or not candidate.get("calculation_trace")
        ) and "missing_calculation_trace" not in quality_flags:
            quality_flags.append("missing_calculation_trace")
        candidate.update(
            {
                "fact_key": request.fact_key,
                "name": request.name,
                "fact_type": request.fact_type,
                "derived_from": dependencies,
                "quality_flags": quality_flags,
                "status": "partial" if quality_flags else candidate.get("status", "verified"),
            }
        )
        seen.add(request.fact_key)
        bound_facts.append(candidate)
        diagnostics["bound"].append(request.fact_key)
    result["facts"] = bound_facts
    diagnostics["missing"] = [request.fact_key for request in requests if request.fact_key not in seen]
    return diagnostics


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


def _requested_metric_labels(*, goal: str, required_outputs: list[str], analysis_request: dict | None) -> list[str]:
    labels = _string_list(required_outputs)
    if isinstance(analysis_request, dict):
        labels.extend(_string_list(analysis_request.get("required_outputs")))
    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    if deduped:
        return deduped
    text_labels = []
    if goal:
        text_labels.append(goal)
    if isinstance(analysis_request, dict) and analysis_request.get("goal"):
        text_labels.append(str(analysis_request.get("goal")))
    return text_labels


def _missing_template_metrics(*, goal: str, required_outputs: list[str], analysis_request: dict | None) -> list[str]:
    labels = _requested_metric_labels(
        goal=goal,
        required_outputs=required_outputs,
        analysis_request=analysis_request,
    )
    missing: list[str] = []
    for label in labels:
        coverage = _template_metric_coverage(label)
        for metric in coverage["missing"]:
            if metric not in missing:
                missing.append(metric)
    return _sort_missing_metrics(missing)


def _sort_missing_metrics(metrics: list[str]) -> list[str]:
    priority = {
        "total_return": 0,
        "volatility": 1,
        "max_drawdown": 2,
    }
    return sorted(metrics, key=lambda item: (priority.get(item, 100), item))


def _template_metric_coverage(label: str) -> dict[str, list[str]]:
    text = str(label or "").strip().lower()
    if not text:
        return {"covered": [], "missing": []}
    compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    missing = _unsupported_metric_names(text, compact)
    if missing:
        return {"covered": [], "missing": missing}
    covered = _supported_metric_names(text, compact)
    if covered:
        return {"covered": covered, "missing": []}
    return {"covered": [], "missing": [str(label).strip()]}


def _unsupported_metric_names(text: str, compact: str) -> list[str]:
    missing: list[str] = []
    metric_markers = (
        ("total_return", ("totalreturn", "cumulativereturn", "returnrate", "收益率", "总收益", "累计收益")),
        ("volatility", ("volatility", "stdreturn", "波动率", "波动")),
        ("max_drawdown", ("maxdrawdown", "maximumdrawdown", "drawdown", "最大回撤", "回撤")),
    )
    for name, markers in metric_markers:
        if any(marker in compact or marker in text for marker in markers) and name not in missing:
            missing.append(name)
    if "returns" in text and "total_return" not in missing:
        missing.append("total_return")
    return missing


def _supported_metric_names(text: str, compact: str) -> list[str]:
    exact = {
        "recordcount": "record_count",
        "rowcount": "record_count",
        "count": "record_count",
        "startvalue": "start_value",
        "firstvalue": "start_value",
        "endvalue": "end_value",
        "lastvalue": "end_value",
        "highestvalue": "highest_value",
        "maxvalue": "max_value",
        "maximumvalue": "max_value",
        "lowestvalue": "lowest_value",
        "minvalue": "min_value",
        "minimumvalue": "min_value",
        "maxmindifference": "max_min_difference",
        "difference": "difference",
        "startendchange": "start_end_change",
        "change": "change",
        "最大值": "max_value",
        "最高值": "highest_value",
        "最小值": "min_value",
        "最低值": "lowest_value",
        "最大值和最小值的差异": "max_min_difference",
        "最大最小差异": "max_min_difference",
        "差异": "difference",
        "差值": "difference",
        "起始值": "start_value",
        "开始值": "start_value",
        "结束值": "end_value",
        "最终值": "end_value",
        "记录数": "record_count",
        "条数": "record_count",
    }
    covered: list[str] = []
    for key, metric in exact.items():
        normalized_key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", key.lower())
        if compact == normalized_key or normalized_key in compact:
            if metric not in covered:
                covered.append(metric)
    if ("最大" in text or "最高" in text or "max" in text or "maximum" in text) and "max_drawdown" not in compact:
        metric = "max_value"
        if metric not in covered:
            covered.append(metric)
    if "最小" in text or "最低" in text or "min" in text or "minimum" in text:
        metric = "min_value"
        if metric not in covered:
            covered.append(metric)
    if any(marker in text for marker in ("差异", "差值", "difference")):
        metric = "max_min_difference" if (("最大" in text and "最小" in text) or "max" in text and "min" in text) else "difference"
        if metric not in covered:
            covered.append(metric)
    return [metric for metric in covered if metric in TEMPLATE_SUPPORTED_METRICS]


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
    fact_requests: list[DataFactRequest],
    input_facts: list[dict],
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
    all_metrics = {
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
    requested_outputs = _string_list(required_outputs or (analysis_request or {}).get("required_outputs"))
    selected_metric_keys = _requested_metric_keys(requested_outputs, all_metrics)
    goal_metric_keys = _metric_keys_implied_by_text(goal, all_metrics)
    if goal_metric_keys:
        selected_metric_keys = [
            key for key in selected_metric_keys if key in goal_metric_keys
        ] or goal_metric_keys
    metrics = {key: all_metrics[key] for key in selected_metric_keys}
    if not metrics and not requested_outputs:
        metrics = dict(all_metrics)
    details = _selected_analysis_details(
        selected_metric_keys=list(metrics),
        requested_outputs=requested_outputs,
        first=first,
        last=last,
        max_point=max_point,
        min_point=min_point,
        columns=columns,
    )
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
                "raw_metrics": {key: all_metrics[key] for key in metrics},
                "adjusted_metrics": dict(metrics),
            }
        )
    summary = _analysis_request_summary(goal, metrics, details)
    facts = _template_facts_from_metrics(fact_requests, metrics, input_facts)
    return {
        "result": {
            "summary": summary,
            "metrics": metrics,
            "details": details,
            "facts": facts,
        },
        "diagnostics": {
            "analysis_request": analysis_request or {},
            "template": "canonical_timeseries_metrics_v1",
            "constraints": constraints,
            "input_point_count": len(sorted_points),
        },
    }


def _template_facts_from_metrics(
    requests: list[DataFactRequest],
    metrics: dict,
    input_facts: list[dict],
) -> list[dict]:
    input_keys = {
        normalize_fact_key(fact.get("fact_key") or fact.get("fact_id") or fact.get("name") or "")
        for fact in input_facts
    }
    facts: list[dict] = []
    for request in requests:
        metric_key = str(request.requirements.get("metric_key") or request.name)
        if metric_key not in metrics:
            continue
        if any(dependency not in input_keys for dependency in request.derived_from):
            continue
        value = metrics[metric_key]
        facts.append(
            {
                "fact_key": request.fact_key,
                "name": request.name,
                "fact_type": request.fact_type,
                "statement": f"{request.name} is {value}.",
                "value": value,
                "subject": request.subject,
                "dimensions": request.dimensions,
                "time_range": request.time_range,
                "derived_from": request.derived_from,
                "calculation_trace": {
                    "template": "canonical_timeseries_metrics_v1",
                    "metric_key": metric_key,
                },
            }
        )
    return facts


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


def _structured_code_required_error(
    *,
    goal: str,
    database_evidence: DatabaseEvidence,
    required_metrics: list[str],
    missing_metrics: list[str],
) -> StructuredToolError:
    repair_contract = {
        "mode": "generated_code_required",
        "input_evidence": database_evidence.evidence_id,
        "analysis_goal": goal,
        "required_metrics": required_metrics,
        "missing_metrics": missing_metrics,
        "instruction": (
            "Call code_interpreter again with Python code. The code must compute the missing metrics "
            "from rows/points and return result = {'summary': str, 'metrics': dict, 'details': dict}."
        ),
        "expected_result_shape": {
            "summary": "string",
            "metrics": {metric: "number_or_null" for metric in missing_metrics},
            "details": "object",
        },
    }
    message = (
        "analysis_request_v1 cannot cover requested metrics without generated code: "
        + ", ".join(missing_metrics)
    )
    validation_failure = {
        "scope": "tool_input",
        "capability": "analysis",
        "tool": "code_interpreter",
        "error_code": "code_required_for_metrics",
        "message": message,
        "required_contract": {
            "template_supported_metrics": sorted(TEMPLATE_SUPPORTED_METRICS),
            "required_metrics": required_metrics,
            "missing_metrics": missing_metrics,
        },
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
        error_type="code_required_for_metrics",
        retryable=True,
        recommended_next_action="code_interpreter",
        diagnostics={
            "template_supported_metrics": sorted(TEMPLATE_SUPPORTED_METRICS),
            "required_metrics": required_metrics,
            "missing_metrics": missing_metrics,
            "repair_contract": repair_contract,
        },
        validation_failure=validation_failure,
    )


def _structured_code_execution_error(
    exc: AnalysisCodeError,
    *,
    goal: str,
    database_evidence: DatabaseEvidence,
    columns: list[str],
    required_outputs: list[str],
    analysis_request: dict | None,
    expected_result_schema: dict,
    failed_code: str | None = None,
    canonical_context: dict | None = None,
) -> StructuredToolError:
    required_metrics = _requested_metric_labels(
        goal=goal,
        required_outputs=required_outputs,
        analysis_request=analysis_request,
    )
    if canonical_context is None:
        rows, points, context_columns = _analysis_inputs(database_evidence)
        canonical_context = build_canonical_analysis_context(
            rows=rows,
            points=points,
            columns=columns or context_columns,
            metadata=database_evidence.metadata,
            diagnostics=database_evidence.diagnostics,
        )
    repair_contract = {
        "mode": "code_execution_repair",
        "input_evidence": database_evidence.evidence_id,
        "analysis_goal": goal,
        "required_metrics": required_metrics,
        "expected_result_shape": expected_result_schema
        or {"summary": "string", "metrics": "object", "details": "object"},
        "available_inputs": {
            "rows": "list[dict]",
            "points": "list[dict]",
            "columns": columns,
            "data": "dict with rows, points, and series",
            "metadata": "dict",
            "diagnostics": "dict",
            "canonical": "df, time, value, time_col, value_col, series, analysis_context; analysis_context.schema.dimension_cols lists grouping dimensions",
        },
        "canonical_inputs": canonical_context.get("schema") if isinstance(canonical_context, dict) else {},
        "failed_code": str(failed_code or ""),
        "failed_code_summary": _failed_code_summary(failed_code),
        "error_classification": _classify_analysis_code_error(str(exc)),
        "instruction": (
            "Call code_interpreter again with corrected Python code. The sandbox provides variables "
            "df, data, time, value, time_col, value_col, series, analysis_context, rows, points, columns, "
            "metadata, and diagnostics. Prefer df and inspect analysis_context['schema']['dimension_cols'] "
            "before grouping multi-series evidence; do not assume a field/metric/series column name. Use pandas frequency aliases compatible with current "
            "pandas, for example 'h' for hourly grouping. The code must assign "
            "a dict to result with non-empty result['summary'], result['metrics'], and result['details']."
        ),
    }
    raw_message = str(exc)
    message = raw_message if raw_message.startswith("analysis_code sandbox failed:") else f"analysis_code sandbox failed: {raw_message}"
    validation_failure = {
        "scope": "tool_input",
        "capability": "analysis",
        "tool": "code_interpreter",
        "error_code": "analysis_code_execution_failed",
        "message": message,
        "required_contract": {
            "available_inputs": repair_contract["available_inputs"],
            "required_metrics": required_metrics,
            "expected_result_shape": repair_contract["expected_result_shape"],
        },
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
        error_type="analysis_code_execution_failed",
        retryable=True,
        recommended_next_action="code_interpreter",
        diagnostics={
            "available_inputs": repair_contract["available_inputs"],
            "required_metrics": required_metrics,
            "repair_contract": repair_contract,
        },
        validation_failure=validation_failure,
    )


def _preflight_analysis_code(code: str | None, canonical_context: dict | None = None) -> str | None:
    text = str(code or "")
    if not text.strip():
        return "analysis_code cannot be empty."
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return f"analysis_code syntax error before sandbox execution: {exc.msg} at line {exc.lineno}."
    if not _assigns_result(tree):
        return "analysis_code must assign a dict named result with summary, metrics, and details."
    if re.search(r"freq\s*=\s*['\"]H['\"]", text) or re.search(r"resample\s*\(\s*['\"]H['\"]", text):
        return "analysis_code uses pandas hourly frequency 'H'; use lowercase 'h' for current pandas compatibility."
    missing_names = sorted(_missing_runtime_names(tree, canonical_context or {}))
    if missing_names:
        return (
            "analysis_code references unavailable runtime variables: "
            + ", ".join(missing_names[:8])
            + ". Use provided variables df, time, value, time_col, value_col, series, rows, points, columns, metadata, diagnostics, input_facts, fact_by_key, or define variables before use."
        )
    return None


def _assigns_result(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "result":
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "result":
            return True
    return False


def _missing_runtime_names(tree: ast.AST, canonical_context: dict) -> set[str]:
    provided = {
        "rows",
        "points",
        "columns",
        "database_evidence",
        "data",
        "metadata",
        "diagnostics",
        "math",
        "statistics",
        "mean",
        "median",
        "stdev",
        "pstdev",
        "sqrt",
        "analysis_context",
        "series",
        "df",
        "time",
        "value",
        "time_col",
        "value_col",
        "input_facts",
        "fact_by_key",
        "pd",
        "np",
    }
    builtins = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "globals",
        "hasattr",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "repr",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "BaseException",
        "Exception",
        "ArithmeticError",
        "LookupError",
        "IndexError",
        "KeyError",
        "NameError",
        "TypeError",
        "ValueError",
        "ZeroDivisionError",
    }
    assigned: set[str] = set()
    imported: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned.update(_argument_names(node.args))
        elif isinstance(node, ast.Lambda):
            assigned.update(_argument_names(node.args))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.comprehension):
            assigned.update(_store_names(node.target))
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
    available = provided | builtins | assigned | imported
    schema = canonical_context.get("schema") if isinstance(canonical_context.get("schema"), dict) else {}
    if schema.get("value_col"):
        available.add(str(schema["value_col"]))
    if schema.get("time_col"):
        available.add(str(schema["time_col"]))
    return {name for name in loaded if name not in available and not name.startswith("__")}


def _argument_names(args: ast.arguments) -> set[str]:
    names = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _store_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
    return names


def _classify_analysis_code_error(message: str) -> dict:
    lowered = str(message or "").lower()
    if "syntax" in lowered:
        code = "syntax_error"
    elif "not defined" in lowered or "unavailable runtime variables" in lowered:
        code = "undefined_variable"
    elif "keyerror" in lowered or "list indices must be integers" in lowered or "field" in lowered:
        code = "input_shape_error"
    elif "frequency" in lowered or "freq" in lowered:
        code = "pandas_frequency_error"
    elif "result" in lowered:
        code = "result_contract_error"
    else:
        code = "execution_error"
    return {"code": code, "message": str(message or "")[:800]}


def _failed_code_summary(code: str | None) -> dict:
    text = str(code or "")
    if not text.strip():
        return {}
    lines = text.splitlines()
    return {
        "line_count": len(lines),
        "char_count": len(text),
        "preview": "\n".join(lines[:24])[:2000],
    }


def _llm_content(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _structured_analysis_validation_error(
    exc: AnalysisCodeError,
    *,
    goal: str,
    database_evidence: DatabaseEvidence,
    result_payload: dict,
    failed_code: str | None = None,
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
            "failed_code": str(failed_code or ""),
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
                "failed_code": str(failed_code or ""),
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
    if not metrics:
        return goal
    parts = []
    for key, value in metrics.items():
        if key in {"max_value", "highest_value"}:
            parts.append(f"{key}={value} at {details.get('max_time')}")
        elif key in {"min_value", "lowest_value"}:
            parts.append(f"{key}={value} at {details.get('min_time')}")
        elif key == "start_value":
            parts.append(f"{key}={value} at {details.get('start_time')}")
        elif key == "end_value":
            parts.append(f"{key}={value} at {details.get('end_time')}")
        else:
            parts.append(f"{key}={value}")
    return f"{goal}: " + ", ".join(parts) + "."


def _requested_metric_keys(requested_outputs: list[str], available_metrics: dict) -> list[str]:
    if not requested_outputs:
        return []
    selected: list[str] = []
    for output in requested_outputs:
        text = str(output or "").strip().lower()
        compact = text.replace("_", "").replace("-", "").replace(" ", "")
        candidates: list[str] = []
        if output in available_metrics:
            candidates.append(output)
        if "difference" in text or "差" in text or "差异" in text:
            candidates.extend(["max_min_difference", "difference"])
        if "highest" in text or "maximum" in text or "max" in text or "最大" in text or "最高" in text:
            if "time" in text or "时间" in text:
                continue
            candidates.extend(["max_value", "highest_value"])
        if "lowest" in text or "minimum" in text or "min" in text or "最小" in text or "最低" in text:
            if "time" in text or "时间" in text:
                continue
            candidates.extend(["min_value", "lowest_value"])
        if "start" in text or "first" in text or "起始" in text or "开始" in text:
            candidates.append("start_value")
        if "end" in text or "last" in text or "结束" in text or "最终" in text or "最后" in text:
            candidates.append("end_value")
        if "count" in text or "record" in text or "数量" in text or "条数" in text:
            candidates.append("record_count")
        for key in available_metrics:
            key_compact = key.replace("_", "")
            if compact and (compact == key_compact or compact in key_compact or key_compact in compact):
                candidates.append(key)
        for key in candidates:
            if key in available_metrics and key not in selected:
                selected.append(key)
                break
    return selected


def _metric_keys_implied_by_text(text: str, available_metrics: dict) -> list[str]:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return []
    result: list[str] = []
    asks_difference = any(marker in lowered for marker in ("difference", "差异", "差值", "相差", "之差"))
    asks_change = any(marker in lowered for marker in ("change", "变化", "涨跌", "增减", "percentage", "百分比"))
    if asks_difference:
        for candidates in (("max_value", "highest_value"), ("min_value", "lowest_value"), ("max_min_difference", "difference")):
            for key in candidates:
                if key in available_metrics and key not in result:
                    result.append(key)
                    break
        return result
    for markers, candidates in (
        (("highest", "maximum", "max", "最大", "最高"), ("max_value", "highest_value")),
        (("lowest", "minimum", "min", "最小", "最低"), ("min_value", "lowest_value")),
        (("start", "first", "起始", "开始", "首个"), ("start_value",)),
        (("end", "last", "结束", "最终", "最后", "最晚"), ("end_value",)),
        (("count", "record", "数量", "条数", "多少条"), ("record_count",)),
    ):
        if any(marker in lowered for marker in markers):
            for key in candidates:
                if key in available_metrics and key not in result:
                    result.append(key)
                    break
    if asks_change:
        for key in ("percentage_change", "start_end_change", "change"):
            if key in available_metrics and key not in result:
                result.append(key)
                break
    return result


def _selected_analysis_details(
    *,
    selected_metric_keys: list[str],
    requested_outputs: list[str],
    first: dict,
    last: dict,
    max_point: dict,
    min_point: dict,
    columns: list[str],
) -> dict:
    if not selected_metric_keys and not requested_outputs:
        return {
            "start_time": first.get("timestamp"),
            "end_time": last.get("timestamp"),
            "max_time": max_point.get("timestamp"),
            "min_time": min_point.get("timestamp"),
            "first_point": first,
            "last_point": last,
            "max_point": max_point,
            "min_point": min_point,
            "columns": columns,
            "requested_outputs": requested_outputs,
        }
    details: dict = {
        "columns": columns,
        "requested_outputs": requested_outputs,
    }
    selected = set(selected_metric_keys)
    if selected & {"max_value", "highest_value", "max_min_difference", "difference"}:
        details["max_time"] = max_point.get("timestamp")
        details["max_point"] = max_point
    if selected & {"min_value", "lowest_value", "max_min_difference", "difference"}:
        details["min_time"] = min_point.get("timestamp")
        details["min_point"] = min_point
    if selected & {"start_value", "start_end_change", "change"}:
        details["start_time"] = first.get("timestamp")
        details["first_point"] = first
    if selected & {"end_value", "start_end_change", "change"}:
        details["end_time"] = last.get("timestamp")
        details["last_point"] = last
    return details


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
    metrics = result.get("metrics") if isinstance(result, dict) else {}
    details = result.get("details") if isinstance(result, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(details, dict):
        details = {}

    meaningful_values = [
        value for key, value in metrics.items()
        if str(key).lower() not in {"row_count", "count", "n", "series_length", "return_count"}
    ]
    if meaningful_values and any(value is not None for value in meaningful_values):
        return

    for key in ("raw_metrics", "adjusted_metrics"):
        nested_metrics = details.get(key)
        if isinstance(nested_metrics, dict) and any(value is not None for value in nested_metrics.values()):
            return
    if _has_non_empty_detail_output(details):
        return

    raise AnalysisCodeError(
        "code_interpreter numeric analysis produced no non-empty computed output; "
        "use evidence with numeric values and populate result.metrics or result.details with the requested computation."
    )


def _has_non_empty_detail_output(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, bytes)):
        return bool(value.strip())
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, dict):
        return any(_has_non_empty_detail_output(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_non_empty_detail_output(item) for item in value)
    return True


def _result_indicates_outlier_treatment(result: dict) -> bool:
    if not isinstance(result, dict):
        return False
    textual_markers = {
        "outlier",
        "anomaly",
        "anomalous",
        "excluded",
        "exclude",
        "winsor",
        "adjusted",
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
