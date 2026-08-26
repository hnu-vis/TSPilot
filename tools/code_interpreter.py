"""Computation-only Code Interpreter over grounded evidence."""
from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.analysis.code_policy import AnalysisPolicyError, prepare_analysis_code
from core.analysis.python_runner import AnalysisCodeError
from core.artifact_sources import (
    database_evidence_for_sources,
    primary_analysis_input,
    resolve_artifact_sources,
    source_prompt_manifest,
)
from core.key_insight.binder import LLMInsightBinder
from core.key_insight.contracts import insight_request_contract_error
from core.timeseries.evidence_resolution import resolve_database_evidence
from runtime.llm_trace import llm_trace_span
from runtime.prompt_locale import prompt_locale_instruction
from runtime.timeout_policy import load_timeout_policy
from sandbox import execute_python_sandbox_v1
from sandbox.analysis_context import build_canonical_analysis_context
from schemas.analysis import AnalysisResult, ComputedInsight, DerivedEvidence
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsightRequest
from tools.base import BaseTool


class CodeInterpreterInput(BaseModel):
    database_evidence: DatabaseEvidence | dict | str | None = None
    source_refs: list[str] = Field(default_factory=list)
    analysis_goal: str
    code: str | None = None
    constraints: dict = Field(default_factory=dict)
    insight_requests: list[KeyInsightRequest] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data):
        if isinstance(data, dict):
            data = dict(data)
            evidence = data.get("database_evidence")
            if isinstance(evidence, list) and len(evidence) == 1:
                data["database_evidence"] = evidence[0]
            refs = data.get("source_refs")
            if isinstance(refs, str):
                data["source_refs"] = [refs]
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


class _PrimarySourceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class CodeInterpreterTool(BaseTool):
    """Calculate requested Insight values; semantic binding is a separate LLM stage."""

    def __init__(
        self,
        llm=None,
        binder: LLMInsightBinder | None = None,
        *,
        llm_timeout_seconds: float | None = None,
        sandbox_timeout_seconds: float | None = None,
    ):
        policy = load_timeout_policy().tool("code_interpreter")
        self._llm = llm
        self._llm_timeout_seconds = float(
            llm_timeout_seconds
            if llm_timeout_seconds is not None
            else policy.stage_seconds("llm_call_seconds")
        )
        self._sandbox_timeout_seconds = float(
            sandbox_timeout_seconds
            if sandbox_timeout_seconds is not None
            else policy.stage_seconds("sandbox_seconds")
        )
        self._binder = binder or LLMInsightBinder(
            llm,
            timeout_seconds=self._llm_timeout_seconds,
        )

    async def execute(self, validated_input: CodeInterpreterInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        source_refs = list(validated_input.source_refs)
        if isinstance(validated_input.database_evidence, str):
            database_ref = validated_input.database_evidence.strip()
            if database_ref not in {"", "latest", "latest_database_evidence", "current"}:
                source_refs.insert(0, database_ref if database_ref.startswith("evidence:") else f"evidence:{database_ref}")
        try:
            sources = resolve_artifact_sources(request_state, source_refs) if request_state is not None and source_refs else []
        except ValueError as exc:
            raise ValueError(f"code_interpreter requires grounded database_evidence or source_refs: {exc}") from exc
        evidence = (
            database_evidence_for_sources(request_state, sources)
            if request_state is not None and sources
            else _resolve_database_evidence(validated_input.database_evidence, request_state)
        )
        if evidence is None:
            raise ValueError("code_interpreter requires grounded source_refs or database_evidence")
        if not sources and request_state is not None:
            evidence_ref = f"evidence:{evidence.evidence_id}"
            sources = resolve_artifact_sources(request_state, [evidence_ref])
        elif (
            sources
            and request_state is not None
            and _requires_anomaly_adjusted_series(validated_input)
        ):
            sources = _include_database_ancestor_source(request_state, sources, evidence)
        code = str(validated_input.code or "").strip()
        generated_code = not code
        response_language = getattr(request_state, "response_language", "en")
        if generated_code and len(sources) > 1:
            sources = await self._select_primary_source(
                sources=sources,
                goal=validated_input.analysis_goal,
                requests=validated_input.insight_requests,
                response_language=response_language,
            )
        input_source_refs = [source["source_ref"] for source in sources] or [f"evidence:{evidence.evidence_id}"]
        rows, points, columns = _analysis_inputs(evidence)
        primary_input = primary_analysis_input(sources)
        if primary_input is not None:
            rows = primary_input["rows"]
            points = primary_input["points"]
            columns = primary_input["columns"]
        input_insights = _input_insights(request_state, validated_input.insight_requests)
        context = build_canonical_analysis_context(
            rows=rows,
            points=points,
            columns=columns,
            metadata=evidence.metadata,
            diagnostics=evidence.diagnostics,
        )
        context["sources"] = sources
        context["source_by_ref"] = {source["source_ref"]: source for source in sources}
        context["primary_source"] = (
            {
                "source_ref": primary_input["source_ref"],
                "dataset_name": primary_input["dataset_name"],
                "shape": primary_input["shape"],
            }
            if primary_input is not None
            else None
        )
        context["input_insights"] = input_insights
        context["analysis_constraints"] = dict(validated_input.constraints)
        context["authoritative_anomaly_usage_required"] = _requires_anomaly_adjusted_series(validated_input)
        anomaly_context = _authoritative_anomaly_context(request_state, evidence.evidence_id)
        if anomaly_context:
            context["anomaly_context"] = anomaly_context

        if generated_code:
            code = await self._generate_code(
                goal=validated_input.analysis_goal,
                requests=validated_input.insight_requests,
                context=context,
                response_language=response_language,
            )
        repair_attempts = 0
        while True:
            try:
                preflight_error = _preflight_analysis_code(
                    code,
                    require_grounded_computation=generated_code,
                    input_row_count=max(len(rows), len(points)),
                )
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
                    timeout_seconds=self._sandbox_timeout_seconds,
                    work_dir=_work_dir(request_state, code_hash),
                )
                analysis_id = _analysis_id(evidence.evidence_id, validated_input.analysis_goal, code_hash)
                derived_evidence, derived_name_map = _materialize_derived_evidence(
                    sandbox_output.result.get("derived_evidence", []),
                    analysis_id=analysis_id,
                    input_evidence_id=evidence.evidence_id,
                    input_source_refs=input_source_refs,
                    input_insights=input_insights,
                )
                computed = _validate_computed_insights(
                    sandbox_output.result.get("computed_insights", []),
                    requests=validated_input.insight_requests,
                    derived_name_map=derived_name_map,
                )
                anomaly_usage_error = _authoritative_anomaly_usage_error(
                    code=code,
                    computed=computed,
                    context=context,
                    input_source_refs=input_source_refs,
                )
                if anomaly_usage_error:
                    raise AnalysisCodeError(anomaly_usage_error)
                referenced_derived_ids = {
                    evidence_id
                    for insight in computed
                    for evidence_id in insight.derived_evidence_ids
                }
                unreferenced_derived_ids = set(derived_name_map.values()) - referenced_derived_ids
                if unreferenced_derived_ids:
                    raise AnalysisCodeError(
                        "every derived evidence artifact must be referenced by at least one computed insight through "
                        "derived_evidence_names; "
                        f"unreferenced={sorted(unreferenced_derived_ids)}"
                    )
                produced = await self._binder.bind(
                    requests=validated_input.insight_requests,
                    computed=computed,
                    analysis_id=analysis_id,
                    analysis_goal=validated_input.analysis_goal,
                    input_evidence_id=evidence.evidence_id,
                    input_source_refs=input_source_refs,
                    computation_code=code,
                    response_language=response_language,
                )
                break
            except AnalysisCodeError as exc:
                if not generated_code or repair_attempts >= 2:
                    raise
                repair_attempts += 1
                code = await self._generate_code(
                    goal=validated_input.analysis_goal,
                    requests=validated_input.insight_requests,
                    context=context,
                    response_language=response_language,
                    repair_context=_analysis_repair_context(str(exc), code),
                )
        result = AnalysisResult(
            analysis_id=analysis_id,
            analysis_goal=validated_input.analysis_goal,
            code_hash=code_hash,
            input_evidence_id=evidence.evidence_id,
            input_source_refs=input_source_refs,
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
                "input_source_refs": input_source_refs,
                "primary_source": context.get("primary_source"),
            },
        )
        return result.model_dump(mode="json")

    async def _select_primary_source(
        self,
        *,
        sources: list[dict],
        goal: str,
        requests: list[KeyInsightRequest],
        response_language: str,
    ) -> list[dict]:
        """Use semantic ownership to choose the canonical df for multi-source analysis."""

        payload = {
            "analysis_goal": goal,
            "insight_requests": [item.model_dump(mode="json", exclude_none=True) for item in requests],
            "artifact_sources": source_prompt_manifest(sources),
        }
        system = (
            "Choose the one grounded artifact source that should back the canonical df for this calculation. "
            "Select by semantic ownership, not list order: calculations about an existing forecast should use the forecast artifact; "
            "calculations whose result is the anomaly set or anomaly scores should use the anomaly artifact. An anomaly artifact is "
            "only an exclusion overlay when the goal asks for metrics over observations after removing detected anomalies; in that "
            "case the complete observation Evidence must be primary and the anomaly artifact remains auxiliary. Raw evidence is "
            "primary whenever the requested values are owned by the observations themselves. Other sources remain available for composition. "
            "Return exactly one source_ref copied verbatim from artifact_sources and a concise reason."
        )
        messages = [("system", system), ("human", json.dumps(payload, ensure_ascii=False, default=str))]
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                async with llm_trace_span(
                    "Source Selection Repair" if attempt else "Source Selection",
                    summary=(
                        "选择计算所使用的主数据源"
                        if response_language == "zh"
                        else "Select the primary source for the calculation"
                    ),
                    messages=messages,
                ) as trace_span:
                    if hasattr(self._llm, "with_structured_output"):
                        runnable = self._llm.with_structured_output(
                            _PrimarySourceSelection, method="json_schema", include_raw=True,
                        )
                        bundle = await asyncio.wait_for(
                            runnable.ainvoke(messages), timeout=self._llm_timeout_seconds
                        )
                        if isinstance(bundle, dict):
                            trace_response = bundle.get("raw")
                            if trace_span is not None:
                                trace_span.attach_response(
                                    trace_response,
                                    messages=messages,
                                    output_text=_llm_content(trace_response),
                                )
                            parsed = bundle.get("parsed")
                            if parsed is None:
                                raise ValueError(
                                    bundle.get("parsing_error")
                                    or "primary source selection was not parsed"
                                )
                        else:
                            parsed = bundle
                            if trace_span is not None:
                                trace_span.attach_response(
                                    bundle,
                                    messages=messages,
                                    output_text=_llm_content(bundle),
                                )
                        selection = (
                            parsed
                            if isinstance(parsed, _PrimarySourceSelection)
                            else _PrimarySourceSelection.model_validate(parsed)
                        )
                    else:
                        response = await asyncio.wait_for(
                            self._llm.ainvoke(messages), timeout=self._llm_timeout_seconds
                        )
                        raw_content = _llm_content(response)
                        if trace_span is not None:
                            trace_span.attach_response(
                                response,
                                messages=messages,
                                output_text=raw_content,
                            )
                        selection = _PrimarySourceSelection.model_validate_json(raw_content)
                by_ref = {source["source_ref"]: source for source in sources}
                if selection.source_ref not in by_ref:
                    raise ValueError("selected source_ref is not one of the supplied artifact sources")
                return [by_ref[selection.source_ref], *[
                    source for source in sources if source["source_ref"] != selection.source_ref
                ]]
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    messages.append(("human", f"Correct the primary source selection error: {exc}"))
        raise AnalysisCodeError(f"code_interpreter primary source selection failed: {last_error}") from last_error

    async def _generate_code(
        self,
        *,
        goal: str,
        requests: list[KeyInsightRequest],
        context: dict,
        response_language: str,
        repair_context: dict | None = None,
    ) -> str:
        if self._llm is None:
            raise AnalysisCodeError("code_interpreter requires code or an LLM code generator")
        system = prompt_locale_instruction(response_language) + (
            "You generate Python for a computation-only Code Interpreter. Return exactly one JSON object with key code. "
            "The sandbox provides df, time, value, time_col, value_col, series, analysis_context, rows, points, columns, "
            "sources, source_by_ref, "
            "metadata, diagnostics, input_insights, insight_by_key, pd, np, math, and statistics. Do not import or access files. "
            "The prompt contains source contracts rather than source records. Treat source_ref, dataset names, shapes, row counts, "
            "and schema fields as the planning interface; the complete referenced records exist only in the runtime variables above. "
            "Do not call globals, locals, vars, eval, exec, compile, __import__, getattr, setattr, or any introspection helper. "
            "Do not use try/except/raise; inspect the documented inputs directly with ordinary conditionals. "
            "Choose algorithm complexity in proportion to the supplied source shapes. Prefer pandas/numpy vectorization, "
            "sorting, groupby/rolling operations, or a single linear scan for large inputs. Pair/triple enumeration is "
            "acceptable only for demonstrably small bounded inputs; do not use nested data-dependent loops when their "
            "projected work is large. For peak/valley or interval searches over large series, use prefix/suffix extrema "
            "or an equivalent linear-time algorithm. "
            "Do not access network, processes, environment variables, or clocks. All computed values must be produced by the Python program from the "
            "provided sandbox inputs. Generated code must read at least one grounded "
            "data input (for example df, rows, points, sources, source_by_ref, input_insights, or analysis_context); never hardcode computed answers or merely copy "
            "answer values from this prompt. Preserve temporal and other ordering constraints expressed by the requested calculation. "
            "df, rows, points, columns, time_col, and value_col are bound to the first explicit artifact source in source_refs, "
            "not to that artifact's database ancestor. Treat analysis_context.primary_source as authoritative for what df represents. "
            "Always read df/rows/points directly for the primary source; do not look the primary source up again through source_by_ref. "
            "Use source_by_ref only when the calculation intentionally composes additional referenced sources. Each source exposes "
            "datasets as a list of {name, rows, scalar, ...} and also exposes every dataset directly by its name, for example "
            "source_by_ref[ref]['forecast_points'] or source_by_ref[ref]['anomaly_points']. "
            "Forecast and anomaly artifacts are immutable specialized outputs. When a forecast source is supplied, never fit, "
            "extrapolate, smooth, clean, or generate another forecast in Python; derive requested values from its forecast_points. "
            "When an anomaly source is supplied, never redetect or replace its anomaly set. For a comparison between the latest "
            "observation and forecast endpoint, read each value from its owning source through source_by_ref. "
            "Never manufacture uncertainty bands from an arbitrary fixed percentage or unsupported constant. If grounded forecast "
            "intervals are absent and an interval was explicitly requested, return that Insight as unavailable; if no interval was "
            "requested, do not create one. "
            "Match the computation's scope to the requested claim. An overall/global trend must use the observations across the "
            "analysis interval through a defensible trend estimator and report that method and the number of valid observations in "
            "calculation_trace. Comparing only the first and last values supports an endpoint-change Insight, never an unqualified "
            "overall/global trend. Filtering or sorting all observations before selecting only the endpoints still does not make the "
            "estimator consume the interval observations. If the available observations cannot support the requested scope, return that Insight unavailable. "
            "When filtering timestamps, convert both the data column and every comparison boundary to compatible timezone-aware or "
            "timezone-naive datetime values before comparison; never compare pandas Timestamp values with raw strings. "
            "Assign one dict to result with exactly two keys: "
            "computed_insights and derived_evidence. computed_insights must contain exactly one object per requested insight_key. "
            "Each object contains insight_key, value or items, a non-empty calculation_trace describing formula/inputs as concise text or JSON, and optional "
            "unavailable_reason only when the grounded inputs make the requested calculation impossible. Never invent a placeholder value. "
            "Produce only the requested calculation. Do not copy unrelated extrema, timestamps, boundaries, or other existing facts into a computed Insight or derived Evidence for display context; consumers must cite those facts from their existing Insight or artifact refs. "
            "When an Insight contains concrete timestamped observations or decisions that can be visually verified, preserve the summary in value and also emit one item per locator. "
            "Each item must carry its JSON-native value, timestamp, and semantic dimensions such as role; buy/sell points, extrema, boundaries, and similar multi-point results must not exist only as timestamps nested inside one value object. "
            "When a derived Evidence artifact contains the complete records used to verify an Insight, list that artifact's exact name in "
            "the Insight object's derived_evidence_names. Do not emit an unreferenced derived artifact or nest a second items collection "
            "inside an Insight item when those records belong in derived Evidence. Do not produce statements, names, semantic classes, display roles, Key Insight objects, Data Views, "
            "charts, summaries, repair policies, or final-answer prose. Persist a named derived Evidence artifact whenever the requested calculation "
            "creates a complete grouped, aggregated, resampled, differenced, joined, filtered, or otherwise transformed table/series that directly "
            "supports an Insight or is required by the requested downstream shape. Do not discard that relationship after reducing it to a scalar: "
            "the scalar remains the Insight value, while the complete relationship belongs in derived_evidence and its name belongs in that Insight's "
            "derived_evidence_names. derived_evidence may be empty only when the calculation produces no reusable or independently inspectable table/series. "
            "Each artifact contains name, either rows as a list of JSON objects "
            "or scalar as a JSON object, and transform_summary. Do not emit a shape field; the tool derives artifact shape from its data. "
            "Never duplicate, filter, or rename the selected raw Evidence as derived Evidence when its values are unchanged; consumers already have the complete source artifact. "
            "Never use DataFrame.shape or values.tolist() as artifact rows; use df.to_dict(orient='records'). Use exact authoritative anomaly_context points when present; do not run another detector. "
            "When anomaly_context is used, include its exact source_ref in every affected calculation_trace. "
            "When the goal, analysis constraints, or Insight requirements request a cleaned/anomaly-excluded observation "
            "series, an explicit anomaly artifact is authoritative: exclude its exact identities from the primary observations "
            "and include that anomaly source_ref in every affected Insight trace. An anomaly artifact used only as unrelated "
            "context does not alter calculations owned by another source. Merely citing an applicable anomaly artifact without "
            "applying it is invalid. "
            "Convert numpy/pandas values and timestamps to JSON-native scalars and ISO strings. Never return NaN or Infinity."
        )
        payload = {
            "goal": goal,
            "insight_requests": [request.model_dump(mode="json", exclude_none=True) for request in requests],
            "canonical_inputs": _analysis_schema_contract(context),
            "primary_source": context.get("primary_source"),
            "input_insights": context.get("input_insights", []),
            "analysis_constraints": context.get("analysis_constraints", {}),
            "anomaly_context": _anomaly_prompt_contract(context.get("anomaly_context")),
            "artifact_sources": source_prompt_manifest(context.get("sources", [])),
            "repair_context": repair_context,
        }
        messages = [("system", system), ("human", json.dumps(payload, ensure_ascii=False, default=str))]
        is_repair = repair_context is not None
        last_error: Exception | None = None
        for attempt in range(2):
            raw_content = ""
            try:
                async with llm_trace_span(
                    "Analysis Contract Repair" if attempt else "Analysis Repair" if is_repair else "Analysis Planning",
                    summary=(
                        "修正模型返回的分析方案格式"
                        if response_language == "zh" and attempt
                        else "修正未通过验证的分析方案"
                        if response_language == "zh" and is_repair
                        else "选择分析方法并生成执行方案"
                        if response_language == "zh"
                        else "Repair the analysis-plan output contract"
                        if attempt
                        else "Repair an analysis plan that failed validation"
                        if is_repair
                        else "Choose the analysis method and execution plan"
                    ),
                    messages=messages,
                ) as trace_span:
                    if hasattr(self._llm, "with_structured_output"):
                        runnable = self._llm.with_structured_output(
                            _GeneratedCode, method="json_schema", include_raw=True,
                        )
                        bundle = await asyncio.wait_for(
                            runnable.ainvoke(messages), timeout=self._llm_timeout_seconds
                        )
                        if isinstance(bundle, dict):
                            trace_response = bundle.get("raw")
                            raw_content = _llm_content(trace_response)
                            if trace_span is not None:
                                trace_span.attach_response(
                                    trace_response,
                                    messages=messages,
                                    output_text=raw_content,
                                )
                            parsed = bundle.get("parsed")
                            if parsed is None:
                                raise ValueError(
                                    bundle.get("parsing_error")
                                    or "structured output was not parsed"
                                )
                            generated = (
                                parsed
                                if isinstance(parsed, _GeneratedCode)
                                else _GeneratedCode.model_validate(parsed)
                            )
                        else:
                            generated = (
                                bundle
                                if isinstance(bundle, _GeneratedCode)
                                else _GeneratedCode.model_validate(bundle)
                            )
                            if trace_span is not None:
                                trace_span.attach_response(
                                    bundle,
                                    messages=messages,
                                    output_text=_llm_content(bundle),
                                )
                    else:
                        response = await asyncio.wait_for(
                            self._llm.ainvoke(messages), timeout=self._llm_timeout_seconds
                        )
                        raw_content = _llm_content(response)
                        if trace_span is not None:
                            trace_span.attach_response(
                                response,
                                messages=messages,
                                output_text=raw_content,
                            )
                        generated = _GeneratedCode.model_validate_json(raw_content)
                    return generated.code.strip()
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    messages.extend([
                        ("assistant", raw_content),
                        (
                            "human",
                            f"Your output violated the required schema: {exc}. "
                            "Return one corrected schema-valid object only.",
                        ),
                    ])
        raise AnalysisCodeError(
            f"code generation LLM violated its output contract: {last_error}"
        ) from last_error


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
    input_source_refs: list[str] | None = None,
    input_insights: list[dict],
) -> tuple[list[DerivedEvidence], dict[str, str]]:
    if not isinstance(raw, list):
        raise AnalysisCodeError("derived_evidence must be a list")
    lineage = [
        *(input_source_refs or [f"evidence:{input_evidence_id}"]),
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
        payload = dict(item)
        payload.pop("shape", None)
        payload["shape"] = _derived_evidence_shape(payload)
        try:
            artifact = DerivedEvidence.model_validate({**payload, "evidence_id": evidence_id, "lineage": lineage})
        except Exception as exc:
            raise AnalysisCodeError(f"invalid derived evidence '{name}': {exc}") from exc
        result.append(artifact)
        name_map[name] = evidence_id
    return result, name_map


def _derived_evidence_shape(payload: dict) -> str:
    scalar = payload.get("scalar")
    if isinstance(scalar, dict) and scalar:
        return "scalar"
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        return "records"
    fields = {str(key).lower() for row in rows for key in row}
    if {"lower", "upper"}.issubset(fields):
        return "intervals"
    if fields & {"timestamp", "time", "datetime", "date"}:
        return "timeseries"
    return "records"


def _resolve_database_evidence(value, request_state) -> DatabaseEvidence | None:
    if request_state is None:
        if isinstance(value, DatabaseEvidence):
            return value
        if isinstance(value, dict):
            return DatabaseEvidence.model_validate(value)
        return None
    try:
        return resolve_database_evidence(value, request_state, tool_label="Code Interpreter")
    except ValueError as exc:
        raise ValueError(f"code_interpreter requires grounded database_evidence: {exc}") from exc


def _include_database_ancestor_source(request_state, sources: list[dict], evidence: DatabaseEvidence) -> list[dict]:
    """Expose a specialized artifact's complete database ancestor as an auxiliary source."""

    evidence_ref = f"evidence:{evidence.evidence_id}"
    if any(str(source.get("source_ref")) == evidence_ref for source in sources):
        return sources
    return [*sources, *resolve_artifact_sources(request_state, [evidence_ref])]


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


def _analysis_schema_contract(context: dict) -> dict:
    """Expose the executable namespace contract while keeping records in the sandbox data plane."""

    schema = context.get("schema") if isinstance(context, dict) else None
    if not isinstance(schema, dict):
        return {}
    return {
        str(key): value
        for key, value in schema.items()
        if not str(key).startswith("sample_")
    }


def _anomaly_prompt_contract(context: Any) -> dict | None:
    """Describe authoritative anomaly data without embedding its points in the generation prompt."""

    if not isinstance(context, dict):
        return None
    points = [item for item in context.get("anomaly_points", []) if isinstance(item, dict)]
    fields: list[dict[str, str]] = []
    for point in points:
        for key, value in point.items():
            name = str(key)
            if any(field["name"] == name for field in fields):
                continue
            fields.append({"name": name, "type": type(value).__name__})
    return {
        "source_ref": context.get("source_ref"),
        "point_count": len(points),
        "schema_fields": fields,
        "runtime_variable": "anomaly_context",
    }


def _preflight_analysis_code(
    code: str | None,
    *,
    require_grounded_computation: bool = False,
    input_row_count: int | None = None,
) -> str | None:
    try:
        prepared = prepare_analysis_code(str(code or ""))
    except AnalysisPolicyError as exc:
        return str(exc)
    if require_grounded_computation:
        grounded_inputs = {
            "df",
            "time",
            "value",
            "rows",
            "points",
            "input_insights",
            "insight_by_key",
            "analysis_context",
            "anomaly_context",
            "sources",
            "source_by_ref",
        }
        if not prepared.loaded_names.intersection(grounded_inputs):
            return (
                "generated analysis code must compute from grounded sandbox inputs; "
                "hardcoded result literals are not accepted"
            )
        complexity_error = _generated_complexity_error(
            prepared.code,
            input_row_count=input_row_count,
        )
        if complexity_error:
            return complexity_error
    return None


def _analysis_repair_context(error: str, rejected_code: str) -> dict:
    """Give code repair an explicit optimization contract after sandbox timeouts."""

    context = {"validation_error": error, "rejected_code": rejected_code}
    normalized = error.casefold()
    if "sandbox timeout" in normalized or ("exceeded" in normalized and "timeout" in normalized):
        context.update({
            "failure_type": "sandbox_timeout",
            "required_repair": (
                "Replace the algorithm; do not merely reformat or make small edits to the rejected code. "
                "Use vectorized pandas/numpy operations, sorting plus prefix/suffix extrema, or one linear scan. "
                "The replacement must be O(n log n) or better and must not enumerate observation pairs/triples "
                "or contain nested data-dependent loops."
            ),
            "forbidden_patterns": [
                "nested data-dependent loops",
                "all-pairs enumeration",
                "all-triples enumeration",
                "retrying the same algorithm with cosmetic changes",
            ],
        })
    return context


_MAX_PROJECTED_LOOP_ITERATIONS = 10_000_000_000


def _generated_complexity_error(code: str, *, input_row_count: int | None) -> str | None:
    """Reject nested iteration only when the actual input size makes it unsafe."""

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return None  # Syntax diagnostics are owned by the shared code policy.

    if input_row_count is None or input_row_count <= 0:
        return None

    depth = _maximum_iteration_depth(tree)
    if depth < 2:
        return None
    projected = input_row_count ** depth
    if projected >= _MAX_PROJECTED_LOOP_ITERATIONS:
        return (
            f"generated analysis code has iteration depth {depth} over {input_row_count} input records "
            f"(projected work about {projected:,} iterations), exceeding the safe preflight budget of "
            f"{_MAX_PROJECTED_LOOP_ITERATIONS:,}; rewrite it with vectorized pandas/numpy operations, "
            "sorting with prefix/suffix extrema, or a lower-complexity scan"
        )
    return None


def _maximum_iteration_depth(tree: ast.AST) -> int:
    def visit(node: ast.AST, depth: int) -> int:
        if isinstance(node, (ast.For, ast.While, ast.comprehension)):
            depth += 1
        maximum = depth
        for child in ast.iter_child_nodes(node):
            maximum = max(maximum, visit(child, depth))
        return maximum

    return visit(tree, 0)


def _authoritative_anomaly_usage_error(
    *,
    code: str,
    computed: list[ComputedInsight],
    context: dict,
    input_source_refs: list[str],
) -> str | None:
    """Require transparent use of an explicitly supplied anomaly artifact.

    This is a semantic execution invariant, not an outlier detector: the LLM still
    authors the cleaning calculation and repairs it when the invariant is missing.
    """

    anomaly = context.get("anomaly_context") if isinstance(context, dict) else None
    if not isinstance(anomaly, dict) or not anomaly.get("anomaly_points"):
        return None
    if context.get("authoritative_anomaly_usage_required") is not True:
        return None
    explicit_refs = [str(ref) for ref in input_source_refs if str(ref).startswith("anomaly:")]
    if not explicit_refs:
        return None
    source_ref = str(anomaly.get("source_ref") or explicit_refs[0])
    loaded_names = prepare_analysis_code(code).loaded_names
    if not loaded_names.intersection({"anomaly_context", "analysis_context", "source_by_ref", "sources"}):
        return (
            f"explicit anomaly source '{source_ref}' was not consumed. Rebuild the computation from the primary series "
            "after excluding the exact authoritative anomaly identities; do not redetect anomalies."
        )
    missing_trace = [
        item.insight_key
        for item in computed
        if source_ref not in _flatten_trace(item.calculation_trace)
    ]
    if missing_trace:
        return (
            f"explicit anomaly source '{source_ref}' is absent from calculation_trace for {missing_trace}. "
            "Apply the authoritative anomaly exclusions and record the exact source_ref in every affected trace."
        )
    return None


def _requires_anomaly_adjusted_series(value: CodeInterpreterInput) -> bool:
    """Recognize an explicit semantic request to compute over an anomaly-adjusted series."""

    contract = {
        "goal": value.analysis_goal,
        "constraints": value.constraints,
        "requirements": [item.requirements for item in value.insight_requests],
    }
    text = json.dumps(contract, ensure_ascii=False, default=str).casefold()
    return any(token in text for token in (
        "exclude_anomal", "anomaly_exclusion", "anomaly-adjusted", "outlier_exclusion",
        "exclude_outlier", "cleaned_series", "排除异常", "剔除异常", "排除离群", "剔除离群",
    ))


def _flatten_trace(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_trace(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_flatten_trace(item) for item in value)
    return str(value or "")


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
