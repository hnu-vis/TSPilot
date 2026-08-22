"""Single-stage grounded native-ECharts visualization tool."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field

from core.visualization import EChartsCompiler, PresentationCatalog, VisualizationArtifactStore
from runtime.llm_trace import llm_trace_span
from runtime.prompt_locale import localized_payload_label, prompt_locale_instruction
from runtime.token_usage import record_llm_token_usage
from runtime.timeout_policy import load_timeout_policy
from schemas.echarts_plan import EChartsPlan, StructuredEChartsPlan, VisualizationEvidenceRequest
from schemas.state import RequestStateModel
from schemas.visualization import VisualizationPayload
from tools.base import BaseTool, StructuredToolError


class VisualizationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    source_refs: list[str] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)


class VisualizationResult(BaseModel):
    status: Literal["created", "needs_sources", "unavailable"] = "created"
    summary: str
    visualization_ids: list[str] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    required_data_request: VisualizationEvidenceRequest | None = None
    unavailable_reason: str | None = None


class VisualizationTool(BaseTool):
    """Ask the LLM for native option JSON, then ground and validate it."""

    def __init__(self, *, llm, artifact_store: VisualizationArtifactStore, llm_timeout_seconds: float | None = None, **_ignored):
        self._llm = llm
        self._artifact_store = artifact_store
        self._llm_timeout_seconds = float(
            llm_timeout_seconds
            if llm_timeout_seconds is not None
            else load_timeout_policy().tool("visualization").stage_seconds("llm_call_seconds")
        )

    async def close(self) -> None:
        return None

    async def execute(self, validated_input: VisualizationInput, *, request_state: RequestStateModel, **kwargs) -> dict:
        request = validated_input.model_copy(update={"constraints": _semantic_constraints(validated_input.constraints)})
        catalog = PresentationCatalog(request_state)
        preferred, unknown = catalog.expand_preferences(request.source_refs)
        if unknown:
            raise _tool_error(f"unknown visualization source refs: {sorted(unknown)}", stage="source_inventory")
        inventory = catalog.planner_inventory(preferred)
        target_ids = _requested_insight_ids(request.source_refs, catalog)
        repair = None
        for attempt in range(3):
            try:
                plan = await self._plan(request, inventory, target_ids, request_state, repair)
                if plan.required_data_request is not None:
                    return _dependency_result(_normalize_dependency(plan.required_data_request, catalog), request_state)
                _validate_plan_targets(plan, target_ids)
                payloads = EChartsCompiler(catalog).compile(plan)
                descriptors = [self._artifact_store.put(item) for item in payloads]
                refs = list(dict.fromkeys(ref for item in descriptors for ref in item.source_refs))
                return VisualizationResult(
                    summary=localized_payload_label(
                        request_state.response_language,
                        zh=f"已创建 {len(descriptors)} 个有证据绑定的原生 ECharts 图表。",
                        en=f"Created {len(descriptors)} grounded native ECharts artifact(s).",
                    ),
                    visualization_ids=[item.visualization_id for item in descriptors],
                    visualizations=descriptors,
                    source_refs=refs,
                ).model_dump(mode="json")
            except (StructuredToolError, ValueError) as exc:
                repair = _repair_context(attempt, exc)
        return _unavailable_result(
            "Native ECharts composition remained invalid after two LLM repair attempts: "
            + str((repair or {}).get("error") or "unknown validation error"),
            request_state.response_language,
        )

    async def _plan(self, request, inventory, target_ids, request_state, repair) -> EChartsPlan:
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You design one or more complete native Apache ECharts 5 option JSON objects for an analytical answer. "
            "Return one closed EChartsPlan; put each option inside option_json as a JSON string. Use only line, scatter, "
            "or bar series and native markPoint, markLine, and markArea. Bind complete records only with "
            "dataset.source={\"$dataset\":\"EXACT_SOURCE_REF\"}; never emit literal dataset.source or series.data. "
            "Bind a grounded scalar inside mark data with {\"$value\":{\"source_ref\":\"EXACT_SOURCE_REF\","
            "\"field\":\"EXACT_FIELD\"}}. Every series must bind a dataset and use encode fields exactly from that "
            "source schema. Use numeric xAxisIndex/yAxisIndex only when necessary and only for axes that exist. Prefer "
            "one x axis, one y axis, one complete context line, concise legend names, UTC time axes, non-overlapping labels, "
            "and exactly the endpoint/interval marks needed to verify the conclusion. Do not duplicate a series to simulate "
            "points. Do not include transforms, functions, code, HTML, DOM, external assets, URLs, toolbox data views, or "
            "calculated values. All requested target Insight ids must enter through a $dataset or $value placeholder. "
            "source_refs are inferred by the compiler and must not be declared. If required values are absent, return only "
            "required_data_request and let sql_query own raw data, code_interpreter own calculations, forecast own forecasts, "
            "or anomaly own anomaly detection. Never invent or fall back to a generic chart. Repair errors contain exact JSON "
            "Pointers; rebuild the complete option and correct those paths. Keep all user-visible chart text in the requested "
            "response language.\n"
            f"User visualization request: {request.message}\n"
            f"Original task: {request_state.message}\n"
            f"Requested target Insight ids: {json.dumps(target_ids, ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Grounded Source Inventory: {json.dumps(inventory, ensure_ascii=False)}\n"
            f"Validation repair context: {json.dumps(repair, ensure_ascii=False) if repair else 'none'}"
        )
        started = time.perf_counter()
        messages = [("system", prompt), ("user", request.message)]
        response, content, parsed, error = await _invoke_structured(
            self._llm,
            StructuredEChartsPlan,
            messages,
            timeout_seconds=self._llm_timeout_seconds,
            trace_title=localized_payload_label(
                request_state.response_language,
                zh="原生 ECharts 规划",
                en="Native ECharts Planning",
            ),
            trace_summary=localized_payload_label(
                request_state.response_language,
                zh="生成并校验原生 ECharts option",
                en="Generate and validate a native ECharts option",
            ),
        )
        record_llm_token_usage(
            request_state,
            source="visualization.echarts_planning",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        if error is not None or parsed is None:
            raise _tool_error(f"Native ECharts Planning returned an invalid structured plan: {error}", stage="echarts_planning")
        return parsed.to_runtime()


async def _invoke_structured(llm, schema, messages, *, timeout_seconds, trace_title, trace_summary=None):
    response = None
    content = ""
    try:
        async with llm_trace_span(trace_title, summary=trace_summary, messages=messages) as trace_span:
            if hasattr(llm, "with_structured_output"):
                runnable = llm.with_structured_output(schema, method="json_schema", include_raw=True, strict=True)
                bundle = await asyncio.wait_for(runnable.ainvoke(messages), timeout=timeout_seconds)
                if isinstance(bundle, dict):
                    response = bundle.get("raw")
                    content = _llm_content(response)
                    if trace_span is not None:
                        trace_span.attach_response(response, messages=messages, output_text=content)
                    parsed = bundle.get("parsed")
                    if parsed is None:
                        raise ValueError(str(bundle.get("parsing_error") or "structured output was not parsed"))
                    return response, content, parsed if isinstance(parsed, schema) else schema.model_validate(parsed), None
                return bundle, _llm_content(bundle), bundle if isinstance(bundle, schema) else schema.model_validate(bundle), None
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout_seconds)
            content = _llm_content(response)
            if trace_span is not None:
                trace_span.attach_response(response, messages=messages, output_text=content)
            return response, content, schema.model_validate(json.loads(_json_object(content))), None
    except (json.JSONDecodeError, ValueError, OutputParserException) as exc:
        return response, content, None, exc


def _requested_insight_ids(refs: list[str], catalog: PresentationCatalog) -> list[str]:
    result = []
    for ref in refs:
        if not str(ref).startswith("insight:") or "#" in str(ref):
            continue
        source = catalog.resolve(ref)
        insight_id = str(getattr(source.value, "insight_id", "") or "")
        if insight_id:
            result.append(insight_id)
    return list(dict.fromkeys(result))


def _validate_plan_targets(plan: EChartsPlan, required_ids: list[str]) -> None:
    if set(plan.target_insight_ids) != set(required_ids):
        raise _tool_error(
            f"/target_insight_ids: expected exactly {required_ids}, received {plan.target_insight_ids}",
            stage="echarts_validation",
        )


def _repair_context(attempt: int, exc: Exception) -> dict:
    pointer = getattr(exc, "pointer", None)
    return {
        "stage": "echarts_compilation",
        "attempt": attempt + 1,
        "error": str(exc)[:2400],
        "json_pointer": pointer,
        "instruction": "Rebuild the complete native option from grounded placeholders and correct the cited JSON Pointer.",
    }


def _normalize_dependency(requirement: VisualizationEvidenceRequest, catalog: PresentationCatalog):
    refs = catalog.analysis_input_source_refs(requirement.input_source_refs) if requirement.input_source_refs else []
    return requirement.model_copy(update={
        "input_source_refs": refs,
        "input_evidence": next((ref.split(":", 1)[1] for ref in refs if ref.startswith("evidence:")), None),
    })


def _dependency_result(requirement: VisualizationEvidenceRequest, request_state: RequestStateModel) -> dict:
    signature = _dependency_signature(requirement.model_dump(mode="json"))
    for observation in reversed(request_state.observations):
        if observation.tool_name != "visualization" or not observation.success:
            continue
        payload = observation.payload if isinstance(observation.payload, dict) else {}
        previous = payload.get("required_data_request")
        if isinstance(previous, dict) and _dependency_signature(previous) == signature:
            return _unavailable_result(
                "The required visual source remained unavailable after its owner action completed.",
                request_state.response_language,
            )
        break
    return VisualizationResult(
        status="needs_sources",
        summary=localized_payload_label(
            request_state.response_language,
            zh="原生 ECharts 规划前还需要一个有证据依据的数据源。",
            en="Visualization requires an additional grounded source before native ECharts composition.",
        ),
        required_data_request=requirement,
    ).model_dump(mode="json")


def _dependency_signature(payload: dict) -> str:
    stable = {
        "required_action": payload.get("required_action"),
        "purpose": payload.get("purpose"),
        "required_shape": payload.get("required_shape"),
        "required_fields": payload.get("required_fields") or [],
        "required_properties": payload.get("required_properties") or [],
        "input_source_refs": payload.get("input_source_refs") or [],
    }
    return json.dumps(stable, sort_keys=True, default=str)


def _unavailable_result(reason: str, language: str | None = None) -> dict:
    return VisualizationResult(
        status="unavailable",
        summary=localized_payload_label(
            language,
            zh="由于有证据约束的原生 ECharts 组装未通过，因此未发布图表。",
            en="No chart was published because grounded native ECharts composition did not pass.",
        ),
        unavailable_reason=str(reason),
    ).model_dump(mode="json")


def _semantic_constraints(value: dict | None) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item for key, item in value.items()
        if key not in {"mode", "repair_contract", "validation_failure"} and not str(key).startswith("_")
    }


def _tool_error(message: str, *, stage: str) -> StructuredToolError:
    return StructuredToolError(
        message,
        error_type="visualization_planning_error",
        retryable=True,
        diagnostics={"stage": stage},
        recommended_next_action="Retry visualization with a corrected grounded native option.",
    )


def _json_object(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    return text[start:end + 1]


def _llm_content(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content).strip()
    return str(content or "").strip()
