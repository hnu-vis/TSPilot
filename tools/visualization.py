"""Two-stage, LineChart-first visualization tool."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field

from core.visualization import LineChartCompiler, PresentationCatalog, VisualizationArtifactStore
from core.visualization.planning_schema import (
    build_linechart_response_schema,
    content_line_capability_errors,
)
from runtime.llm_trace import llm_trace_span
from runtime.prompt_locale import prompt_locale_instruction
from runtime.token_usage import record_llm_token_usage
from runtime.timeout_policy import load_timeout_policy
from schemas.linechart_plan import (
    LineChartPlan,
    StructuredVisualContentPlan,
    VisualContentPlan,
    VisualizationEvidenceRequest,
)
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
    """Decide what the user needs to inspect, then compose grounded LineCharts."""

    def __init__(
        self,
        *,
        llm,
        artifact_store: VisualizationArtifactStore,
        llm_timeout_seconds: float | None = None,
        **_ignored,
    ):
        self._llm = llm
        self._artifact_store = artifact_store
        self._llm_timeout_seconds = float(
            llm_timeout_seconds
            if llm_timeout_seconds is not None
            else load_timeout_policy().tool("visualization").stage_seconds("llm_call_seconds")
        )

    async def close(self) -> None:
        return None

    async def execute(
        self,
        validated_input: VisualizationInput,
        *,
        request_state: RequestStateModel,
        **kwargs,
    ) -> dict:
        request = validated_input.model_copy(update={
            "constraints": _semantic_constraints(validated_input.constraints),
        })
        catalog = PresentationCatalog(request_state)
        preferred, unknown = catalog.expand_preferences(request.source_refs)
        if unknown:
            raise _tool_error(f"unknown visualization source refs: {sorted(unknown)}", stage="source_inventory")
        inventory = catalog.planner_inventory(preferred)
        target_insight_ids = _requested_insight_ids(request.source_refs, catalog)

        content = None
        content_repair = None
        for attempt in range(3):
            try:
                content = await self._plan_content(
                    request, inventory, target_insight_ids, request_state, content_repair,
                )
                if content.required_data_request is not None:
                    return _dependency_result(_normalize_dependency(content.required_data_request, catalog), request_state)
                _validate_content_plan(content, inventory, target_insight_ids)
                break
            except (StructuredToolError, ValueError) as exc:
                content_repair = _repair_context("content_planning", attempt, exc)
        else:
            return _unavailable_result(
                "Visual content planning remained invalid after two LLM repair attempts: "
                + str((content_repair or {}).get("error") or "unknown validation error")
            )
        assert content is not None

        repair_context = None
        for attempt in range(3):
            try:
                plan = await self._plan_linechart(request, inventory, content, request_state, repair_context)
                if plan.required_data_request is not None:
                    return _dependency_result(_normalize_dependency(plan.required_data_request, catalog), request_state)
                visualizations = LineChartCompiler(catalog).compile(content, plan)
                descriptors = [self._artifact_store.put(item) for item in visualizations]
                source_refs = list(dict.fromkeys(ref for item in descriptors for ref in item.source_refs))
                return VisualizationResult(
                    summary=f"Created {len(descriptors)} grounded LineChart artifact(s).",
                    visualization_ids=[item.visualization_id for item in descriptors],
                    visualizations=descriptors,
                    source_refs=source_refs,
                ).model_dump(mode="json")
            except (StructuredToolError, ValueError) as exc:
                repair_context = _repair_context("linechart_compilation", attempt, exc)
        return _unavailable_result(
            "LineChart composition remained invalid after two LLM repair attempts: "
            + str((repair_context or {}).get("error") or "unknown validation error")
        )

    async def _plan_content(self, request, inventory, target_ids, request_state, repair) -> VisualContentPlan:
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You are the visual-content planner for a LineChart-first analytical system. Decide what a human must see "
            "to inspect the user's conclusions before choosing chart components. Return exactly one VisualContentPlan. "
            "Prefer one primary chart with a complete time-series host and place every compatible target Insight with its "
            "context in that goal. Create a supporting goal only when time domain, measure, or unit incompatibility would "
            "make one chart misleading. Every content item must use an exact source_ref from the inventory. Include all "
            "requested target Insight ids and their inspectable context. When returning goals, set required_data_request to "
            "null; when returning required_data_request, return no goals and no target Insight ids. target_insight_ids must "
            "contain exactly the requested target Insight ids, and every one must appear in at least one visible content "
            "item's insight_ids. If that is impossible from the inventory, request the missing source instead. Do not output "
            "lines, points, axes, fields, colors, "
            "or renderer options. If a required relationship is not already materialized, return only required_data_request: "
            "sql_query owns raw observations, code_interpreter owns derived series/calculations, forecast owns forecast output, "
            "and anomaly owns anomaly output. Never invent a fallback.\n"
            f"User request: {request.message}\n"
            f"Original task: {request_state.message}\n"
            f"Requested target Insight ids: {json.dumps(target_ids, ensure_ascii=False)}\n"
            f"User constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Grounded source inventory: {json.dumps(inventory, ensure_ascii=False)}\n"
            f"Validation repair context: {json.dumps(repair, ensure_ascii=False) if repair else 'none'}"
        )
        structured = await self._invoke_plan(
            StructuredVisualContentPlan,
            [("system", prompt), ("user", request.message)],
            request_state,
            title="Visual Content Planning",
            summary="明确用户需要在 LineChart 中观察的内容",
            source="visualization.content_planning",
        )
        return structured.to_runtime()

    async def _plan_linechart(self, request, inventory, content, request_state, repair) -> LineChartPlan:
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You compose complete grounded LineChart plans from an already-fixed VisualContentPlan. Return exactly one "
            "LineChartPlan. Produce one chart for every content goal and cover every content_id at least once. Each component "
            "selects exactly one content_id; never output source_ref because the compiler derives its unique source from that "
            "content item, and never output component_id because the compiler assigns stable component identity. Field values "
            "are closed enums generated from that content item's real source. Every chart requires one host_line, whose "
            "closed content enum guarantees that it renders the goal's declared host source with two or more grounded points. "
            "Use additional lines for contextual or comparison series, point for located "
            "observations/events, band for true lower/upper uncertainty, interval for grounded start/end ranges, reference_line "
            "only for a scalar with the same measure and unit as its y axis, and annotation for text or semantically different "
            "scalars. Prefer one y axis; use a second only when the content requires it and the visual comparison remains honest. "
            "Enable requested standard interactions. Do not calculate, aggregate, rename sources, invent fields, or add charts. "
            "If the fixed content cannot be expressed from available fields, return required_data_request instead of a fallback.\n"
            f"User request: {request.message}\n"
            f"Visual content plan: {json.dumps(content.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Grounded source inventory: {json.dumps(inventory, ensure_ascii=False)}\n"
            f"Validation repair context: {json.dumps(repair, ensure_ascii=False) if repair else 'none'}"
        )
        response_schema = build_linechart_response_schema(content, inventory)
        structured = await self._invoke_plan(
            response_schema,
            [("system", prompt), ("user", request.message)],
            request_state,
            title="LineChart Composition",
            summary="组装 LineChart 组件、字段、坐标轴与交互",
            source="visualization.linechart_composition",
        )
        return structured.to_runtime()

    async def _invoke_plan(self, schema, messages, request_state, *, title, summary, source):
        started = time.perf_counter()
        response, content, parsed, error = await _invoke_structured(
            self._llm,
            schema,
            messages,
            timeout_seconds=self._llm_timeout_seconds,
            trace_title=title,
            trace_summary=summary,
        )
        record_llm_token_usage(
            request_state,
            source=source,
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        if error is not None or parsed is None:
            raise _tool_error(f"{title} returned an invalid structured plan: {error}", stage=source)
        return parsed


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


def _validate_content_plan(content: VisualContentPlan, inventory: dict, required_ids: list[str]) -> None:
    known_refs = {str(item.get("source_ref")) for item in inventory.get("sources", []) if isinstance(item, dict)}
    used_refs = {item.source_ref for goal in content.goals for item in goal.content} | {goal.host_source_ref for goal in content.goals}
    unknown = used_refs - known_refs
    if unknown:
        raise _tool_error(f"content plan references unknown sources: {sorted(unknown)}", stage="content_validation")
    missing_targets = set(required_ids) - set(content.target_insight_ids)
    if missing_targets:
        raise _tool_error(f"content plan omitted requested Insights: {sorted(missing_targets)}", stage="content_validation")
    content_targets = {insight_id for goal in content.goals for item in goal.content for insight_id in item.insight_ids}
    if set(content.target_insight_ids) - content_targets:
        raise _tool_error("target Insights must be attached to visible content items", stage="content_validation")
    line_errors = content_line_capability_errors(content, inventory)
    if line_errors:
        raise _tool_error(
            "content goals require a line-capable host: " + json.dumps(line_errors, ensure_ascii=False),
            stage="content_validation",
        )


def _repair_context(stage: str, attempt: int, exc: Exception) -> dict:
    diagnostics = getattr(exc, "diagnostics", None)
    return {
        "stage": stage,
        "attempt": attempt + 1,
        "error": str(exc)[:1800],
        "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
        "instruction": "Rebuild the complete plan from the closed source contract; do not invent a fallback.",
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
            return _unavailable_result("The required visual source remained unavailable after its owner action completed.")
        break
    return VisualizationResult(
        status="needs_sources",
        summary="Visualization requires an additional grounded source before LineChart composition.",
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
        "insight_keys": sorted(
            str(item.get("insight_key") or item.get("name") or "")
            for item in payload.get("insight_requests") or []
            if isinstance(item, dict)
        ),
    }
    return json.dumps(stable, sort_keys=True, default=str)


def _unavailable_result(reason: str) -> dict:
    return VisualizationResult(
        status="unavailable",
        summary="No LineChart was published because grounded composition did not pass.",
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
        recommended_next_action="Retry visualization with a corrected grounded plan.",
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
