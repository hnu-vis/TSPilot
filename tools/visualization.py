"""Single-stage grounded native-ECharts visualization tool."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Literal

from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ConfigDict, Field

from core.visualization import EChartsCompiler, PresentationCatalog, VisualizationArtifactStore, grounded_annotation_fields
from runtime.llm_trace import llm_trace_span
from runtime.prompt_locale import localized_payload_label, prompt_locale_instruction
from runtime.token_usage import record_llm_token_usage
from runtime.timeout_policy import load_timeout_policy
from schemas.echarts_plan import (
    EChartsPlan,
    StructuredEChartsPlan,
    StructuredEChartsPlanWithoutTimeAnnotations,
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
    """Ask the LLM for a semantic chart plan, then compile and validate it."""

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
        inventory = (
            catalog.targeted_planner_inventory(preferred)
            if preferred
            else catalog.planner_inventory()
        )
        target_ids = _requested_insight_ids(request.source_refs, catalog)
        repair = None
        failures: list[dict] = []
        for attempt in range(3):
            try:
                plan = await self._plan(request, inventory, target_ids, request_state, repair)
                if plan.required_data_request is not None:
                    return _dependency_result(_normalize_dependency(plan.required_data_request, catalog), request_state)
                _validate_plan_targets(plan, target_ids)
                payloads = EChartsCompiler(catalog).compile(plan)
                _validate_lineage_coverage(payloads, target_ids, inventory, catalog)
                _validate_trajectory_context(payloads, inventory)
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
                repair = _repair_context(attempt, exc, request_state.response_language, failures)
                failures.append(repair)
        return _unavailable_result(
            "Native ECharts composition remained invalid after two LLM repair attempts: "
            + " | ".join(str(item.get("error") or "unknown validation error") for item in failures),
            request_state.response_language,
        )

    async def _plan(self, request, inventory, target_ids, request_state, repair) -> EChartsPlan:
        grounded_input = _line_chart_grounded_input(inventory, catalog=PresentationCatalog(request_state))
        plan_schema = (
            StructuredEChartsPlan
            if _grounded_input_has_annotation_type(grounded_input, "time")
            else StructuredEChartsPlanWithoutTimeAnnotations
        )
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "Create one native Apache ECharts 5 line chart that answers the user's question from Line Chart Grounded Input. "
            "Make the analytical conclusion visually verifiable; do not display unrelated sources merely because they exist.\n\n"
            "GROUNDED INPUT GUIDE\n"
            "reference_contract explains how refs bind grounded data and Insight values. Each eligible source then states only its purpose, "
            "scope, fields, bounded example_data, lineage, and supported Insights. Each Insight states its content once and identifies its "
            "visual support. Example rows illustrate the source but dataset.source.$dataset resolves the complete materialized source. "
            "A field meaning_status=not_declared means its business meaning is unknown; preserve its name and do not guess a metric, unit, "
            "transformation, or observation grain. Shape labels alone do not prove intermediate observations exist: use scope.record_count, "
            "purpose, example_data, and limitations together.\n\n"
            "SOURCE AND INSIGHT USE\n"
            "Choose the smallest set of eligible_line_sources that still preserves the observed trajectory needed to verify the answer. "
            "When a related multi-observation series exists, use it as the primary line; never connect only two calculated endpoints and "
            "present that straight segment as a price trend, rebound path, stability interval, or other observed trajectory. Treat endpoint-only "
            "sources as calculation evidence and express their values through grounded marks on the complete related line. Each source states its exact time fields, "
            "numeric fields, purpose, scope, and the Insights it supports. Use an Insight's visual_support.line_sources as the visual "
            "evidence for that computed claim. The content object contains the grounded result, calculation trace, operands, and grounding "
            "document; do not recalculate or invent any value. Prefer one time axis, one value axis, and one primary line. Add another "
            "line only when a comparison requires a distinct compatible time series, and cover every compared operand. Put scalar results "
            "in the chart title, summary, tooltip, or grounded annotations. A scalar with a different unit from the line stays in text and "
            "must not become a y-axis value.\n\n"
            "For collection Insights, available_annotation_values may intentionally expose only bounded first/last examples. Do not call a "
            "selected item longest, largest, first, last, or otherwise ranked unless the Insight statement, selection, rank, or calculation "
            "trace explicitly establishes that property. Otherwise describe it only as the grounded interval or point that it is.\n\n"
            "TYPED CHART CONTRACT\n"
            "Return one closed EChartsPlan with exactly one primary chart. Do not write option_json or any renderer-native ECharts JSON; "
            "the compiler owns datasets, axes, series styling, legends, tooltips, marks, and JSON serialization. In each chart, declare 1-2 "
            "typed line series using series_id, name, exact source_ref, x_field, and y_field. Copy x_field from a source field with "
            "role=time_coordinate and y_field from one with role=numeric_measure. Use multiple series only for a necessary compatible "
            "comparison. Keep series_id unique and reference it from annotations. Optional y_axis_name is a concise grounded unit label.\n"
            "Minimal chart fields: chart_id, purpose, priority, title, summary, accessibility_description, accessibility_table_columns, "
            "series, point_annotations, interval_annotations, reference_lines, y_axis_name.\n\n"
            "GROUNDED ANNOTATIONS\n"
            "Use point_annotations for visually important located points. Each point must pair one available_annotation_values entry "
            "with value_type=time and one with value_type=number from the same source_ref and coordinate_group; copy only source_ref and "
            "value_id. Use interval_annotations only when exact start and end time values share the same coordinate_group; their source_refs "
            "may differ when the endpoints are separate Insights for the same kind of located value. "
            "Use reference_lines for a scalar number compatible with the target series y unit. "
            "A reference line must be a level, threshold, or average in the same quantity as the y field; never use a delta, duration, count, "
            "percentage, or another calculation result as an axis level. Add only annotations required to locate or verify the requested "
            "conclusion: do not add unrelated extrema, and do not duplicate a point value as a horizontal reference line. "
            "value_id is an opaque grounded selector: never invent or alter it, and never output a JSON path. If a compatible annotation is "
            "unavailable, omit it and retain the valid supporting line. In particular, if the chosen Insight lists no value_type=time entry, "
            "it cannot support a point or interval annotation even when it has a numeric result; leave those annotation lists empty. "
            "Annotation source_ref must be an Insight listed under insights, never an eligible_line_sources data view.\n\n"
            "DEPENDENCY AND RETRY\n"
            "Set target_insight_ids=[]. If no eligible source can visually support the requested calculation, return required_data_request "
            "for code_interpreter to materialize a multi-record time+number derived series from the stated lineage. With a chart, "
            "required_data_request must be null. On retry, regenerate the complete typed plan from the grounded input and validation findings. "
            "Keep all visible text in the requested response language.\n"
            f"User visualization request: {request.message}\n"
            f"Original task: {request_state.message}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Line Chart Grounded Input: {json.dumps(grounded_input, ensure_ascii=False)}\n"
            f"Validation repair context: {json.dumps(repair, ensure_ascii=False) if repair else 'none'}"
        )
        started = time.perf_counter()
        messages = [("system", prompt), ("user", request.message)]
        response, content, parsed, error = await _invoke_structured(
            self._llm,
            plan_schema,
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


def _grounded_input_has_annotation_type(grounded_input: dict, value_type: str) -> bool:
    return any(
        candidate.get("value_type") == value_type
        for insight in grounded_input.get("insights") or []
        for candidate in (insight.get("content") or {}).get("available_annotation_values") or []
    )


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


def _line_chart_grounded_input(inventory: dict, *, catalog: PresentationCatalog) -> dict[str, object]:
    """Build a question-facing line-source package instead of exposing a flat catalog.

    The package does not decide chart semantics. It makes already-grounded
    relationships explicit: which views are executable lines, which Insights
    they support through catalog lineage, and which exact computed values and
    operands the planner may describe or annotate.
    """

    sources = [source for source in inventory.get("sources") or [] if isinstance(source, dict)]
    eligible: list[dict] = []
    eligible_refs: set[str] = set()
    for source in sources:
        if source.get("kind") != "data_view":
            continue
        contract = source.get("render_contract") if isinstance(source.get("render_contract"), dict) else {}
        time_fields = [str(field) for field in contract.get("time_fields") or [] if field]
        number_fields = [str(field) for field in contract.get("number_fields") or [] if field]
        if int(contract.get("point_count") or 0) < 2 or not time_fields or not number_fields:
            continue
        field_semantics = source.get("field_semantics") if isinstance(source.get("field_semantics"), dict) else {}
        source_ref = str(source.get("source_ref") or "")
        eligible_refs.add(source_ref)
        semantic_contract = source.get("semantic_contract") if isinstance(source.get("semantic_contract"), dict) else {}
        eligible_source = {
            "source_ref": source_ref,
            "purpose": {
                "name": source.get("name"),
                "data_role": semantic_contract.get("data_role"),
                "description": semantic_contract.get("operation_description"),
                "materializes_transformation": semantic_contract.get("materializes_input_transformation"),
                "visual_uses": list(semantic_contract.get("supported_visual_uses") or []),
                "limitations": list(semantic_contract.get("limitations") or []),
            },
            "scope": {
                "shape": source.get("shape"),
                "record_count": source.get("row_count"),
                "time_range": source.get("time_range"),
            },
            "fields": [
                _field_description(field, "time_coordinate", field_semantics)
                for field in time_fields
            ] + [
                _field_description(field, "numeric_measure", field_semantics)
                for field in number_fields
            ],
            "example_data": _source_example_data(catalog, source_ref, source.get("grounded_preview")),
            "lineage": source.get("lineage") or [],
            "supports_insight_refs": [],
        }
        eligible.append(eligible_source)

    insights: list[dict] = []
    supported_by_source: dict[str, list[str]] = {ref: [] for ref in eligible_refs}
    for source in sources:
        if source.get("kind") not in {"insight", "insight_item"}:
            continue
        insight_ref = str(source.get("source_ref") or "")
        relationship_ref = insight_ref.split("#", 1)[0] if source.get("kind") == "insight_item" else insight_ref
        related_refs, _unknown = catalog.expand_preferences([relationship_ref])
        supporting_refs = sorted(eligible_refs & related_refs)
        for ref in supporting_refs:
            supported_by_source[ref].append(insight_ref)
        grounding_document = {
            key: source.get(key)
            for key in (
                "source_ref", "statement", "value", "unit", "time_range", "dimensions", "selection",
                "calculation_trace", "items", "item", "locator",
            )
            if source.get(key) is not None
        }
        insight_description = {
            "source_ref": insight_ref,
            "content": {
                "kind": source.get("kind"),
                "name": source.get("name") or source.get("insight_name"),
                "statement": source.get("statement"),
                "result": source.get("value"),
                "unit": source.get("unit"),
                "calculation_trace": source.get("calculation_trace")
                if source.get("calculation_trace") is not None
                else (source.get("semantic_contract") or {}).get("operation_description"),
                "operands": {
                    key: source.get(key)
                    for key in ("items", "item", "locator", "time_range", "dimensions", "selection")
                    if source.get(key) is not None
                },
                "grounding_document": grounding_document,
                "available_annotation_values": [
                    {key: value for key, value in item.items() if key not in {"field", "compatible_groups"}}
                    for item in grounded_annotation_fields(catalog.resolve(insight_ref))
                ],
            },
            "visual_support": {
                "line_sources": supporting_refs,
                "evidence_refs": source.get("evidence_refs") or [],
                "derived_from": source.get("derived_from") or [],
            },
        }
        insights.append(insight_description)

    for source in eligible:
        source["supports_insight_refs"] = supported_by_source[source["source_ref"]]
    return {
        "chart_scope": "line_chart",
        "reference_contract": {
            "data_source_ref": (
                "Copy an eligible source_ref exactly into dataset.source.$dataset; it resolves the complete materialized source, "
                "not only example_data."
            ),
            "insight_ref": (
                "For an annotation, copy an Insight source_ref and value_id exactly from content.available_annotation_values. "
                "The compiler resolves that opaque ID to the grounded value; never output an internal path."
            ),
        },
        "eligible_line_sources": eligible,
        "insights": insights,
    }


def _field_description(field: str, structural_role: str, field_semantics: dict) -> dict:
    declared = field_semantics.get(field)
    has_declared_meaning = declared is not None and bool(str(declared).strip())
    return {
        "name": field,
        "role": structural_role,
        "meaning": declared if has_declared_meaning else None,
        "meaning_status": "declared" if has_declared_meaning else "not_declared",
    }


def _source_example_data(catalog: PresentationCatalog, source_ref: str, preview) -> dict:
    """Return a bounded, labelled sample while refs retain full-data fidelity."""

    if isinstance(preview, list):
        rows = [dict(row) for row in preview if isinstance(row, dict)]
        origin = "grounded_preview"
    else:
        source = catalog.resolve(source_ref)
        rows = [dict(row) for row in (getattr(source.value, "rows", None) or []) if isinstance(row, dict)]
        origin = "materialized_source"
    if len(rows) <= 4:
        selected = rows
        selection = "all_available_records"
    else:
        selected = [*rows[:2], *rows[-2:]]
        selection = "first_2_and_last_2_records"
    return {
        "sample_only": True,
        "origin": origin,
        "selection": selection,
        "rows": [_compact_sample_row(row) for row in selected],
    }


def _compact_sample_row(row: dict) -> dict:
    visible = [(key, value) for key, value in row.items() if not str(key).startswith("__")]
    return {str(key): _compact_sample_value(value) for key, value in visible[:12]}


def _compact_sample_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, list):
        return [_compact_sample_value(item) for item in value[:8]]
    if isinstance(value, dict):
        return {
            str(key): _compact_sample_value(item)
            for key, item in list(value.items())[:8]
        }
    return str(value)


def _echarts_prompt_inventory(inventory: dict) -> dict[str, list[dict]]:
    """Compatibility projection retained for callers that inspect the old prompt shape."""

    data_sources: list[dict] = []
    insights: list[dict] = []
    for source in inventory.get("sources") or []:
        if not isinstance(source, dict):
            continue
        if source.get("kind") == "data_view":
            data_sources.append({
                key: source.get(key)
                for key in (
                    "source_ref", "name", "shape", "row_count", "schema_fields", "render_contract",
                    "semantic_contract", "lineage", "time_range", "field_semantics", "grounded_preview",
                )
                if source.get(key) is not None
            })
        elif source.get("kind") in {"insight", "insight_item"}:
            insights.append({
                key: source.get(key)
                for key in (
                    "source_ref", "kind", "name", "insight_name", "statement", "value", "unit", "time_range",
                    "dimensions", "selection", "calculation_trace", "item_count", "items", "item", "item_refs",
                    "schema_fields", "locator", "evidence_refs", "derived_from", "semantic_contract",
                )
                if source.get(key) is not None
            })
    return {"data_sources": data_sources, "insights": insights}


def _validate_plan_targets(plan: EChartsPlan, required_ids: list[str]) -> None:
    selected = set(plan.target_insight_ids)
    candidates = set(required_ids)
    unknown = selected - candidates
    if unknown:
        raise _tool_error(
            f"/target_insight_ids: ids must be selected from candidates; unknown={sorted(unknown)}",
            stage="echarts_validation",
        )


def _validate_lineage_coverage(payloads, insight_ids: list[str], inventory: dict, catalog: PresentationCatalog) -> None:
    """Require every distinct renderable derived lineage needed by selected claims to enter a line chart."""

    data_sources = [source for source in inventory.get("sources") or [] if source.get("kind") == "data_view"]
    source_map = {str(source.get("source_ref")): source for source in data_sources}
    candidates_by_lineage: dict[str, set[str]] = {}
    for source in data_sources:
        contract = source.get("render_contract") if isinstance(source.get("render_contract"), dict) else {}
        if int(contract.get("point_count") or 0) < 2 or not contract.get("time_fields") or not contract.get("number_fields"):
            continue
        if _trajectory_alternatives(str(source.get("source_ref")), source, source_map):
            continue
        for lineage_ref in source.get("lineage") or []:
            if str(lineage_ref).startswith("derived_evidence:"):
                candidates_by_lineage.setdefault(str(lineage_ref), set()).add(str(source.get("source_ref")))

    required_lineages: set[str] = set()
    for insight_id in insight_ids:
        try:
            insight = catalog.resolve(f"insight:{insight_id}").value
        except ValueError:
            continue
        for evidence_ref in getattr(insight, "evidence_refs", []) or []:
            if str(getattr(evidence_ref, "source_type", "")) != "derived_evidence":
                continue
            lineage_ref = f"derived_evidence:{getattr(evidence_ref, 'source_id', '')}"
            if lineage_ref in candidates_by_lineage:
                required_lineages.add(lineage_ref)

    if len(required_lineages) < 2:
        return
    selected_refs = {ref for payload in payloads for ref in payload.source_refs}
    missing = sorted(
        lineage_ref
        for lineage_ref in required_lineages
        if not (candidates_by_lineage[lineage_ref] & selected_refs)
    )
    if missing:
        choices = {lineage: sorted(candidates_by_lineage[lineage]) for lineage in missing}
        raise _tool_error(
            f"/source_coverage: line chart omits distinct renderable derived lineages {missing}; add one compatible line from each: {choices}",
            stage="echarts_validation",
        )


def _validate_trajectory_context(payloads, inventory: dict) -> None:
    """Reject endpoint-only lines when a related observation-level trajectory exists."""

    data_sources = {
        str(source.get("source_ref")): source
        for source in inventory.get("sources") or []
        if isinstance(source, dict) and source.get("kind") == "data_view"
    }
    selected_refs = {
        ref
        for payload in payloads
        for ref in payload.source_refs
        if ref in data_sources
    }
    for selected_ref in selected_refs:
        selected = data_sources[selected_ref]
        if int(selected.get("row_count") or 0) > 2:
            continue
        alternatives = _trajectory_alternatives(selected_ref, selected, data_sources)
        if alternatives:
            raise _tool_error(
                f"/series: endpoint-only source '{selected_ref}' cannot represent an observed trajectory; "
                f"use a related multi-observation line source {sorted(alternatives)} and encode endpoints as grounded marks",
                stage="echarts_validation",
            )


def _trajectory_alternatives(selected_ref: str, selected: dict, data_sources: dict[str, dict]) -> list[str]:
    """Find observation-level lines that make a sparse endpoint line semantically misleading."""

    if int(selected.get("row_count") or 0) > 2:
        return []
    contract = selected.get("render_contract") if isinstance(selected.get("render_contract"), dict) else {}
    selected_times = set(contract.get("time_fields") or [])
    selected_numbers = set(contract.get("number_fields") or [])
    selected_lineage = {str(ref) for ref in selected.get("lineage") or [] if str(ref).startswith("evidence:")}
    if not selected_times or not selected_numbers or not selected_lineage:
        return []
    alternatives = []
    for candidate_ref, candidate in data_sources.items():
        candidate_contract = candidate.get("render_contract") if isinstance(candidate.get("render_contract"), dict) else {}
        candidate_lineage = {str(ref) for ref in candidate.get("lineage") or [] if str(ref).startswith("evidence:")}
        if (
            candidate_ref != selected_ref
            and int(candidate.get("row_count") or 0) > 2
            and selected_lineage & candidate_lineage
            and selected_times & set(candidate_contract.get("time_fields") or [])
            and selected_numbers & set(candidate_contract.get("number_fields") or [])
        ):
            alternatives.append(candidate_ref)
    return alternatives


def _repair_context(attempt: int, exc: Exception, language: str | None, previous: list[dict]) -> dict:
    pointer = getattr(exc, "pointer", None)
    error = str(exc)
    if "ECharts planning must return charts or one data request" in error:
        repair_zh = "计划必须二选一：有图表时 required_data_request=null；请求数据时 charts=[]。不得同时返回两者。"
        repair_en = "Choose exactly one branch: charts require required_data_request=null; a data request requires charts=[]."
    elif "/target_insight_ids" in error or "target Insights" in error:
        repair_zh = "将 target_insight_ids 设为 []；Insight 证据由 $value/source_refs 推导，不要让目标元数据阻塞主图。"
        repair_en = "Set target_insight_ids to []; Insight evidence is inferred from $value/source_refs and target metadata must not block the main chart."
    elif "select exactly one record" in error:
        repair_zh = (
            "该 $value 来源有多条记录：读取整体标量结论时使用 record_id=\"scalar\"；读取某个 item 时使用库存中的精确 item_id。"
        )
        repair_en = (
            "The $value source has multiple records: use record_id=\"scalar\" for the parent scalar claim or an exact "
            "inventory item_id for one item."
        )
    elif "inline series.data" in error or "dataset.source must be exactly" in error:
        repair_zh = "删除 series.data；从库存逐字复制一个 data_view source_ref 到 dataset.source 的 $dataset，并用该 view 的精确字段填写 encode。"
        repair_en = "Delete series.data. Copy one data_view source_ref verbatim into dataset.source.$dataset and use exact fields from that view in encode."
    elif "mark data must contain" in error:
        repair_zh = "删除整个无证据 markPoint/markLine/markArea，保留有效的 source-backed dataset 和主 series；不要用字面量替代。"
        repair_en = "Delete the entire ungrounded markPoint/markLine/markArea and keep the valid source-backed dataset and main series; do not substitute literals."
    elif "/option_json: invalid JSON" in error:
        repair_zh = "只返回一个最小 line chart：data_view dataset、time/value 轴和 line series；使用提示中的时序骨架，确保 JSON 字符串闭合且无尾逗号。"
        repair_en = "Return one minimal line chart with data_view datasets, time/value axes, and line series. Use the time-series template and ensure valid closed JSON."
    elif "/source_coverage" in error:
        repair_zh = "当前图漏掉了 Insight lineage 中另一条可渲染时序。按错误列出的精确 view ref 新增 dataset 和兼容量纲的 line，放在同一 time/value 坐标轴；不要删除已有一侧。"
        repair_en = "The chart omits another renderable Insight lineage. Add a dataset and compatible line using each exact view ref listed by the error on the same time/value axes; keep the existing side."
    elif "endpoint-only source" in error:
        repair_zh = "不要把两个计算端点连接成趋势线。使用错误中列出的多观测时序作为主线，并通过 Insight grounding document 的精确坐标把端点画成 markPoint；必要时用 markArea 高亮两点之间的区间。"
        repair_en = "Do not connect two calculated endpoints as a trend line. Use a listed multi-observation source as the primary line and place the endpoints as markPoints from exact Insight grounding-document coordinates; use markArea for the interval when useful."
    elif "main range unreadable" in error:
        repair_zh = "改用 Grounded Input 中已物化过滤或清洗操作的 eligible line source；若 Insight 点与新主线量纲兼容，保留并重新绑定这些标注。"
        repair_en = "Use the eligible line source that materializes the required filtering or cleaning. Preserve and rebind grounded Insight marks when their coordinates remain scale-compatible with the new main line."
    elif "incompatible visual scales" in error or "visually compatible" in error:
        repair_zh = "回到 source-first：保留必需且量纲兼容的 line，删除不同量纲的 series/mark。"
        repair_en = "Return to source-first composition: keep required scale-compatible lines and remove series or marks with another scale."
    elif "duplicate geometry" in error:
        repair_zh = "删除 JSON Pointer 指向的重复 series，且不要用另一条 series 替换；只保留第一个 source-backed 主 series。"
        repair_en = "Delete the duplicate series at the JSON Pointer and do not replace it; keep only the first source-backed main series."
    elif "/type" in error and "series type" in error:
        repair_zh = "删除所有非 line series，只保留绑定多行 time+number data_view 的 line；不要用 bar/scatter 替代。"
        repair_en = "Delete every non-line series. Keep only lines bound to multi-record time+number data views; do not substitute bars or scatter."
    elif "unknown presentation source" in error:
        repair_zh = "不要改写 source_ref。只从库存逐字复制一个可渲染 data_view 的完整 source_ref，并先生成无 mark 的有效主 series。"
        repair_en = "Do not rewrite source refs. Copy one renderable data_view ref verbatim from inventory and first produce a valid unmarked main series."
    elif "typed annotations require an Insight source_ref" in error:
        repair_zh = "annotation 只能引用 insights 中列出的 Insight；不得从 line source 按行号取值。若 Insight 没有完整坐标，删除整个 annotation。"
        repair_en = "Annotations may reference only listed Insights, never line-source rows by position. Remove the entire annotation when no Insight provides its complete coordinate."
    elif "/encode/" in error or "unknown encoded field" in error:
        repair_zh = "从对应 source 的 fields 中逐字复制 x_field 和 y_field；x 必须是 time_coordinate，y 必须是 numeric_measure。"
        repair_en = (
            "Copy x_field and y_field verbatim from the selected source fields; x must be time_coordinate and y must be numeric_measure."
        )
    elif "value_id" in error:
        repair_zh = (
            "从对应 Insight 的 available_annotation_values 逐字复制 value_id；time 位置只能使用 time_*，number 位置只能使用 number_*。"
            "若错误中的对应类型 available value_ids=[]，删除包含它的整个 point/interval/reference annotation，不得改用虚构 ID。"
        )
        repair_en = (
            "Copy value_id verbatim from the Insight's available_annotation_values; time slots require time_* and numeric slots require number_*. "
            "When the error reports available value_ids=[], remove the entire containing point/interval/reference annotation; do not substitute an invented ID."
        )
    else:
        repair_zh = (
            "从头重建一个更小的 typed chart plan，并修正错误指向的问题。删除不必要的 annotation；"
            "series 字段和 annotation value_id 只能从 Grounded Input 逐字复制。不要输出 option_json。"
        )
        repair_en = (
            "Rebuild a smaller typed chart plan from scratch and correct the cited field. Remove unnecessary annotations; "
            "copy every series field and annotation value_id verbatim from Grounded Input. Do not output option_json."
        )
    regenerate_zh = "不要对上一版输出做局部打补丁；请重新读取 Line Chart Grounded Input，并从头生成一个完整、较小且语义一致的 typed plan。"
    regenerate_en = (
        "Do not patch the previous output locally. Re-read Line Chart Grounded Input and regenerate a complete, smaller, "
        "semantically aligned typed plan from scratch."
    )
    return {
        "stage": "echarts_compilation",
        "attempt": attempt + 1,
        "error": error[:2400],
        "json_pointer": pointer,
        "previous_errors": [str(item.get("error") or "") for item in previous],
        "instruction": localized_payload_label(
            language,
            zh=f"{regenerate_zh}{repair_zh}",
            en=f"{regenerate_en} {repair_en}",
        ),
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
