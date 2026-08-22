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
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "Create one native Apache ECharts 5 line chart that answers the user's question from Line Chart Grounded Input. "
            "Make the analytical conclusion visually verifiable; do not display unrelated sources merely because they exist.\n\n"
            "SOURCE AND INSIGHT USE\n"
            "Choose the smallest set of eligible_line_sources needed for the answer. Each source already states its exact time fields, "
            "numeric fields, transformation semantics, and the Insights it supports. Use an Insight's supporting_line_sources as the "
            "visual evidence for that computed claim. The calculation object contains the grounded result, calculation trace, operands, "
            "and locators; do not recalculate or invent any value. Prefer one time axis, one value axis, and one primary line. Add another "
            "line only when a comparison requires a distinct compatible time series, and cover every compared operand. Put scalar results "
            "in the chart title, summary, tooltip, or grounded annotations. A scalar with a different unit from the line stays in text and "
            "must not become a y-axis value.\n\n"
            "ECHARTS OUTPUT CONTRACT\n"
            "Return one closed EChartsPlan with one primary chart and compact valid option_json. Only series.type=\"line\" is allowed. "
            "For every line, set dataset.source exactly to {\"$dataset\":\"EXACT_SOURCE_REF\"}, bind exact exposed field names in encode, "
            "use xAxis.type=\"time\", yAxis.type=\"value\", yAxis.scale=true, series.showSymbol=false, and useUTC=true. Omit legend.data "
            "for one line; with multiple lines, names must be concise, distinct, and exactly match legend entries. Never emit dataset source "
            "arrays, series.data, transforms, functions, executable content, HTML, DOM, URLs, external assets, or invented source refs.\n"
            "Minimal shape: {\"dataset\":{\"source\":{\"$dataset\":\"VIEW_REF\"}},\"xAxis\":{\"type\":\"time\"},"
            "\"yAxis\":{\"type\":\"value\",\"scale\":true},\"series\":{\"name\":\"LABEL\",\"type\":\"line\","
            "\"showSymbol\":false,\"encode\":{\"x\":\"TIME_FIELD\",\"y\":\"NUMBER_FIELD\"}}}.\n\n"
            "GROUNDED ANNOTATIONS\n"
            "Annotations are optional. Add them only when the Insight exposes every exact coordinate. Every semantic mark coordinate or "
            "value must use {\"$value\":{\"source_ref\":\"INSIGHT_REF\",\"field\":\"FIELD_OR_JSON_POINTER\"}}. A field beginning with / "
            "is a JSON Pointer into the Insight grounding document shown in calculation.grounding_document. Existing item_id and "
            "record_id=\"scalar\" selectors may be used for flat records. For markPoint use symbol=\"circle\", symbolSize 8-16, and "
            "label.show=false. If exact coordinates are unavailable, omit the mark and retain the valid supporting line.\n\n"
            "DEPENDENCY AND RETRY\n"
            "Set target_insight_ids=[]. If no eligible source can visually support the requested calculation, return required_data_request "
            "for code_interpreter to materialize a multi-record time+number derived series from the stated lineage. With a chart, "
            "required_data_request must be null. On retry, regenerate the complete option from the grounded input and validation findings; "
            "do not mechanically patch the previous JSON. Keep all visible text in the requested response language.\n"
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


def _line_chart_grounded_input(inventory: dict, *, catalog: PresentationCatalog) -> dict[str, list[dict] | str]:
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
        eligible.append({
            "source_ref": source_ref,
            "name": source.get("name"),
            "shape": source.get("shape"),
            "row_count": source.get("row_count"),
            "time_fields": [
                {"field": field, "meaning": field_semantics.get(field)} for field in time_fields
            ],
            "numeric_fields": [
                {"field": field, "meaning": field_semantics.get(field)} for field in number_fields
            ],
            "time_range": source.get("time_range"),
            "transformation": source.get("semantic_contract"),
            "lineage": source.get("lineage") or [],
            "grounded_preview": source.get("grounded_preview"),
            "supports_insight_refs": [],
        })

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
        insights.append({
            "source_ref": insight_ref,
            "kind": source.get("kind"),
            "name": source.get("name") or source.get("insight_name"),
            "statement": source.get("statement"),
            "calculation": {
                "result": source.get("value"),
                "unit": source.get("unit"),
                "trace": source.get("calculation_trace")
                if source.get("calculation_trace") is not None
                else (source.get("semantic_contract") or {}).get("operation_description"),
                "operands": {
                    key: source.get(key)
                    for key in ("items", "item", "locator", "time_range", "dimensions", "selection")
                    if source.get(key) is not None
                },
                "grounding_document": grounding_document,
            },
            "supporting_line_sources": supporting_refs,
            "evidence_refs": source.get("evidence_refs") or [],
            "derived_from": source.get("derived_from") or [],
        })

    for source in eligible:
        source["supports_insight_refs"] = supported_by_source[source["source_ref"]]
    return {
        "chart_scope": "line_chart",
        "eligible_line_sources": eligible,
        "insights": insights,
    }


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
    candidates_by_lineage: dict[str, set[str]] = {}
    for source in data_sources:
        contract = source.get("render_contract") if isinstance(source.get("render_contract"), dict) else {}
        if int(contract.get("point_count") or 0) < 2 or not contract.get("time_fields") or not contract.get("number_fields"):
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
    elif "incompatible visual scales" in error or "main range unreadable" in error or "visually compatible" in error:
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
    else:
        repair_zh = (
            "从头重建一个更小的完整 option，并修正 JSON Pointer 指向的问题。删除所有不必要的 mark；"
            "mark 数据中的时间和数值只能来自 $value，不能复制预览值。确保 option_json 本身是无注释、无尾随逗号的合法紧凑 JSON。"
        )
        repair_en = (
            "Rebuild a smaller complete option from scratch and correct the cited JSON Pointer. Remove unnecessary marks; "
            "every time or number in mark data must come from $value, never copied preview values. Ensure option_json is "
            "compact valid JSON without comments or trailing commas."
        )
    regenerate_zh = "不要对上一版 JSON 做局部打补丁；请重新读取 Line Chart Grounded Input，并从头生成一个完整、较小且语义一致的 option。"
    regenerate_en = (
        "Do not patch the previous JSON locally. Re-read Line Chart Grounded Input and regenerate a complete, smaller, "
        "semantically aligned option from scratch."
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
