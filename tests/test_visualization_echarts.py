from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.settings import get_settings
from core.visualization import EChartsCompiler, EChartsValidationError, PresentationCatalog, VisualizationArtifactStore
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.analysis import DerivedEvidence
from schemas.database import DatabaseEvidence
from schemas.echarts_plan import EChartsChartPlan, EChartsPlan, StructuredEChartsPlan
from schemas.key_insight import InsightEvidenceRef, InsightItem, KeyInsight
from schemas.state import ConversationStateModel, RequestStateModel
from tools.visualization import (
    VisualizationInput,
    VisualizationTool,
    _echarts_prompt_inventory,
    _line_chart_grounded_input,
    _validate_lineage_coverage,
)


def _state() -> RequestStateModel:
    state = build_request_state(
        ChatRequest(message="Show the complete price trend and peak.", database_context={"database_id": "demo", "database_type": "unit"}),
        get_settings(),
    )
    evidence = DatabaseEvidence(
        evidence_id="evi_prices", result_type="timeseries", database="demo", summary="Complete prices.",
        data={"rows": [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
            {"timestamp": "2026-01-02T00:00:00Z", "value": 12.0},
            {"timestamp": "2026-01-03T00:00:00Z", "value": 15.0},
        ]}, columns=["timestamp", "value"],
    )
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    state.latest_database_evidence = evidence
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_peak", insight_key="peak", name="peak", insight_type="maximum",
        statement="The maximum is 15.", value=15.0, value_shape="collection", method="code_interpreter",
        unit="USD",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_prices")],
        calculation_trace={
            "formula": "max(value)",
            "operands": {"peak": {"timestamp": "2026-01-03T00:00:00Z", "value": 15.0}},
        },
        items=[InsightItem(item_id="peak", label="peak", timestamp="2026-01-03T00:00:00Z", value=15.0)],
    )]
    return state


def _option(**updates):
    option = {
        "useUTC": True,
        "dataset": [{"id": "prices", "source": {"$dataset": "view:evidence:evi_prices:default"}}],
        "xAxis": [{"type": "time"}],
        "yAxis": [{"type": "value", "name": "USD", "scale": True}],
        "series": [{
            "name": "Price", "type": "line", "datasetId": "prices",
            "encode": {"x": "timestamp", "y": "value"}, "showSymbol": False,
            "markPoint": {"data": [{
                "name": "Peak",
                "symbol": "circle", "symbolSize": 12, "label": {"show": False},
                "coord": [
                    {"$value": {"source_ref": "insight:insight_peak", "field": "timestamp"}},
                    {"$value": {"source_ref": "insight:insight_peak", "field": "value"}},
                ],
            }]},
        }],
    }
    option.update(updates)
    return option


def _plan(option=None):
    return EChartsPlan(
        visual_question="Does the complete series support the peak?",
        interpretation="Read the line and highlighted peak.",
        target_insight_ids=["insight_peak"],
        charts=[EChartsChartPlan(
            chart_id="peak", purpose="verify peak", priority="primary", title="Price and peak",
            accessibility_description="A complete price line with its peak.",
            accessibility_table_columns=["timestamp", "value"],
            option_json=json.dumps(option or _option()),
        )],
    )


def _compile(option=None):
    return EChartsCompiler(PresentationCatalog(_state())).compile(_plan(option))[0]


def _assert_provider_objects_are_closed(node, path="$"):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False, path
        for key, value in node.items():
            _assert_provider_objects_are_closed(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_provider_objects_are_closed(value, f"{path}[{index}]")


def test_structured_plan_is_closed_and_does_not_expose_native_option_json():
    schema = StructuredEChartsPlan.model_json_schema()
    _assert_provider_objects_are_closed(schema)
    assert "option_json" not in json.dumps(schema)
    chart = schema["$defs"]["StructuredEChartsChartPlan"]["properties"]
    assert {"series", "point_annotations", "interval_annotations", "reference_lines"} <= set(chart)
    time_ref = schema["$defs"]["EChartsGroundedTimeRef"]["properties"]
    number_ref = schema["$defs"]["EChartsGroundedNumberRef"]["properties"]
    assert "value_id" in time_ref and "field" not in time_ref
    assert "value_id" in number_ref and "field" not in number_ref


def test_typed_annotations_reject_values_from_different_coordinate_groups():
    state = _state()
    state.insight_set.insights[0].items.append(InsightItem(
        item_id="other", label="other", timestamp="2026-01-02T00:00:00Z", value=13.0,
    ))
    chart = EChartsChartPlan.model_validate({
        "chart_id": "peak", "purpose": "verify peak", "priority": "primary",
        "title": "Peak", "accessibility_description": "Peak on complete price line.",
        "series": [{
            "series_id": "prices", "name": "Price",
            "source_ref": "view:evidence:evi_prices:default",
            "x_field": "timestamp", "y_field": "value",
        }],
        "point_annotations": [{
            "series_id": "prices", "name": "Peak",
            "time": {"source_ref": "insight:insight_peak", "value_id": "time_1"},
            "value": {"source_ref": "insight:insight_peak", "value_id": "number_4"},
        }],
    })
    plan = EChartsPlan(
        visual_question="Where is the peak?", interpretation="Read the marked point.", charts=[chart],
    )
    with pytest.raises(EChartsValidationError, match="coordinate_group"):
        EChartsCompiler(PresentationCatalog(state)).compile(plan)


def test_prompt_inventory_separates_renderable_data_from_full_insight_content():
    catalog = PresentationCatalog(_state())
    preferred, unknown = catalog.expand_preferences(["insight:insight_peak"])
    assert not unknown
    prompt_inventory = _echarts_prompt_inventory(catalog.targeted_planner_inventory(preferred))
    assert any(source["source_ref"] == "view:evidence:evi_prices:default" for source in prompt_inventory["data_sources"])
    peak = next(source for source in prompt_inventory["insights"] if source["source_ref"] == "insight:insight_peak")
    assert peak["statement"] == "The maximum is 15." and peak["value"] == 15.0
    assert peak["items"][0]["timestamp"] == "2026-01-03T00:00:00Z"
    assert "query:evi_prices" in peak["evidence_refs"]
    prices = next(source for source in prompt_inventory["data_sources"] if source["source_ref"] == "view:evidence:evi_prices:default")
    assert "evidence:evi_prices" in prices["lineage"]
    assert "projection_root" not in peak


def test_line_chart_grounded_input_exposes_executable_sources_and_calculation_relationships():
    catalog = PresentationCatalog(_state())
    preferred, unknown = catalog.expand_preferences(["insight:insight_peak"])
    assert not unknown
    package = _line_chart_grounded_input(catalog.targeted_planner_inventory(preferred), catalog=catalog)
    assert package["chart_scope"] == "line_chart"
    assert len(package["eligible_line_sources"]) == 1
    line = package["eligible_line_sources"][0]
    assert line["source_ref"] == "view:evidence:evi_prices:default"
    assert package["reference_contract"]["data_source_ref"].startswith("Copy an eligible source_ref")
    assert line["purpose"] == {
        "name": "Complete prices.",
        "data_role": "raw_observations",
        "description": "Database observations materialized from the executed query.",
        "materializes_transformation": False,
        "visual_uses": ["complete_context", "raw_series", "comparison_baseline"],
        "limitations": ["Does not materialize downstream analysis transformations."],
    }
    assert line["scope"]["record_count"] == 3
    assert line["fields"] == [
        {"name": "timestamp", "role": "time_coordinate", "meaning": None, "meaning_status": "not_declared"},
        {"name": "value", "role": "numeric_measure", "meaning": None, "meaning_status": "not_declared"},
    ]
    assert line["example_data"] == {
        "sample_only": True,
        "origin": "materialized_source",
        "selection": "all_available_records",
        "rows": [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
            {"timestamp": "2026-01-02T00:00:00Z", "value": 12.0},
            {"timestamp": "2026-01-03T00:00:00Z", "value": 15.0},
        ],
    }
    assert line["supports_insight_refs"] == ["insight:insight_peak"]
    insight = package["insights"][0]
    assert insight["content"]["statement"] == "The maximum is 15."
    assert insight["content"]["result"] == 15.0
    assert insight["content"]["unit"] == "USD"
    assert insight["content"]["calculation_trace"]["formula"] == "max(value)"
    assert insight["content"]["grounding_document"]["items"][0]["timestamp"] == "2026-01-03T00:00:00Z"
    annotation_values = insight["content"]["available_annotation_values"]
    assert all("field" not in item for item in annotation_values)
    assert {(item["value_id"], item["value_type"], item["description"]) for item in annotation_values} >= {
        ("number_1", "number", "value"),
        ("time_1", "time", "calculation_trace.operands.peak.timestamp"),
        ("number_2", "number", "calculation_trace.operands.peak.value"),
        ("time_2", "time", "items.0.timestamp"),
        ("number_3", "number", "items.0.value"),
    }
    assert insight["visual_support"]["line_sources"] == [
        "view:evidence:evi_prices:default"
    ]


def test_line_chart_grounded_input_bounds_large_source_examples():
    state = _state()
    state.database_evidence_artifacts["evi_prices"].data["rows"] = [
        {"timestamp": f"2026-01-{day:02d}T00:00:00Z", "value": float(day)}
        for day in range(1, 7)
    ]
    catalog = PresentationCatalog(state)
    preferred, _ = catalog.expand_preferences(["insight:insight_peak"])
    package = _line_chart_grounded_input(catalog.targeted_planner_inventory(preferred), catalog=catalog)
    sample = package["eligible_line_sources"][0]["example_data"]
    assert sample["sample_only"] is True
    assert sample["selection"] == "first_2_and_last_2_records"
    assert [row["value"] for row in sample["rows"]] == [1.0, 2.0, 5.0, 6.0]


def test_lineage_coverage_requires_each_distinct_renderable_derived_timeseries():
    state = _state()
    for suffix, offset in (("first", 0), ("second", 10)):
        state.derived_evidence_artifacts[suffix] = DerivedEvidence(
            evidence_id=suffix, name=f"{suffix} interval", shape="timeseries",
            rows=[
                {"timestamp": f"2026-01-{day + offset:02d}T00:00:00Z", "value": float(day)}
                for day in (1, 2, 3)
            ], lineage=["evidence:evi_prices"], transform_summary=f"{suffix} interval records",
        )
        state.insight_set.insights.append(KeyInsight(
            insight_id=f"insight_{suffix}", insight_key=f"mean_{suffix}", name=f"mean {suffix}",
            insight_type="average", statement=f"Mean for {suffix} interval.", value=10.0 + offset,
            method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="derived_evidence", source_id=suffix)],
        ))
    catalog = PresentationCatalog(state)
    preferred, _ = catalog.expand_preferences(["insight:insight_first", "insight:insight_second"])
    inventory = catalog.targeted_planner_inventory(preferred)
    option = _option()
    option["dataset"][0]["source"] = {"$dataset": "view:derived_evidence:first"}
    option["series"][0].pop("markPoint")
    plan = _plan(option).model_copy(update={"target_insight_ids": []})
    payloads = EChartsCompiler(catalog).compile(plan)
    with pytest.raises(Exception, match="/source_coverage"):
        _validate_lineage_coverage(payloads, ["insight_first", "insight_second"], inventory, catalog)

    option["dataset"].append({"id": "second", "source": {"$dataset": "view:derived_evidence:second"}})
    option["series"].append({
        "name": "Second", "type": "line", "datasetId": "second",
        "encode": {"x": "timestamp", "y": "value"},
    })
    payloads = EChartsCompiler(catalog).compile(_plan(option).model_copy(update={"target_insight_ids": []}))
    _validate_lineage_coverage(payloads, ["insight_first", "insight_second"], inventory, catalog)


def test_lineage_coverage_does_not_require_sparse_endpoint_sources_as_lines():
    state = _state()
    for suffix, values in (("endpoints", (10.0, 15.0)), ("window", (10.0, 12.0, 15.0))):
        state.derived_evidence_artifacts[suffix] = DerivedEvidence(
            evidence_id=suffix, name=suffix, shape="timeseries",
            rows=[
                {"timestamp": f"2026-01-0{index}T00:00:00Z", "value": value}
                for index, value in enumerate(values, start=1)
            ], lineage=["evidence:evi_prices"], transform_summary=f"{suffix} records",
        )
        state.insight_set.insights.append(KeyInsight(
            insight_id=f"insight_{suffix}", insight_key=suffix, name=suffix,
            insight_type="interval", statement=suffix, value=list(values), method="code_interpreter",
            evidence_refs=[InsightEvidenceRef(source_type="derived_evidence", source_id=suffix)],
        ))
    catalog = PresentationCatalog(state)
    preferred, _ = catalog.expand_preferences(["insight:insight_endpoints", "insight:insight_window"])
    inventory = catalog.targeted_planner_inventory(preferred)
    option = _option()
    option["dataset"][0]["source"] = {"$dataset": "view:derived_evidence:window"}
    option["series"][0].pop("markPoint")
    payloads = EChartsCompiler(catalog).compile(_plan(option).model_copy(update={"target_insight_ids": []}))
    _validate_lineage_coverage(payloads, ["insight_endpoints", "insight_window"], inventory, catalog)


def test_compiler_injects_complete_records_and_propagates_bindings():
    payload = _compile()
    rows = payload.option["dataset"][0]["source"]
    mark = payload.option["series"][0]["markPoint"]["data"][0]
    assert payload.schema_version == "5" and payload.chart_type == "echarts"
    assert len(rows) == 3 and all("bindingId" in row for row in rows)
    assert mark["coord"] == ["2026-01-03T00:00:00Z", 15.0]
    assert mark["bindingId"]
    assert "insight:insight_peak" in payload.source_refs


def test_data_view_only_chart_does_not_require_an_insight_target():
    option = _option()
    option["series"][0].pop("markPoint")
    plan = _plan(option).model_copy(update={"target_insight_ids": []})
    payload = EChartsCompiler(PresentationCatalog(_state())).compile(plan)[0]
    assert payload.verification.target_insight_ids == []
    assert payload.source_refs == ["view:evidence:evi_prices:default"]


def test_compiler_rejects_supporting_chart_dominated_by_primary():
    primary = _plan()
    supporting_option = _option()
    supporting_option["series"][0].pop("markPoint")
    plan = primary.model_copy(update={
        "charts": [
            primary.charts[0],
            EChartsChartPlan(
                chart_id="redundant", purpose="repeat the same line", priority="supporting",
                title="Repeated line", accessibility_description="The same line again.",
                option_json=json.dumps(supporting_option),
            ),
        ],
    })
    with pytest.raises(EChartsValidationError, match="visually redundant") as error:
        EChartsCompiler(PresentationCatalog(_state())).compile(plan)
    assert error.value.pointer == "/charts/1"


@pytest.mark.parametrize(("mutate", "pointer"), [
    (lambda option: option["dataset"][0].update(source=[{"timestamp": "x", "value": 1}]), "/dataset/0/source"),
    (lambda option: option["dataset"][0].update(transform={"type": "filter"}), "/dataset/0/transform"),
    (lambda option: option["series"][0].update(data=[[1, 2]]), "/series/0/data"),
    (lambda option: option["series"][0].update(type="custom"), "/series/0/type"),
    (lambda option: option["series"][0].update(yAxisIndex=2), "/series/0/yAxisIndex"),
    (lambda option: option["series"][0].update(datasetId="missing"), "/series/0/datasetId"),
    (lambda option: option["series"][0].update(encode={"x": "timestamp", "y": "missing"}), "/series/0/encode/y"),
    (lambda option: option.update(graphic={"image": "https://example.com/x.png"}), "/graphic/image"),
])
def test_compiler_rejects_unsafe_or_ungrounded_options_with_json_pointer(mutate, pointer):
    option = _option()
    mutate(option)
    with pytest.raises(EChartsValidationError) as exc_info:
        _compile(option)
    assert exc_info.value.pointer == pointer


def test_compiler_rejects_duplicate_geometry():
    option = _option()
    option["series"].append({
        "name": "Duplicate", "type": "line", "datasetId": "prices",
        "encode": {"x": "timestamp", "y": "value"},
    })
    with pytest.raises(EChartsValidationError, match="duplicate geometry") as exc_info:
        _compile(option)
    assert exc_info.value.pointer == "/series/1"


def test_value_placeholder_rejects_unknown_field_and_ambiguous_source():
    option = _option()
    option["series"][0]["markPoint"]["data"][0]["coord"][0]["$value"]["field"] = "missing"
    with pytest.raises(EChartsValidationError) as unknown:
        _compile(option)
    assert unknown.value.pointer.endswith("/$value/field")
    option = _option()
    option["series"][0]["markPoint"]["data"][0]["coord"][0] = {
        "$value": {"source_ref": "view:evidence:evi_prices:default", "field": "timestamp"}
    }
    with pytest.raises(EChartsValidationError, match="exactly one record"):
        _compile(option)


def test_value_placeholder_can_select_parent_scalar_when_insight_also_has_items():
    option = _option()
    option["series"][0]["markPoint"]["data"][0]["value"] = {
        "$value": {"source_ref": "insight:insight_peak", "field": "value", "record_id": "scalar"}
    }
    payload = _compile(option)
    assert payload.option["series"][0]["markPoint"]["data"][0]["value"] == 15.0


def test_value_placeholder_resolves_nested_insight_calculation_with_json_pointer():
    option = _option()
    option["series"][0]["markPoint"]["data"][0]["coord"] = [
        {
            "$value": {
                "source_ref": "insight:insight_peak",
                "field": "/calculation_trace/operands/peak/timestamp",
            }
        },
        {
            "$value": {
                "source_ref": "insight:insight_peak",
                "field": "/calculation_trace/operands/peak/value",
            }
        },
    ]
    payload = _compile(option)
    point = payload.option["series"][0]["markPoint"]["data"][0]
    assert point["coord"] == ["2026-01-03T00:00:00Z", 15.0]
    assert point["bindingId"]


def test_value_placeholder_rejects_unknown_json_pointer():
    option = _option()
    option["series"][0]["markPoint"]["data"][0]["coord"][0] = {
        "$value": {"source_ref": "insight:insight_peak", "field": "/calculation_trace/missing"}
    }
    with pytest.raises(EChartsValidationError, match="does not exist") as error:
        _compile(option)
    assert error.value.pointer.endswith("/$value/field")


def test_value_placeholder_json_pointer_must_resolve_to_scalar():
    option = _option()
    option["series"][0]["markPoint"]["data"][0]["coord"][0] = {
        "$value": {"source_ref": "insight:insight_peak", "field": "/calculation_trace/operands/peak"}
    }
    with pytest.raises(EChartsValidationError, match="must resolve to one scalar value"):
        _compile(option)


@pytest.mark.parametrize("series_type", ["bar", "scatter"])
def test_compiler_is_line_chart_only(series_type):
    option = _option()
    option["series"][0]["type"] = series_type
    with pytest.raises(EChartsValidationError, match=r"\['line'\]") as error:
        _compile(option)
    assert error.value.pointer == "/series/0/type"


def test_compiler_rejects_unmatched_legend_and_incompatible_shared_scale():
    option = _option(legend={"data": ["Missing"]})
    with pytest.raises(EChartsValidationError) as legend_error:
        _compile(option)
    assert legend_error.value.pointer == "/legend/data/0"
    state = _state()
    state.derived_evidence_artifacts["tiny"] = DerivedEvidence(
        evidence_id="tiny", name="Tiny values", shape="timeseries",
        rows=[
            {"timestamp": "2026-01-01T00:00:00Z", "tiny": 1.0},
            {"timestamp": "2026-01-02T00:00:00Z", "tiny": 2.0},
            {"timestamp": "2026-01-03T00:00:00Z", "tiny": 3.0},
        ], lineage=["evidence:evi_prices"], transform_summary="Tiny comparison values.",
    )
    option = _option()
    option["dataset"].append({
        "id": "tiny", "source": {"$dataset": "view:derived_evidence:tiny"},
    })
    option["series"].append({
        "name": "Tiny", "type": "line", "datasetId": "tiny", "encode": {"x": "timestamp", "y": "tiny"},
    })
    with pytest.raises(EChartsValidationError, match="incompatible visual scales"):
        EChartsCompiler(PresentationCatalog(state)).compile(_plan(option))


def test_compiler_preserves_finite_extreme_values_and_rejects_wrong_time_field():
    state = _state()
    state.database_evidence_artifacts["evi_prices"].data["rows"][0]["value"] = 1_000_000.0
    option = _option()
    payload = EChartsCompiler(PresentationCatalog(state)).compile(_plan(option))[0]
    assert payload.option["dataset"][0]["source"][0]["value"] == 1_000_000.0
    assert payload.warnings == [
        "This series contains extreme values; the main range may appear compressed on the linear y-axis."
    ]

    option = _option()
    option["series"][0]["encode"]["x"] = "bindingId"
    with pytest.raises(EChartsValidationError, match="time axis") as time_error:
        _compile(option)
    assert time_error.value.pointer == "/series/0/encode/x"


def test_compiler_rejects_markpoint_outside_series_visual_scale():
    state = _state()
    state.insight_set.insights.append(KeyInsight(
        insight_id="insight_unrelated", insight_key="unrelated", name="unrelated",
        insight_type="point_value", statement="An unrelated magnitude.", value=1_000_000.0,
        method="code_interpreter",
    ))
    option = _option()
    option["series"][0]["markPoint"]["data"][0]["coord"][1] = {
        "$value": {"source_ref": "insight:insight_unrelated", "field": "value", "record_id": "scalar"}
    }
    with pytest.raises(EChartsValidationError, match="markPoint y coordinate") as mark_error:
        EChartsCompiler(PresentationCatalog(state)).compile(_plan(option))
    assert mark_error.value.pointer.endswith("/coord/1")


def test_artifact_descriptor_strips_sources_and_full_read_restores_them(tmp_path):
    store = VisualizationArtifactStore(tmp_path)
    descriptor = store.put(_compile())
    assert descriptor.option["dataset"][0]["source"] == []
    assert descriptor.accessibility.table_rows == []
    complete = store.get(descriptor.visualization_id)
    assert complete is not None
    assert len(complete.option["dataset"][0]["source"]) == 3


def test_old_v4_artifact_is_unavailable_and_historical_state_filters_it(tmp_path):
    old = {"schema_version": "4", "chart_type": "line", "visualization_id": "old"}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    assert VisualizationArtifactStore(tmp_path).get("old") is None and path.exists()
    conversation = ConversationStateModel.model_validate({"conversation_id": "c", "recent_visualizations": [old]})
    assert conversation.recent_visualizations == []


class _RepairingLlm:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        prompt = str(messages[0][1])
        self.calls.append(prompt)
        payload = {
            "visual_question": "Does the complete series support the peak?",
            "interpretation": "Read the line and highlighted peak.",
            "target_insight_ids": [],
            "charts": [{
                "chart_id": "peak",
                "purpose": "verify peak",
                "priority": "primary",
                "title": "Price and peak",
                "summary": "Complete price context with its peak.",
                "accessibility_description": "A complete price line with its peak.",
                "accessibility_table_columns": ["timestamp", "value"],
                "series": [{
                    "series_id": "prices",
                    "name": "Price",
                    "source_ref": "view:evidence:evi_prices:default",
                    "x_field": "timestamp",
                    "y_field": "missing" if len(self.calls) == 1 else "value",
                }],
                "point_annotations": [{
                    "series_id": "prices",
                    "name": "Peak",
                    "time": {"source_ref": "insight:insight_peak", "value_id": "time_2"},
                    "value": {"source_ref": "insight:insight_peak", "value_id": "number_3"},
                }],
                "interval_annotations": [],
                "reference_lines": [],
                "y_axis_name": "USD",
            }],
            "required_data_request": None,
        }
        return SimpleNamespace(content=json.dumps(payload), response_metadata={})


@pytest.mark.asyncio
async def test_tool_repairs_unknown_typed_field_using_precise_pointer_and_publishes(tmp_path):
    llm = _RepairingLlm()
    result = await VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path)).execute(
        VisualizationInput(message="Show the price trend and peak.", source_refs=["insight:insight_peak"]),
        request_state=_state(),
    )
    assert result["status"] == "created" and len(llm.calls) == 2
    assert "/series/0/encode/y" in llm.calls[1]
    assert "Line Chart Grounded Input" in llm.calls[1]
    assert "GROUNDED INPUT GUIDE" in llm.calls[1]
    assert '"reference_contract"' in llm.calls[1]
    assert '"example_data"' in llm.calls[1]
    assert '"semantic_description"' not in llm.calls[1]
    assert '"meaning_status": "not_declared"' in llm.calls[1]
    assert '"available_annotation_values"' in llm.calls[1]
    assert "never output a JSON path" in llm.calls[1]
    assert "point_annotations" in llm.calls[1]
    assert "regenerate a complete" in llm.calls[1] and "typed plan" in llm.calls[1]
    complete = VisualizationArtifactStore(tmp_path).get(result["visualization_ids"][0])
    assert complete is not None and complete.option["yAxis"] == {"type": "value", "scale": True, "name": "USD"}
    assert "title" not in complete.option
    assert complete.option["legend"] == {"show": False, "data": ["Price"]}
    assert complete.option["series"][0]["markPoint"]["itemStyle"] == {"color": "#ee6666"}
    assert complete.option["tooltip"] == {"show": True, "trigger": "axis"}
    assert complete.option["dataZoom"] == [
        {"type": "inside", "xAxisIndex": 0},
        {"type": "slider", "show": True, "xAxisIndex": 0, "bottom": 12, "height": 22},
    ]
    assert complete.option["grid"]["bottom"] == 84
