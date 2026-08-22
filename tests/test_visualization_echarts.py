from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.settings import get_settings
from core.visualization import EChartsCompiler, EChartsValidationError, PresentationCatalog, VisualizationArtifactStore
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.echarts_plan import EChartsChartPlan, EChartsPlan, StructuredEChartsPlan
from schemas.key_insight import InsightEvidenceRef, InsightItem, KeyInsight
from schemas.state import ConversationStateModel, RequestStateModel
from tools.visualization import VisualizationInput, VisualizationTool


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
        statement="The maximum is 15.", value_shape="collection", method="code_interpreter",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_prices")],
        items=[InsightItem(item_id="peak", label="peak", timestamp="2026-01-03T00:00:00Z", value=15.0)],
    )]
    return state


def _option(**updates):
    option = {
        "useUTC": True,
        "dataset": [{"id": "prices", "source": {"$dataset": "view:evidence:evi_prices:default"}}],
        "xAxis": [{"type": "time"}],
        "yAxis": [{"type": "value", "name": "USD"}],
        "series": [{
            "name": "Price", "type": "line", "datasetId": "prices",
            "encode": {"x": "timestamp", "y": "value"}, "showSymbol": False,
            "markPoint": {"data": [{
                "name": "Peak",
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


def test_structured_plan_is_closed_while_option_is_a_string():
    schema = StructuredEChartsPlan.model_json_schema()
    _assert_provider_objects_are_closed(schema)
    option_property = schema["$defs"]["EChartsChartPlan"]["properties"]["option_json"]
    assert option_property["type"] == "string"


def test_compiler_injects_complete_records_and_propagates_bindings():
    payload = _compile()
    rows = payload.option["dataset"][0]["source"]
    mark = payload.option["series"][0]["markPoint"]["data"][0]
    assert payload.schema_version == "5" and payload.chart_type == "echarts"
    assert len(rows) == 3 and all("bindingId" in row for row in rows)
    assert mark["coord"] == ["2026-01-03T00:00:00Z", 15.0]
    assert mark["bindingId"]
    assert "insight:insight_peak" in payload.source_refs


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
        option = _option()
        if len(self.calls) == 1:
            option["series"][0]["yAxisIndex"] = 3
        payload = _plan(option).model_dump(mode="json")
        payload["required_data_request"] = None
        return SimpleNamespace(content=json.dumps(payload), response_metadata={})


@pytest.mark.asyncio
async def test_tool_repairs_unknown_axis_using_precise_pointer_and_publishes(tmp_path):
    llm = _RepairingLlm()
    result = await VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path)).execute(
        VisualizationInput(message="Show the price trend and peak.", source_refs=["insight:insight_peak"]),
        request_state=_state(),
    )
    assert result["status"] == "created" and len(llm.calls) == 2
    assert "/series/0/yAxisIndex" in llm.calls[1]
    complete = VisualizationArtifactStore(tmp_path).get(result["visualization_ids"][0])
    assert complete is not None and len(complete.option["yAxis"]) == 1
