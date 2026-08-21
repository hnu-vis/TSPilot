from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.settings import get_settings
from core.visualization import LineChartCompiler, LineChartValidator, PresentationCatalog, VisualizationArtifactStore
from core.visualization.planning_schema import build_linechart_response_schema
from runtime.request_state import build_request_state
from schemas.api import ChatRequest
from schemas.analysis import DerivedEvidence
from schemas.database import DatabaseEvidence
from schemas.key_insight import InsightEvidenceRef, InsightItem, KeyInsight
from schemas.linechart_plan import (
    AnnotationPlan,
    ChartAnnotationTargetPlan,
    LineChartGoalPlan,
    LineChartPlan,
    LineChartYAxisPlan,
    LinePlan,
    PointPlan,
    ReferenceLinePlan,
    StructuredLineChartPlan,
    StructuredVisualContentPlan,
    VisualContentGoal,
    VisualContentItem,
    VisualContentPlan,
)
from tools.visualization import VisualizationInput, VisualizationTool


def _state():
    state = build_request_state(
        ChatRequest(message="Show the price trend and important point.", database_context={"database_id": "demo", "database_type": "unit"}),
        get_settings(),
    )
    evidence = DatabaseEvidence(
        evidence_id="evi_prices", result_type="timeseries", database="demo", summary="Complete prices.",
        data={"rows": [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
            {"timestamp": "2026-01-02T00:00:00Z", "value": 12.0},
            {"timestamp": "2026-01-03T00:00:00Z", "value": 15.0},
        ]},
        columns=["timestamp", "value"],
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


def _content():
    return VisualContentPlan(
        visual_question="Does the complete series support the peak conclusion?",
        interpretation="Read the line and the highlighted peak.",
        target_insight_ids=["insight_peak"],
        goals=[VisualContentGoal(
            goal_id="primary", purpose="Verify the trend and peak", title="Price and peak", priority="primary",
            host_source_ref="view:evidence:evi_prices:default",
            required_interactions=["tooltip", "zoom", "evidence_link"],
            content=[
                VisualContentItem(content_id="history", source_ref="view:evidence:evi_prices:default", purpose="complete history", importance="primary"),
                VisualContentItem(content_id="peak", source_ref="insight:insight_peak", insight_ids=["insight_peak"], purpose="located peak", importance="highlight"),
            ],
        )],
    )


def _plan(y_field="value"):
    return LineChartPlan(charts=[LineChartGoalPlan(
        goal_id="primary", x_axis_type="time", x_axis_label="Time",
        y_axes=[LineChartYAxisPlan(axis_id="price", measure="price")],
        host_line=LinePlan(
            content_id="history", role="history", importance="primary",
            x_field="timestamp", y_field=y_field, y_axis_id="price",
        ),
        points=[PointPlan(
            content_id="peak", role="peak", importance="highlight",
            x_field="timestamp", y_field="value", y_axis_id="price",
        )],
        zoom_enabled=True,
    )])


def _assert_provider_objects_are_closed(node, path="$"):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            assert node.get("additionalProperties") is False, path
        for key, value in node.items():
            _assert_provider_objects_are_closed(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _assert_provider_objects_are_closed(value, f"{path}[{index}]")


@pytest.mark.parametrize("schema", [StructuredVisualContentPlan, StructuredLineChartPlan])
def test_llm_response_format_contains_only_closed_objects(schema):
    _assert_provider_objects_are_closed(schema.model_json_schema())


def test_dynamic_linechart_schema_closes_content_fields_and_omits_component_source():
    inventory = PresentationCatalog(_state()).planner_inventory([])
    schema = build_linechart_response_schema(_content(), inventory)
    valid = _plan().model_dump(mode="json")
    schema.model_validate(valid)

    invalid_field = _plan("invented_value").model_dump(mode="json")
    with pytest.raises(ValueError):
        schema.model_validate(invalid_field)

    invalid_source = _plan().model_dump(mode="json")
    invalid_source["charts"][0]["host_line"]["source_ref"] = "view:invented"
    with pytest.raises(ValueError):
        schema.model_validate(invalid_source)


def test_dynamic_linechart_schema_excludes_scalar_lines_and_categorical_points():
    content = VisualContentPlan(
        visual_question="How does the series relate to the summary?",
        interpretation="Use the time series as the line and the scalar as chart context.",
        goals=[VisualContentGoal(
            goal_id="primary", purpose="Show series and summary", title="Series", priority="primary",
            host_source_ref="view:series",
            content=[
                VisualContentItem(content_id="series", source_ref="view:series", purpose="history", importance="primary"),
                VisualContentItem(content_id="summary", source_ref="view:summary", purpose="summary", importance="support"),
            ],
        )],
    )
    inventory = {"sources": [
        {
            "source_ref": "view:series", "row_count": 3,
            "schema_fields": [
                {"name": "timestamp", "data_type": "datetime"},
                {"name": "value", "data_type": "number"},
            ],
        },
        {
            "source_ref": "view:summary", "kind": "insight", "item_count": 0,
            "value": {"period": "first half", "average": 11.0},
            "render_capabilities": {"scalar_only": False},
            "schema_fields": [
                {"name": "period", "data_type": "string"},
                {"name": "average", "data_type": "number"},
            ],
        },
    ]}
    schema = build_linechart_response_schema(content, inventory)
    _assert_provider_objects_are_closed(schema.model_json_schema())
    plan = _plan().model_dump(mode="json")
    plan["charts"][0]["host_line"]["content_id"] = "series"
    plan["charts"][0]["points"] = []
    schema.model_validate(plan)

    scalar_line = json.loads(json.dumps(plan))
    scalar_line["charts"][0]["host_line"].update(
        {"content_id": "summary", "x_field": "period", "y_field": "average"}
    )
    with pytest.raises(ValueError):
        schema.model_validate(scalar_line)

    categorical_point = json.loads(json.dumps(plan))
    categorical_point["charts"][0]["points"] = [{
        "content_id": "summary", "role": "summary", "importance": "support",
        "label": None, "x_field": "period", "y_field": "average", "y_axis_id": "price",
    }]
    with pytest.raises(ValueError):
        schema.model_validate(categorical_point)


def test_linechart_compiler_preserves_records_bindings_and_insight_coverage():
    payload = LineChartCompiler(PresentationCatalog(_state())).compile(_content(), _plan())[0]

    assert payload.schema_version == "4"
    assert payload.chart_type == "line"
    assert len(payload.data_views[0].records) == 3
    assert payload.lines[0].role == "history"
    assert payload.points[0].role == "peak"
    assert payload.verification.target_insight_ids == ["insight_peak"]
    assert payload.bindings[0].insight_id == "insight_peak"


def test_linechart_validator_rejects_single_point_line():
    state = _state()
    state.database_evidence_artifacts["evi_prices"].data["rows"] = [
        {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
    ]
    with pytest.raises(ValueError, match="at least two grounded points"):
        LineChartCompiler(PresentationCatalog(state)).compile(_content(), _plan())


def test_chart_annotation_cannot_masquerade_as_a_located_point():
    plan = _plan()
    plan.charts[0].annotations = [AnnotationPlan(
        content_id="peak", role="peak note", importance="highlight",
        content_field="label",
        target=ChartAnnotationTargetPlan(target_type="chart"),
    )]
    plan.charts[0].points = []
    payload = LineChartCompiler(PresentationCatalog(_state())).compile(_content(), plan)[0]
    LineChartValidator().validate(payload)
    assert payload.annotations[0].target.target_type == "chart"


def test_reference_line_requires_one_grounded_scalar():
    plan = _plan()
    plan.charts[0].reference_lines = [ReferenceLinePlan(
        content_id="history", role="reference", importance="support",
        value_field="value", y_axis_id="price",
    )]
    with pytest.raises(ValueError, match="exactly one grounded scalar"):
        LineChartCompiler(PresentationCatalog(_state())).compile(_content(), plan)


def test_incompatible_measure_can_be_a_supporting_linechart():
    state = _state()
    state.derived_evidence_artifacts["score"] = DerivedEvidence(
        evidence_id="score", name="Anomaly score", shape="timeseries",
        rows=[
            {"timestamp": "2026-01-01T00:00:00Z", "score": 0.1},
            {"timestamp": "2026-01-02T00:00:00Z", "score": 2.5},
        ],
        lineage=["evidence:evi_prices"], transform_summary="Detector-owned scores.",
    )
    content = _content()
    score_ref = "view:derived_evidence:score"
    content.goals.append(VisualContentGoal(
        goal_id="score", purpose="show incompatible score", title="Anomaly score", priority="supporting",
        host_source_ref=score_ref,
        content=[VisualContentItem(content_id="score_series", source_ref=score_ref, purpose="score", importance="support")],
    ))
    plan = _plan()
    plan.charts.append(LineChartGoalPlan(
        goal_id="score", x_axis_type="time",
        y_axes=[LineChartYAxisPlan(axis_id="score", measure="anomaly_score")],
        host_line=LinePlan(
            content_id="score_series", role="score", importance="support",
            x_field="timestamp", y_field="score", y_axis_id="score",
        ),
    ))
    payloads = LineChartCompiler(PresentationCatalog(state)).compile(content, plan)
    assert [item.priority for item in payloads] == ["primary", "supporting"]


class _TwoStageLlm:
    def __init__(self, *, invalid_first=False, needs_sources=False):
        self.calls = []
        self.invalid_first = invalid_first
        self.needs_sources = needs_sources
        self.composition_calls = 0

    async def ainvoke(self, messages):
        prompt = str(messages[0][1])
        self.calls.append(prompt)
        if "visual-content planner" in prompt:
            if self.needs_sources:
                payload = {
                    "visual_question": None, "interpretation": None, "target_insight_ids": [], "goals": [],
                    "required_data_request": {
                        "required_action": "code_interpreter", "purpose": "derive a visual series",
                        "message": None, "required_shape": "timeseries", "required_fields": ["timestamp", "value"],
                        "required_properties": [], "input_evidence": None,
                        "input_source_refs": ["evidence:evi_prices"],
                        "insight_requests": [{"name": "derived series", "insight_type": "series", "insight_key": "derived_series"}],
                    },
                }
            else:
                payload = _content().model_dump(mode="json")
        else:
            self.composition_calls += 1
            payload = _plan("missing" if self.invalid_first and self.composition_calls == 1 else "value").model_dump(mode="json")
        return SimpleNamespace(content=json.dumps(payload), response_metadata={})


@pytest.mark.asyncio
async def test_visualization_normal_path_uses_exactly_two_llm_calls_and_persists(tmp_path):
    llm = _TwoStageLlm()
    store = VisualizationArtifactStore(tmp_path)
    result = await VisualizationTool(llm=llm, artifact_store=store).execute(
        VisualizationInput(message="Show the price trend and peak.", source_refs=["insight:insight_peak"]),
        request_state=_state(),
    )

    assert result["status"] == "created"
    assert len(llm.calls) == 2
    descriptor = result["visualizations"][0]
    assert descriptor["data_views"][0]["records"] == []
    complete = store.get(result["visualization_ids"][0])
    assert complete is not None
    assert len(complete.data_views[0].records) == 3


@pytest.mark.asyncio
async def test_visualization_repairs_only_the_composition_stage(tmp_path):
    llm = _TwoStageLlm(invalid_first=True)
    result = await VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path)).execute(
        VisualizationInput(message="Show the price trend and peak.", source_refs=["insight:insight_peak"]),
        request_state=_state(),
    )
    assert result["status"] == "created"
    assert len(llm.calls) == 3
    assert llm.composition_calls == 2
    assert "Validation repair context" in llm.calls[-1]


@pytest.mark.asyncio
async def test_visualization_returns_needs_sources_before_composition(tmp_path):
    llm = _TwoStageLlm(needs_sources=True)
    result = await VisualizationTool(llm=llm, artifact_store=VisualizationArtifactStore(tmp_path)).execute(
        VisualizationInput(message="Show a derived series.", source_refs=["evidence:evi_prices"]),
        request_state=_state(),
    )
    assert result["status"] == "needs_sources"
    assert result["required_data_request"]["required_action"] == "code_interpreter"
    assert len(llm.calls) == 1


def test_artifact_store_rejects_old_v4_without_deleting_it(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": "4", "visualization_id": "old", "marks": []}), encoding="utf-8")
    assert VisualizationArtifactStore(tmp_path).get("old") is None
    assert path.exists()
