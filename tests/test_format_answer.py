from __future__ import annotations

import asyncio

import pytest

from app.settings import get_settings
from core.visualization import PresentationCatalog, VisualizationArtifactStore, VisualizationMaterializer
from runtime.request_state import build_request_state
from schemas.analysis import AnalysisResult
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.key_insight import KeyInsight, InsightEvidenceRef
from schemas.output import FinalResponsePlan, PlannedAnswerSection, VisualGoal, VisualLayerPlan
from tools.base import StructuredToolError
from tools.format_answer import FormatAnswerInput, FormatAnswerTool


class _ForbiddenLlm:
    async def ainvoke(self, _messages):
        raise AssertionError("format_answer must not invoke an LLM")


def _state():
    state = build_request_state(
        ChatRequest(message="show recent prices", database_context={"database_id": "demo", "database_type": "unit"}),
        get_settings(),
    )
    evidence = DatabaseEvidence(
        evidence_id="evi_prices", result_type="timeseries", database="demo", query_language="unit",
        query="unit:prices", summary="Loaded three prices.",
        data={"rows": [
            {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
            {"timestamp": "2026-01-02T00:00:00Z", "value": 12.0},
            {"timestamp": "2026-01-03T00:00:00Z", "value": 15.0},
        ], "time_field": "timestamp", "value_field": "value"},
        columns=["timestamp", "value"],
    )
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    state.latest_database_evidence = evidence
    return state


def _trend_goal(source_ref="view:evidence:evi_prices:default"):
    return VisualGoal(
        purpose="show the observed trend", title="Price trend", required_roles=["series"],
        layers=[VisualLayerPlan(role="series", source_ref=source_ref, mark="line", encoding={"x": "timestamp", "y": "value"})],
    )


def _attach_trend_visualization(state, tmp_path):
    complete = VisualizationMaterializer(state).materialize(_trend_goal())
    descriptor = VisualizationArtifactStore(tmp_path).put(complete)
    state.visualizations = [descriptor]
    return descriptor


def test_format_answer_references_existing_visualization_without_second_llm_call(tmp_path):
    state = _state()
    descriptor = _attach_trend_visualization(state, tmp_path)
    plan = FinalResponsePlan(
        title="Recent prices", summary="The latest value is 15.",
        sections=[PlannedAnswerSection(section_type="analysis", heading="Trend", content="Prices rose.", source_refs=["evidence:evi_prices"])],
        visualization_ids=[descriptor.visualization_id],
    )
    result = asyncio.run(FormatAnswerTool(llm=_ForbiddenLlm()).execute(FormatAnswerInput(response_plan=plan), request_state=state))

    visualization = result["visualizations"][0]
    assert visualization["schema_version"] == "3"
    assert visualization["data_ref"]
    assert visualization["layers"][0]["mark"] == "line"
    assert visualization["datasets"][0]["series"] == []
    assert visualization["datasets"][0]["source_ref"] == "view:evidence:evi_prices:default"
    assert result["claims"][0]["visualization_ids"] == [visualization["visualization_id"]]


def test_format_answer_accepts_selected_visualization_as_section_source(tmp_path):
    state = _state()
    descriptor = _attach_trend_visualization(state, tmp_path)
    plan = FinalResponsePlan(
        summary="The chart is ready.",
        sections=[PlannedAnswerSection(
            section_type="visualization",
            content="The full trend is shown in the chart.",
            source_refs=[descriptor.visualization_id],
        )],
        visualization_ids=[descriptor.visualization_id],
    )

    result = asyncio.run(FormatAnswerTool().execute(
        FormatAnswerInput(response_plan=plan),
        request_state=state,
    ))

    assert result["sections"][0]["structured_payload"]["source_refs"] == [
        f"visualization:{descriptor.visualization_id}"
    ]
    assert result["claims"][0]["visualization_ids"] == [descriptor.visualization_id]


def test_planner_inventory_contains_marks_views_and_no_renderer_templates():
    inventory = PresentationCatalog(_state()).planner_inventory()
    assert "line" in inventory["marks"] and "table" in inventory["marks"]
    assert any(item["source_ref"] == "view:evidence:evi_prices:default" for item in inventory["sources"])
    assert "templates" not in inventory


def test_format_answer_routes_unknown_visualization_id_to_visualization_tool():
    state = _state()
    plan = FinalResponsePlan(summary="Answer", visualization_ids=["viz_missing"])
    with pytest.raises(StructuredToolError) as caught:
        asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    failure = caught.value.validation_failure
    assert failure["capability"] == "visualization"
    assert failure["retry_policy"]["max_equivalent_retries"] == 1
    assert failure["retry_policy"]["required_action"] == "visualization"


def test_format_answer_canonicalizes_semantic_insight_key_reference():
    state = _state()
    state.insight_set.insights = [KeyInsight(
        insight_id="insight_max_1",
        insight_key="max_value",
        name="Maximum value",
        insight_type="extreme",
        statement="Maximum value is 15.",
        value=15.0,
        method="sql_query",
        evidence_refs=[InsightEvidenceRef(source_type="query", source_id="evi_prices")],
    )]
    plan = FinalResponsePlan(
        summary="Maximum value is 15.",
        sections=[PlannedAnswerSection(
            section_type="insight",
            content="Maximum value is 15.",
            source_refs=["insight:max_value"],
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(
        FormatAnswerInput(response_plan=plan),
        request_state=state,
    ))

    assert result["sections"][0]["structured_payload"]["source_refs"] == ["insight:insight_max_1"]
    assert result["claims"][0]["insight_ids"] == ["insight_max_1"]


def test_format_answer_routes_unknown_prose_source_back_to_terminate():
    state = _state()
    plan = FinalResponsePlan(
        summary="Answer",
        sections=[PlannedAnswerSection(section_type="text", content="Answer", source_refs=["insight:invented"])],
    )

    with pytest.raises(StructuredToolError) as caught:
        asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))

    assert caught.value.recommended_next_action == "terminate"
    assert caught.value.error_type == "final_response_reference_invalid"
    assert caught.value.validation_failure["retry_policy"]["required_action"] == "terminate"


def test_analysis_artifact_requires_explicit_data_view_for_visualization():
    state = _state()
    analysis = AnalysisResult(
        analysis_id="ana_returns", analysis_goal="daily returns", code_hash="sha256:returns",
        input_evidence_id="evi_prices", input_row_count=3, status="succeeded", summary="Computed.",
        result={
            "summary": "Computed.", "metrics": {"maximum_return": 0.25},
            "details": {"diagnostic_rows": [{"value": index} for index in range(100)]},
            "data_views": [{
                "view_id": "returns", "name": "Daily returns", "shape": "timeseries",
                "rows": [
                    {"date": "2026-01-02", "return": 0.2},
                    {"date": "2026-01-03", "return": 0.25},
                ],
                "schema_fields": [{"name": "date", "data_type": "time"}, {"name": "return", "data_type": "number"}],
                "lineage": ["evidence:evi_prices"],
            }],
        },
    )
    state.analysis_artifacts[analysis.analysis_id] = analysis
    plan = FinalResponsePlan(
        summary=analysis.summary,
        sections=[PlannedAnswerSection(section_type="analysis", content=analysis.summary, source_refs=["analysis:ana_returns"])],
    )
    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    assert result["visualizations"] == []
    assert result["references"][0]["source_type"] == "analysis"
    reference = result["references"][0]["evidence"]
    assert len(reference["result"]["details"]["diagnostic_rows"]) == 21
    assert reference["result"]["details"]["diagnostic_rows"][-1] == {"_truncated_item_count": 80}
    assert reference["data_views"][0]["row_count"] == 2
    assert "rows" not in reference["data_views"][0]


def test_format_answer_routes_poisoned_analysis_lineage_back_to_code_interpreter():
    state = _state()
    analysis = AnalysisResult(
        analysis_id="ana_poisoned",
        analysis_goal="daily returns",
        code_hash="sha256:poisoned",
        input_evidence_id="evi_prices",
        input_row_count=3,
        status="succeeded",
        summary="Computed.",
        result={
            "summary": "Computed.",
            "metrics": {},
            "details": {},
            "data_views": [{
                "view_id": "returns",
                "name": "Returns",
                "shape": "timeseries",
                "rows": [{"date": "2026-01-02", "return": 0.2}],
                "lineage": ["evidence:invented_alias"],
            }],
        },
        diagnostics={"executed_code": "result = {'data_views': []}"},
    )
    state.analysis_artifacts[analysis.analysis_id] = analysis
    plan = FinalResponsePlan(
        summary="Computed.",
        sections=[PlannedAnswerSection(
            section_type="analysis",
            content="Computed.",
            source_refs=["view:analysis:ana_poisoned:returns"],
        )],
    )

    with pytest.raises(StructuredToolError) as caught:
        asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))

    failure = caught.value.validation_failure
    assert caught.value.error_type == "analysis_lineage_invalid"
    assert failure["retry_policy"]["required_action"] == "code_interpreter"
    assert failure["repair_contract"]["input_evidence"] == "evi_prices"
    assert failure["repair_contract"]["allowed_lineage_refs"] == ["evidence:evi_prices"]
