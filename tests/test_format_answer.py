from __future__ import annotations

import asyncio

import pytest

from app.settings import get_settings
from runtime.request_state import build_request_state
from schemas.analysis import AnalysisResult
from schemas.api import ChatRequest
from schemas.database import DatabaseEvidence
from schemas.data_fact import DataFact, FactEvidenceRef
from schemas.output import FinalResponsePlan, PlannedAnswerSection, VisualIntent
from schemas.timeseries import ForecastResult, TimeSeriesPoint
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
        evidence_id="evi_prices",
        result_type="timeseries",
        database="demo",
        query_language="unit",
        query="unit:prices",
        summary="Loaded three prices.",
        data={
            "rows": [
                {"timestamp": "2026-01-01T00:00:00Z", "value": 10.0},
                {"timestamp": "2026-01-02T00:00:00Z", "value": 12.0},
                {"timestamp": "2026-01-03T00:00:00Z", "value": 15.0},
            ],
            "time_field": "timestamp",
            "value_field": "value",
        },
        columns=["timestamp", "value"],
    )
    state.database_evidence_artifacts[evidence.evidence_id] = evidence
    state.latest_database_evidence = evidence
    return state


def test_format_answer_materializes_prose_and_visuals_without_second_llm_call():
    state = _state()
    plan = FinalResponsePlan(
        title="Recent prices",
        summary="The latest value is 15.",
        sections=[PlannedAnswerSection(
            section_type="analysis", heading="Trend", content="Prices rose over the observed period.",
            source_refs=["evidence:evi_prices"],
        )],
        visual_intents=[VisualIntent(
            purpose="show the observed trend", template_id="timeseries.trend", title="Price trend",
            source_refs=["evidence:evi_prices"],
        )],
    )

    result = asyncio.run(FormatAnswerTool(llm=_ForbiddenLlm()).execute(
        FormatAnswerInput(response_plan=plan), request_state=state,
    ))

    assert result["summary"] == plan.summary
    assert result["visualizations"][0]["schema_version"] == "2"
    assert result["visualizations"][0]["template_id"] == "timeseries.trend"
    assert result["references"][0]["source_id"] == "evi_prices"
    assert result["claims"][0]["visualization_ids"] == [result["visualizations"][0]["visualization_id"]]


def test_format_answer_rejects_unknown_grounding_reference():
    state = _state()
    plan = FinalResponsePlan(
        summary="Answer",
        sections=[PlannedAnswerSection(section_type="answer", content="Unsupported.", source_refs=["evidence:invented"])],
    )

    with pytest.raises(ValueError, match="unknown presentation source"):
        asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))


def test_format_answer_rejects_duplicate_primary_views_for_same_purpose():
    state = _state()
    plan = FinalResponsePlan(
        summary="Answer",
        visual_intents=[
            VisualIntent(purpose="trend", template_id="timeseries.trend", title="A", source_refs=["evi_prices"]),
            VisualIntent(purpose="trend", template_id="timeseries.trend", title="B", source_refs=["evi_prices"]),
        ],
    )

    with pytest.raises(ValueError, match="multiple primary"):
        asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))


def test_analysis_result_is_a_first_class_presentation_source():
    state = _state()
    analysis = AnalysisResult(
        analysis_id="ana_distribution",
        analysis_goal="distribution",
        code_hash="sha256:test",
        input_evidence_id="evi_prices",
        input_row_count=3,
        status="succeeded",
        summary="Distribution computed.",
        result={
            "summary": "Distribution computed.",
            "metrics": {},
            "details": {"rows": [{"value": 10.0}, {"value": 12.0}, {"value": 15.0}]},
        },
    )
    state.analysis_artifacts[analysis.analysis_id] = analysis
    plan = FinalResponsePlan(
        summary="Distribution computed.",
        sections=[PlannedAnswerSection(section_type="analysis", content=analysis.summary, source_refs=["analysis:ana_distribution"])],
        visual_intents=[VisualIntent(
            purpose="show distribution", template_id="distribution.histogram", title="Distribution",
            source_refs=["analysis:ana_distribution"], encodings={"value": "value"},
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    assert result["visualizations"][0]["dataset"]["series"][0]["points"]
    assert result["references"][0]["source_type"] == "analysis"


def test_forecast_plan_reads_full_artifact_by_reference():
    state = _state()
    forecast = ForecastResult(
        forecast_id="forecast_evi_prices",
        model_name="unit",
        horizon=2,
        forecast_points=[
            TimeSeriesPoint(timestamp="2026-01-04T00:00:00Z", value=16.0),
            TimeSeriesPoint(timestamp="2026-01-05T00:00:00Z", value=17.0),
        ],
        diagnostics={"coverage": {"input_evidence_refs": ["evi_prices"]}},
    )
    state.forecast_artifacts[forecast.forecast_id] = forecast
    plan = FinalResponsePlan(
        summary="The next values are 16 and 17.",
        visual_intents=[VisualIntent(
            purpose="show history and forecast", template_id="timeseries.forecast", title="Forecast",
            source_refs=["forecast:forecast_evi_prices"],
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    visualization = result["visualizations"][0]
    assert len(visualization["dataset"]["series"]) == 2
    assert len(visualization["dataset"]["series"][1]["points"]) == 2
    assert len(visualization["bindings"]) == 2


def test_standalone_fact_can_be_rendered_as_metric_without_background_query():
    state = _state()
    fact = DataFact(
        fact_id="fact_latest",
        name="Latest price",
        fact_type="point_value",
        statement="Latest price is 15.",
        value=15.0,
        unit="USD",
        method="sql_query",
        evidence_refs=[FactEvidenceRef(source_type="query", source_id="evi_prices")],
    )
    state.fact_set.facts = [fact]
    plan = FinalResponsePlan(
        summary=fact.statement,
        visual_intents=[VisualIntent(
            purpose="show the standalone latest value", template_id="metric.single", title="Latest price",
            source_refs=["fact:fact_latest"], fact_refs=["fact:fact_latest"],
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    assert result["visualizations"][0]["dataset"]["metric"]["value"] == 15.0
