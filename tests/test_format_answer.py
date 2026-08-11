from __future__ import annotations

import asyncio

import pytest

from app.settings import get_settings
from core.visualization import PresentationCatalog
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


def test_planner_inventory_prefers_charts_for_requested_time_series():
    inventory = PresentationCatalog(_state()).planner_inventory()

    assert any(
        "time series" in rule and "table.detail" in rule
        for rule in inventory["rules"]
    )


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


def test_analysis_visualization_discovers_domain_named_row_collection():
    state = _state()
    analysis = AnalysisResult(
        analysis_id="ana_returns",
        analysis_goal="daily returns",
        code_hash="sha256:returns",
        input_evidence_id="evi_prices",
        input_row_count=3,
        status="succeeded",
        summary="Daily returns computed.",
        result={
            "summary": "Daily returns computed.",
            "metrics": {"point_count": 3, "maximum_return": 0.2},
            "details": {
                "daily_series": [
                    {"date": "2026-01-01", "close": 10.0, "return": None},
                    {"date": "2026-01-02", "close": 12.0, "return": 0.2},
                    {"date": "2026-01-03", "close": 15.0, "return": 0.25},
                ],
                "maximum_window": [{"date": "2026-01-03", "return": 0.25}],
            },
        },
    )
    state.analysis_artifacts[analysis.analysis_id] = analysis
    plan = FinalResponsePlan(
        summary=analysis.summary,
        visual_intents=[VisualIntent(
            purpose="show close and return series",
            template_id="timeseries.comparison",
            title="Close and returns",
            source_refs=["analysis:ana_returns"],
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    series = result["visualizations"][0]["dataset"]["series"]

    assert {item["name"] for item in series} == {"close", "return"}
    assert all(len(item["points"]) >= 2 for item in series)
    assert all(point["x"] is not None for item in series for point in item["points"])


def test_analysis_visualization_rejects_numeric_series_encoding():
    state = _state()
    analysis = AnalysisResult(
        analysis_id="ana_numeric_series",
        analysis_goal="daily returns",
        code_hash="sha256:numeric-series",
        input_evidence_id="evi_prices",
        input_row_count=2,
        status="succeeded",
        summary="Daily returns computed.",
        result={
            "summary": "Daily returns computed.",
            "metrics": {},
            "details": {
                "daily_series": [
                    {"date": "2026-01-01", "close": 10.0, "return": 0.1},
                    {"date": "2026-01-02", "close": 12.0, "return": 0.2},
                ],
            },
        },
    )
    state.analysis_artifacts[analysis.analysis_id] = analysis
    plan = FinalResponsePlan(
        summary=analysis.summary,
        visual_intents=[VisualIntent(
            purpose="show close and return",
            template_id="timeseries.comparison",
            title="Close and returns",
            source_refs=["analysis:ana_numeric_series"],
            encodings={"x": "date", "y": "close", "series": "return"},
        )],
    )

    with pytest.raises(ValueError, match="series encoding 'return' is numeric"):
        asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))


def test_derived_analysis_does_not_mix_its_raw_input_evidence_granularity():
    state = _state()
    analysis = AnalysisResult(
        analysis_id="ana_daily",
        analysis_goal="daily close",
        code_hash="sha256:daily",
        input_evidence_id="evi_prices",
        input_row_count=3,
        status="succeeded",
        summary="Daily close computed.",
        result={
            "summary": "Daily close computed.",
            "metrics": {},
            "details": {
                "daily_series": [
                    {"date": "2026-01-01", "daily_close": 10.0},
                    {"date": "2026-01-02", "daily_close": 12.0},
                ],
            },
        },
    )
    state.analysis_artifacts[analysis.analysis_id] = analysis
    plan = FinalResponsePlan(
        summary=analysis.summary,
        visual_intents=[VisualIntent(
            purpose="show daily close",
            template_id="timeseries.trend",
            title="Daily close",
            source_refs=["analysis:ana_daily", "evidence:evi_prices"],
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    series = result["visualizations"][0]["dataset"]["series"]

    assert [(item["name"], len(item["points"])) for item in series] == [("daily_close", 2)]


def test_timeseries_facts_are_materialized_as_primary_series():
    state = _state()
    state.fact_set.facts = [
        DataFact(
            fact_id="fact_close",
            name="daily_close",
            fact_type="timeseries",
            statement="Daily closes.",
            value=[
                {"timestamp": "2026-01-01", "value": 10.0},
                {"timestamp": "2026-01-02", "value": 12.0},
            ],
            method="code_interpreter",
        ),
        DataFact(
            fact_id="fact_return",
            name="daily_return",
            fact_type="timeseries",
            statement="Daily returns.",
            value=[
                {"timestamp": "2026-01-01", "value": 0.1},
                {"timestamp": "2026-01-02", "value": 0.2},
            ],
            method="code_interpreter",
        ),
    ]
    plan = FinalResponsePlan(
        summary="Series computed.",
        visual_intents=[VisualIntent(
            purpose="compare close and return",
            template_id="timeseries.trend",
            title="Close and return",
            fact_refs=["fact:fact_close", "fact:fact_return"],
            encodings={"x": "timestamp", "y": "value"},
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    series = result["visualizations"][0]["dataset"]["series"]

    assert [(item["name"], len(item["points"])) for item in series] == [
        ("daily_close", 2),
        ("daily_return", 2),
    ]


def test_interval_fact_encodings_define_highlight_boundaries():
    state = _state()
    fact = DataFact(
        fact_id="fact_window",
        name="maximum window",
        fact_type="analysis",
        statement="Maximum window.",
        value={"window_start": "2026-01-01", "window_end": "2026-01-03", "deviation": 0.2},
        method="code_interpreter",
    )
    state.fact_set.facts = [fact]
    plan = FinalResponsePlan(
        summary=fact.statement,
        visual_intents=[VisualIntent(
            purpose="highlight maximum window",
            template_id="interval.highlight",
            title="Maximum window",
            source_refs=["evidence:evi_prices"],
            fact_refs=["fact:fact_window"],
            encodings={"x": "timestamp", "y": "value", "start": "window_start", "end": "window_end"},
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    area = next(layer for layer in result["visualizations"][0]["layers"] if layer["kind"] == "area")

    assert [point["x"] for point in area["points"]] == ["2026-01-01", "2026-01-03"]


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


def test_composite_fact_metric_extracts_its_unambiguous_numeric_value():
    state = _state()
    fact = DataFact(
        fact_id="fact_window_std",
        name="Largest rolling standard deviation",
        fact_type="analysis",
        statement="The largest seven-day rolling standard deviation is 0.0357.",
        value={"start_date": "2023-01-14", "end_date": "2023-01-20", "std_dev": 0.0357},
        method="code_interpreter",
    )
    state.fact_set.facts = [fact]
    plan = FinalResponsePlan(
        summary=fact.statement,
        visual_intents=[VisualIntent(
            purpose="show the largest rolling deviation",
            template_id="metric.single",
            title=fact.name,
            fact_refs=["fact:fact_window_std"],
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))
    metric = result["visualizations"][0]["dataset"]["metric"]

    assert metric["value"] == 0.0357
    assert metric["start_date"] == "2023-01-14"
    assert metric["end_date"] == "2023-01-20"


def test_references_are_deduplicated_after_aliases_are_canonicalized():
    state = _state()
    fact = DataFact(
        fact_id="fact_latest",
        name="Latest price",
        fact_type="point_value",
        statement="Latest price is 15.",
        value=15.0,
        method="sql_query",
    )
    state.fact_set.facts = [fact]
    plan = FinalResponsePlan(
        summary=fact.statement,
        sections=[PlannedAnswerSection(
            section_type="analysis",
            content=fact.statement,
            source_refs=["fact_latest", "fact:fact_latest"],
        )],
    )

    result = asyncio.run(FormatAnswerTool().execute(FormatAnswerInput(response_plan=plan), request_state=state))

    assert [(reference["source_type"], reference["source_id"]) for reference in result["references"]] == [
        ("fact", "fact_latest")
    ]
