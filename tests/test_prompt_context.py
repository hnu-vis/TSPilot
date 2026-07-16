from __future__ import annotations

from prompts.data_agent import DataAgentPromptBuilder
from runtime.request_state import apply_observation, build_conversation_state, build_request_state
from schemas.api import ChatRequest
from schemas.insight import InsightResult
from schemas.tool import ToolObservation
from schemas.visualization import VisualizationPayload
from app.settings import get_settings


class _EvidenceSpec:
    result_target = "evidence"


def test_prompt_builder_summarizes_heavy_context():
    settings = get_settings()
    request = ChatRequest(message="分析季节性")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    full_payload = {
        "evidence_id": "evi_demo",
        "result_type": "timeseries",
        "database": "demo",
        "query_language": "flux",
        "query": "demo",
        "summary": "Loaded 100 points.",
        "data": {
            "points": [{"timestamp": f"2023-01-01T00:{i:02d}:00Z", "value": float(i)} for i in range(100)],
            "rows": [{"timestamp": f"2023-01-01T00:{i:02d}:00Z", "value": float(i)} for i in range(100)],
            "series": [],
            "time_field": "timestamp",
            "value_field": "value",
            "series_name": "value",
            "labels": {},
        },
        "columns": ["timestamp", "value"],
        "metadata": {"database_type": "influxdb"},
        "diagnostics": {},
    }
    apply_observation(request_state, observation, full_payload, _EvidenceSpec())
    request_state.latest_insight = InsightResult(
        insight_id="ins_demo",
        verified_facts=[],
        visualizations=[
            VisualizationPayload(
                visualization_id="viz_demo",
                visualization_type="chart",
                visualization_kind="line",
                renderer="linechart",
                title="Demo",
                summary="Demo chart",
                chart={
                    "x_axis_data": [f"t{i}" for i in range(100)],
                    "series_data": [{"name": "value", "data": list(range(100))}],
                },
            )
        ],
    )
    request_state.visualizations = request_state.latest_insight.visualizations

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)

    assert any(action["action"] == "todowrite" for action in context["available_actions"])
    assert context["execution_state"]["artifacts"]["has_database_evidence"] is True
    assert context["execution_state"]["last_successful_tool"] == "sql_query"
    assert len(context["latest_database_evidence"]["data"]["points"]) <= 8
    assert context["query_history"][0]["query"] == "demo"
    assert context["query_history"][0]["row_count"] == 100
    assert len(context["query_history"][0]["preview"]["rows"]) <= 3
    assert context["visualizations"][0]["chart_summary"]["x_axis_count"] == 100
    assert "chart" not in context["visualizations"][0]


def test_prompt_builder_exposes_sql_observation_details():
    settings = get_settings()
    request = ChatRequest(message="算平均值")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    request_state.observations.append(
        ToolObservation(
            tool_name="sql_query",
            success=True,
            summary="ok",
            payload={
                "evidence_id": "evi_sql",
                "query_language": "sql",
                "query": "SELECT AVG(value) AS avg_value FROM metrics",
                "columns": ["avg_value"],
                "data": {"rows": [{"avg_value": 12.3}, {"avg_value": 13.4}]},
                "diagnostics": {
                    "summary_stats": {"rows_count": 2},
                    "sql_query": {"execution_time_ms": 8},
                    "irrelevant": "hidden",
                },
            },
        )
    )

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    payload = context["latest_observation_summaries"][0]["payload"]

    assert payload["query"] == "SELECT AVG(value) AS avg_value FROM metrics"
    assert payload["query_language"] == "sql"
    assert payload["columns"] == ["avg_value"]
    assert payload["data_preview"]["rows"] == [{"avg_value": 12.3}, {"avg_value": 13.4}]
    assert payload["diagnostics"]["summary_stats"] == {"rows_count": 2}
    assert "irrelevant" not in payload["diagnostics"]


def test_prompt_builder_bounds_long_query_context():
    settings = get_settings()
    request = ChatRequest(message="复杂查询")
    request_state = build_request_state(request, settings)
    conversation_state = build_conversation_state(request, request_state.conversation_id or "conv")
    observation = ToolObservation(tool_name="sql_query", success=True, summary="ok", payload={})
    long_query = "SELECT " + ", ".join(f"{i} AS c{i}" for i in range(2000))
    full_payload = {
        "evidence_id": "evi_long",
        "result_type": "table",
        "database": "demo",
        "query_language": "sql",
        "query": long_query,
        "summary": "x" * 5000,
        "data": {"rows": [{"value": 1.0}]},
        "columns": ["value"],
        "metadata": {"raw_schema": "y" * 5000},
        "diagnostics": {
            "query_trace": {
                "rendered_query": {"query_text": long_query},
                "large": ["z" * 1000 for _ in range(20)],
            }
        },
    }
    apply_observation(request_state, observation, full_payload, _EvidenceSpec())

    context = DataAgentPromptBuilder().build_context(request_state, conversation_state)
    evidence = context["latest_database_evidence"]
    history_item = context["query_history"][0]

    assert len(evidence["query"]) < len(long_query)
    assert "truncated" in evidence["query"]
    assert len(history_item["query"]) < len(long_query)
    assert "truncated" in history_item["metadata"]["raw_schema"]
    assert "truncated_items" in evidence["diagnostics"]["query_trace"]["large"][-1]
